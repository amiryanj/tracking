import os
import cv2
import numpy as np
from parse_utils import PeTrackParser
from DEFINITIONS import *
from linear_models import MyKalman

scn_nbr = 5
run_nbr = 5
parser = PeTrackParser()
main_dir = 'C:/Users/Javad/Dropbox/PAMELA data/new_cut_video'
# main_dir = '/home/cyrus/Dropbox/PAMELA data/new_cut_video'

output_file = main_dir + '/S%d/run%d/S%d_run%d-kalman.txt' % (scn_nbr, run_nbr, scn_nbr, run_nbr)
# if os.path.exists(output_file):
#     p_data, t_data, ids = parser.load(output_file)
# else:
p_data, t_data, ids = parser.load(main_dir + '/S%d/run%d/S%d_run%d-new.txt' % (scn_nbr, run_nbr, scn_nbr, run_nbr))

# p_legs = [x[:, :3] for x in p_data]
# p_heads = [x[:, 3:] for x in p_data]

p_out = p_data
t_out = t_data

# ======== Kalman Filter ===============
# fps = 12.5
# p_heads_filtered = []
# for x in p_data:
#     kalman = MyKalman(12.5/fps)
#     filtered_pos, filtered_vel = kalman.filter(x[:, 3:5])
#     smoothed_pos, smoothed_vel = kalman.smooth(x[:, 3:5])
#     p_heads_filtered.append(smoothed_pos)
#     x[:, 3:5] = smoothed_pos
# p_heads = p_heads_filtered.copy()

# =============== Homography ===========
Homog = np.eye(3, dtype=np.float)
homog_file = main_dir + '/homog.txt'
if os.path.exists(homog_file):
    Homog = np.loadtxt(homog_file, dtype=float)


cap = cv2.VideoCapture(main_dir + '/S%d/S%d_run%d0001-100000-undistort.mp4' %(scn_nbr, scn_nbr, run_nbr))

vwr = 0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# vwr = cv2.VideoWriter(main_dir + '/(1)-output-red.mp4', fourcc, 30, (1600, 1200))


def headToFoot(x, Homog):
    # xHomogenous = np.hstack((x, np.ones((x.shape[0], 1))))
    xHomogenous = np.hstack((x[:2], np.ones(1)))
    if xHomogenous.ndim > 1:
        x_tr = np.transpose(xHomogenous)
        x_tr = np.matmul(Homog, x_tr)  # to camera frame
        xXYZ = np.transpose(x_tr / x_tr[2])  # to pixels (from millimeters)
        return xXYZ[:, :2].astype(int)
    else:
        xHomogenous = np.dot(Homog, xHomogenous)  # to camera frame
        xXYZ = xHomogenous / xHomogenous[2]  # to pixels (from millimeters)
        return xXYZ[:2].astype(int)


def goto(frame):
    global frame_id, raw_frame
    frame_id = max(frame - 1, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, try_frame = cap.read()
    if ret:
        raw_frame = try_frame


def checkPointCircle(point, center, radius=5):
    return np.linalg.norm(point[:2] - center[:2]) < radius


# t means frame_index everywhere
def merge(indices):
    # find maximum timespan of all indices
    t0, t1 = [],  []
    for kk, ind_k in enumerate(indices):
        t0.append(min(t_data[ind_k]))
        t1.append(max(t_data[ind_k]))
    t0, t1 = min(t0), max(t1)

    merged_t = []
    merged_p = np.zeros((t1+1 - t0, 6), dtype=np.float64)

    # merging (+averaging if found parallel tracks)
    for t in range(t0, t1+1):
        num_contributions = 0
        for kk, ind_k in enumerate(indices):
            if t in t_data[ind_k]:
                ind_t = t_data[ind_k].index(t)
                merged_p[t - t0] += p_data[ind_k][ind_t]
                num_contributions += 1
        if num_contributions > 0:
            merged_p[t - t0] /= num_contributions
            merged_t.append(t)

    # interpolation
    for t in range(t0, t1 + 1):  # FIXME => debug here
        if t in merged_t:
            last_t_detected = t
            last_t_index = merged_t.index(t)
        else:
            t_A = last_t_detected
            t_B = merged_t[last_t_index+1]
            merged_p[t-t0] = (t_B - t) / (t_B - t_A) * merged_p[t_A - t0] +\
                             (t - t_A) / (t_B - t_A) * merged_p[t_B - t0]

    merged_t = [i for i in range(t0, t1+1)]
    return merged_t, merged_p


def repair(pid, new_p, cur_t):
    try:
        ind_t = t_data[pid].index(cur_t)  # should always exist
    except ValueError:
        if t_data[pid][0] - 150 < cur_t < t_data[pid][0]:  # up to 30 frame before pid's first frame can add data
            gap_size = t_data[pid][0] - cur_t
            prepend_ps = np.zeros((gap_size, 6), dtype=p_data[pid].dtype)
            for tt in range(cur_t, t_data[pid][0]):
                prepend_ps[tt-cur_t][3:5] = new_p * float(t_data[pid][0] - tt) / gap_size + \
                                            p_data[pid][0][3:5] * float(tt - cur_t) / gap_size
            p_data[pid] = np.concatenate([prepend_ps, p_data[pid]])
            t_data[pid] = [tt for tt in range(cur_t, t_data[pid][0])] + t_data[pid]
            ind_t = 0
        elif t_data[pid][-1] < cur_t < t_data[pid][-1] + 150:

            gap_size = cur_t - t_data[pid][-1]
            append_ps = np.zeros((gap_size, 6), dtype=p_data[pid].dtype)
            for tt in range(t_data[pid][-1]+1, cur_t+1):
                append_ps[tt - t_data[pid][-1]-1][3:5] = new_p * float(tt - t_data[pid][-1]) / gap_size + \
                                                         p_data[pid][-1][3:5] * float(cur_t - tt) / gap_size
            p_data[pid] = np.concatenate([p_data[pid], append_ps])
            t_data[pid] = t_data[pid] + [tt for tt in range(t_data[pid][-1]+1, cur_t+1)]
            ind_t = t_data[pid][-1]
        return

    p_data[pid][ind_t][3:5] = new_p

    repair_gap = 6
    if ind_t >= repair_gap:
        for tt in range(ind_t-repair_gap, ind_t):
            p_data[pid][tt][3:5] = new_p * float(tt+repair_gap-ind_t) / repair_gap +\
                                   p_data[pid][ind_t-repair_gap][3:5] * float(ind_t-tt) / repair_gap
    if ind_t < len(t_data[pid]) - repair_gap:
        for tt in range(ind_t, ind_t + repair_gap):
            p_data[pid][tt][3:5] = new_p * float(repair_gap-tt+ind_t) / repair_gap +\
                                   p_data[pid][ind_t+repair_gap][3:5] * float(tt-ind_t) / repair_gap



selected_ids = []
confirmed_ids = []
def click_in(event, x, y, flags, param):
    # grab references to the global variables
    global selected_ids, confirmed_ids
    global ped_inds_t_in, time_inds_t_in

    # if the left mouse button was clicked, select the closest head id
    # if SHIFT_KEY is hold , add them into a list()
    if event == cv2.EVENT_LBUTTONDOWN:
        # print('click [%d, %d], flags = ' %(x, y), flags)
        if flags & cv2.EVENT_FLAG_ALTKEY and len(selected_ids) == 1:
            repair(selected_ids[0], np.array([x,y]).astype(float), frame_id)
            return
        if flags & cv2.EVENT_FLAG_CTRLKEY:
            if len(selected_ids) != 0:
                print('%.1f, %.1f, %d, %d' % (p_data[selected_ids[0]][frame_id][3],
                                              p_data[selected_ids[0]][frame_id][4],
                                              x, y))
            return

        if flags & cv2.EVENT_FLAG_SHIFTKEY == 0 and flags & cv2.EVENT_FLAG_ALTKEY == 0:  # if shift is not hold clean the list
            selected_ids = []
        for ii, p_ind in enumerate(ped_inds_t_out):
            t_ind = time_inds_t_out[ii]
            pi_head_i = p_out[p_ind][t_ind][3:5]
            if checkPointCircle(np.array([x, y]), pi_head_i, 6):
                if p_ind not in selected_ids:
                    selected_ids.append(p_ind)
                else:
                    selected_ids.remove(p_ind)

        confirmed_ids.sort()
    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags < 0:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                goto(frame_id)
            else:
                goto(frame_id - 9)
        else:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                goto(frame_id + 2)
            else:
                goto(frame_id + 11)


def validCount():
    count = 0
    for ii, Ti in enumerate(t_out):
        if len(Ti) != 0:
            count += 1
    return count




pause = False
frame_id = -1
ped_inds_t_in, time_inds_t_in = [], []
# cv2.namedWindow('frame_in', cv2.WINDOW_NORMAL)
cv2.namedWindow('frame_out', cv2.WINDOW_NORMAL)
# cv2.setMouseCallback("frame_in", click_in)
cv2.setMouseCallback("frame_out", click_in)
raw_frame = 0

while True:
    if not pause:
        # Capture frame-by-frame
        frame_id += 1
        ret, raw_frame = cap.read()
        if not ret:
            # print("video finished or broken!")
            frame_id -=1
            raw_frame = frame_in.copy()
    frame_in = np.copy(raw_frame)
    frame_out = np.copy(raw_frame)
    cv2.putText(frame_in, str(frame_id), (40, 140), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 4, (200, 30, 100), 2)

    # input data # ============================================
    ped_inds_t_in = []
    time_inds_t_in = []
    for ind, Ti in enumerate(t_data):
        if frame_id in Ti:
            time_inds_t_in.append(Ti.index(frame_id))
            ped_inds_t_in.append(ind)

    for ii, p_ind in enumerate(ped_inds_t_in):
        t_ind = time_inds_t_in[ii]
        pi_head = p_data[p_ind][t_ind][3:5]
        if pi_head[0] > frame_in.shape[1]: continue

        cv2.circle(frame_in, (int(pi_head[0]), int(pi_head[1])), 6, GREEN_COLOR, -1)
        cv2.putText(frame_in, str(p_ind), (int(pi_head[0]), int(pi_head[1])),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.7, LIGHT_RED_COLOR, 2)
    cv2.putText(frame_in, '# %d' % len(ped_inds_t_in), (30, 400),
                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3, RED_COLOR, 5)
    # =========================================================

    # output data # ===========================================
    ped_inds_t_out = []
    time_inds_t_out = []
    for ind, Ti in enumerate(t_out):
        if frame_id in Ti:
            time_inds_t_out.append(Ti.index(frame_id))
            ped_inds_t_out.append(ind)

    for ii, p_ind in enumerate(ped_inds_t_out):
        t_ind = time_inds_t_out[ii]
        pi_head = p_out[p_ind][t_ind][3:5]
        if pi_head[0] > frame_in.shape[1]: continue

        cv2.circle(frame_out, (int(pi_head[0]), int(pi_head[1])), 6, GREEN_COLOR, -1)
        cv2.putText(frame_out, '%d' % p_ind, (int(pi_head[0]), int(pi_head[1])),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.8, LIGHT_RED_COLOR, 2)

        pi_foot = headToFoot(pi_head, Homog)
        cv2.circle(frame_out, (int(pi_foot[0]), int(pi_foot[1])), 9, BLUE_COLOR, 3)

    cv2.putText(frame_out, '# %d' % len(ped_inds_t_out), (30, 400),
                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3, RED_COLOR, 5)
    cv2.putText(frame_out, '# %d' % len(confirmed_ids), (30, 600),
                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3, BLUE_COLOR, 5)
    # =========================================================

    # =================== Show selected ids ===================
    for kk, selected_id in enumerate(selected_ids):
        if selected_id <len(t_data) and frame_id in t_data[selected_id]:
            t_ind = t_data[selected_id].index(frame_id)
            p_selected_head = p_data[selected_id][t_ind][3:5]
            cv2.circle(frame_in, (int(p_selected_head[0]), int(p_selected_head[1])), 9, LIGHT_BLUE_COLOR, -1)
            cv2.circle(frame_out, (int(p_selected_head[0]), int(p_selected_head[1])), 9, LIGHT_BLUE_COLOR, -1)

            cv2.rectangle(frame_out, (int(p_selected_head[0]) - 20, int(p_selected_head[1]) - 20),
                          (int(p_selected_head[0]) + 20, int(p_selected_head[1]) + 20),
                          MAGENTA_COLOR, 3)

    for kk, confirmed_id in enumerate(confirmed_ids):
        if frame_id in t_out[confirmed_id]:
            t_ind = t_out[confirmed_id].index(frame_id)
            p_confirmed_head = p_out[confirmed_id][t_ind][3:5]
            cv2.rectangle(frame_out, (int(p_confirmed_head[0]) - 15, int(p_confirmed_head[1]) - 15),
                          (int(p_confirmed_head[0]) + 15, int(p_confirmed_head[1]) + 15),
                          AQUAMARINE_COLOR, 3)

    cv2.putText(frame_out, 'Scenario #%d Run #%d Frame %d' %(scn_nbr, run_nbr, frame_id), (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, LIGHT_RED_COLOR, 2)
    cv2.putText(frame_out, 'selected IDs=%s' %selected_ids, (30, 200),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, LIGHT_BLUE_COLOR, 2)
    cv2.putText(frame_out, 'Total Out= %d' % validCount(), (30, 1000),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, MAGENTA_COLOR, 5)
    cv2.putText(frame_out, 'confirmed IDs=%s' % confirmed_ids, (30, 1130),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, LIGHT_GREEN_COLOR, 2)
    # =========================================================

    # Display the resulting frame
    # cv2.imshow('frame_in', frame_in)
    cv2.imshow('frame_out', frame_out)
    key = cv2.waitKeyEx(30)
    if key != -1:
        # print(key)
        pass
    if key & 0xFF == ord('q'):
        break
    elif key == 32:  # space
        pause = not pause
    elif key == RIGHT_ARROW_KEY:
        pause = True
        if frame_id < int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1:
            frame_id += 1
            ret, raw_frame = cap.read()
    elif key == LEFT_ARROW_KEY:
        pause = True
        goto(frame_id)
        # frame_id = max(frame_id - 1, 0)
        # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        # ret, raw_frame = cap.read()
    elif key == UP_ARROW_KEY:
        pause = True
        frame_id = max(frame_id - 10, 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, raw_frame = cap.read()
    elif key == DOWN_ARROW_KEY:
        pause = True
        frame_id = min(frame_id + 10, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, raw_frame = cap.read()
    elif key == HOME_KEY:
        pause = True
        frame_id = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, raw_frame = cap.read()

    elif key & 0xFF == ord('m'):
        if len(selected_ids) < 1:
            # print('less than 2 ped are selected to merge!')
            pass

        else:
            merged_ts, merged_ps = merge(selected_ids)
            for kk, ind_k in enumerate(selected_ids):
                t_out[ind_k] = []
                p_out[ind_k] = []
            t_out.append(merged_ts)
            p_out.append(merged_ps)
            confirmed_ids.append(len(t_out)-1)

    elif key == DELETE_KEY:
        for kk, ind_k in enumerate(selected_ids):
            t_out[ind_k] = []
            p_out[ind_k] = []

    elif key & 0xFF == ord('r'):
        for ii, Ti in enumerate(t_data):
            if len(Ti) < 5:
                t_out[ii] = []
                p_out[ii] = []

    elif key & 0xFF == ord('p'):  # print
        # parser.save(output_file, p_out[confirmed_ids[0]:], t_out[confirmed_ids[0]:])
        parser.save(output_file, p_out, t_out)
        # print('confirmed_ids {%d}= ' % len(confirmed_ids),  confirmed_ids)


    if vwr is not 0:
        vwr.write(frame_in)


# When everything done, release the capture
cap.release()
if vwr is not 0:
    vwr.release()
cv2.destroyAllWindows()

