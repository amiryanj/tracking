import numpy as np
import cv2
from parse_utils import PeTrackParser
from DEFINITIONS import *

scn_nbr = 8
run_nbr = 1

parser = PeTrackParser()
main_dir = 'C:/Users/Javad/Dropbox/PAMELA data/new_cut_video'

# p_leg_red, t_leg_red = parser.load(main_dir + '/S%d/run%d/S%d_run%d_red_height170.txt' % (scn_nbr, run_nbr, scn_nbr, run_nbr))
# p_head_red, t_head_red = parser.load(main_dir + '/S%d/run%d/S%d_run%d_red_height0.txt' % (scn_nbr, run_nbr, scn_nbr, run_nbr))
# p_leg_yellow, t_leg_yellow = parser.load(main_dir + '/S%d/run%d/S%d_run%d_yellow_height170.txt' % (scn_nbr, run_nbr, scn_nbr, run_nbr))
# p_head_yellow, t_head_yellow = parser.load(main_dir + '/S%d/run%d/S%d_run%d_yellow_height0.txt' % (scn_nbr, run_nbr, scn_nbr, run_nbr))
# t_data_red = t_leg_red
# t_data_yellow = t_head_yellow
# print('#red =', len(t_data_red), ' #yellow =', len(t_data_yellow))
# p_legs = list() + p_leg_red + p_leg_yellow
# p_heads = list() + p_head_red + p_head_yellow
# t_data = list() + t_data_red + t_data_yellow

# assert len(t_leg_red) == len(t_head_red)
# assert len(t_leg_yellow) == len(t_head_yellow)
# for ii, Ti in enumerate(t_leg_red):
#     assert len(Ti) == len(t_head_red[ii]), "mismatch - reds - %d" % ii
# for ii, Ti in enumerate(t_leg_yellow):
#     # print('# legs[%d]= ' % ii , len(t_leg_yellow[ii]))
#     # print('# heads[%d]= ' % ii , len(t_head_yellow[ii]))
#     assert len(Ti) == len(t_head_yellow[ii]), "mismatch - yellows - %d" % ii

main_dir = '/home/cyrus/Dropbox/PAMELA data/new_cut_video'
p_heads, t_head = parser.load(main_dir + '/S%d/run%d/S%d_run%d-heads.txt' % (scn_nbr, run_nbr, scn_nbr, run_nbr))
p_legs, t_leg = parser.load(main_dir + '/S%d/run%d/S%d_run%d-legs.txt' % (scn_nbr, run_nbr, scn_nbr, run_nbr))
t_data = t_head

p_legs_out = p_legs.copy()
p_heads_out = p_heads.copy()
t_out = t_data.copy()

cap = cv2.VideoCapture(main_dir + '/S%d/S%d_run%d0001-100000-undistort.mp4' %(scn_nbr, scn_nbr, run_nbr))
output_heads = main_dir + '/S%d/run%d/S%d_run%d-heads.txt' % (scn_nbr, run_nbr, scn_nbr, run_nbr)
output_legs = main_dir + '/S%d/run%d/S%d_run%d-legs.txt' % (scn_nbr, run_nbr, scn_nbr, run_nbr)
output_confirm_ids = main_dir + '/S%d/run%d/S%d_run%d-confirmed-ids.txt' % (scn_nbr, run_nbr, scn_nbr, run_nbr)

vwr = 0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# vwr = cv2.VideoWriter(main_dir + '/(1)-output-red.mp4', fourcc, 30, (1600, 1200))

def goto(frame):
    global frame_id, raw_frame
    frame_id = max(frame - 1, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, raw_frame = cap.read()


def checkPointCircle(point, center, radius=5):
    return np.linalg.norm(point - center) < radius

selected_ids = []
confirmed_ids = []
def click_in(event, x, y, flags, param):
    # grab references to the global variables
    global selected_ids, confirmed_ids
    global ped_inds_t_in, time_inds_t_in

    # if the left mouse button was clicked, select the closest head id
    # if SHIFT_KEY is hold , add them into a list()
    if event == cv2.EVENT_LBUTTONDOWN:
        print('click [%d, %d], flags = ' %(x, y), flags)
        if flags & cv2.EVENT_FLAG_SHIFTKEY == 0 and flags & cv2.EVENT_FLAG_CTRLKEY == 0:  # if shift is not hold clean the list
            selected_ids = []
        for ii, p_ind in enumerate(ped_inds_t_out):
            t_ind = time_inds_t_out[ii]
            pi_head_i = p_heads_out[p_ind][t_ind]
            if checkPointCircle(np.array([x, y]), pi_head_i, 6):
                if p_ind not in selected_ids:
                    selected_ids.append(p_ind)
                else:
                    selected_ids.remove(p_ind)
                if flags & cv2.EVENT_FLAG_CTRLKEY:
                    if p_ind not in confirmed_ids:
                        confirmed_ids.append(p_ind)
                    else:
                        confirmed_ids.remove(p_ind)
                # print('ID =  %d , frames =' % p_ind, t_data[p_ind])
                # break
        confirmed_ids.sort()
    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags < 0:
            goto(frame_id - 9)
        else:
            goto(frame_id + 11)


def validCount():
    count = 0
    for ii, Ti in enumerate(t_out):
        if len(Ti) != 0:
            count += 1
    return count


# t means frame_index everywhere
def merge(indices):
    # find maximum timespan of all indices
    t0, t1 = [],  []
    for kk, ind_k in enumerate(indices):
        t0.append(min(t_data[ind_k]))
        t1.append(max(t_data[ind_k]))
    t0, t1 = min(t0), max(t1)

    # TODO everything for head and leg
    merged_t = []
    merged_p_head = np.zeros((t1+1 - t0, 2), dtype=np.float64)
    merged_p_leg = np.zeros((t1 + 1 - t0, 2), dtype=np.float64)

    # merging (+averaging if found parallel tracks)
    for t in range(t0, t1+1):
        num_contributions = 0
        for kk, ind_k in enumerate(indices):
            if t in t_data[ind_k]:
                ind_t = t_data[ind_k].index(t)
                merged_p_leg[t-t0] += p_legs[ind_k][ind_t]
                merged_p_head[t - t0] += p_heads[ind_k][ind_t]
                num_contributions += 1
        if num_contributions > 0:
            merged_p_leg[t-t0] /= num_contributions
            merged_p_head[t - t0] /= num_contributions
            merged_t.append(t)

    # interpolation
    for t in range(t0, t1 + 1):  # FIXME => debug here
        if t in merged_t:
            last_t_detected = t
            last_t_index = merged_t.index(t)
        else:
            t_A = last_t_detected
            t_B = merged_t[last_t_index+1]
            merged_p_leg[t-t0] = (t_B - t) / (t_B - t_A) * merged_p_leg[t_A - t0] +\
                                 (t - t_A) / (t_B - t_A) * merged_p_leg[t_B - t0]
            merged_p_head[t - t0] = (t_B - t) / (t_B - t_A) * merged_p_head[t_A - t0] +\
                                    (t - t_A) / (t_B - t_A) * merged_p_head[t_B - t0]

    merged_t = [i for i in range(t0, t1+1)]
    return merged_t, merged_p_head, merged_p_leg


pause = False
frame_id = -1
ped_inds_t_in, time_inds_t_in = [], []
cv2.namedWindow('frame_in', cv2.WINDOW_NORMAL)
cv2.namedWindow('frame_out', cv2.WINDOW_NORMAL)
cv2.setMouseCallback("frame_in", click_in)
cv2.setMouseCallback("frame_out", click_in)
raw_frame = 0

while True:
    if not pause:
        # Capture frame-by-frame
        frame_id += 1
        ret, raw_frame = cap.read()
        if not ret:
            print("video finished or broken!")
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
        # pi_leg = p_legs[p_ind][t_ind]

        pi_head = p_heads[p_ind][t_ind]
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
        # pi_leg = p_legs[p_ind][t_ind]

        pi_head = p_heads_out[p_ind][t_ind]  # FIXME
        if pi_head[0] > frame_in.shape[1]: continue
        # cv2.circle(frame, (int(pi_leg[0]), int(pi_leg[1])), 5, RED_COLOR, 2)
        cv2.circle(frame_out, (int(pi_head[0]), int(pi_head[1])), 6, GREEN_COLOR, -1)

        cv2.putText(frame_out, '%d' % p_ind, (int(pi_head[0]), int(pi_head[1])),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.6, LIGHT_RED_COLOR, 1)
    cv2.putText(frame_out, '# %d' % len(ped_inds_t_out), (30, 400),
                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3, RED_COLOR, 5)
    cv2.putText(frame_out, '# %d' % len(confirmed_ids), (30, 600),
                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3, BLUE_COLOR, 5)
    # =========================================================

    # =================== Show selected ids ===================
    for kk, selected_id in enumerate(selected_ids):
        if selected_id <len(t_data) and frame_id in t_data[selected_id]:
            t_ind = t_data[selected_id].index(frame_id)
            p_selected_head = p_heads[selected_id][t_ind]
            cv2.circle(frame_in, (int(p_selected_head[0]), int(p_selected_head[1])), 9, LIGHT_BLUE_COLOR, -1)
            cv2.circle(frame_out, (int(p_selected_head[0]), int(p_selected_head[1])), 9, LIGHT_BLUE_COLOR, -1)

            cv2.rectangle(frame_out, (int(p_selected_head[0]) - 20, int(p_selected_head[1]) - 20),
                          (int(p_selected_head[0]) + 20, int(p_selected_head[1]) + 20),
                          MAGENTA_COLOR, 3)

    for kk, confirmed_id in enumerate(confirmed_ids):
        if frame_id in t_out[confirmed_id]:
            t_ind = t_out[confirmed_id].index(frame_id)
            p_confirmed_head = p_heads_out[confirmed_id][t_ind]
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
    cv2.imshow('frame_in', frame_in)
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
            print('less than 2 ped are selected to merge!')

        else:
            merged_t, merged_p_head, merged_p_leg = merge(selected_ids)
            for kk, ind_k in enumerate(selected_ids):
                t_out[ind_k] = []
                p_heads_out[ind_k] = []
                p_legs_out[ind_k] = []
            t_out.append(merged_t)
            p_heads_out.append(merged_p_head)
            p_legs_out.append(merged_p_leg)
            confirmed_ids.append(len(t_out)-1)

    elif key == DELETE_KEY:
        for kk, ind_k in enumerate(selected_ids):
            t_out[ind_k] = []
            p_heads_out[ind_k] = []
            p_legs_out[ind_k] = []

    elif key & 0xFF == ord('r'):
        for ii, Ti in enumerate(t_data):
            if len(Ti) < 5:
                t_out[ii] = []
                p_heads_out[ii] = []
                p_legs_out[ii] = []

    elif key & 0xFF == ord('p'):  # print
        parser.heigth = 1.7
        parser.save(output_heads, p_heads_out[1:], t_out[1:])
        parser.heigth = 0
        parser.save(output_legs, p_legs_out[1:], t_out[1:])
        print('confirmed_ids {%d}= ' % len(confirmed_ids),  confirmed_ids)
        with open(output_confirm_ids, 'w') as confirm_file:
            confirm_file.write('# %d\n' % len(confirmed_ids))
            confirm_file.write('# Robot = %d\n' % confirmed_ids[0])
            confirm_file.write(str(confirmed_ids))



    if vwr is not 0:
        vwr.write(frame_in)


# When everything done, release the capture
cap.release()
if vwr is not 0:
    vwr.release()
cv2.destroyAllWindows()

