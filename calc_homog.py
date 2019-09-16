import numpy as np
import cv2
import csv
import os
from parse_utils import PeTrackParser
from DEFINITIONS import *


def headToFoot(x, Homog):
    if x.ndim == 1:
        xHomogenous = np.hstack((x[:2], np.ones(1)))
        x_tr = np.matmul(Homog, xHomogenous.transpose())  # to camera frame
        xXYZ = np.transpose(x_tr / x_tr[2])  # to pixels (from millimeters)
        return xXYZ[:2]
    elif x.ndim == 2:
        xHomogenous = np.hstack((x[:, :2], np.ones((x.shape[0], 1))))
        x_tr = np.matmul(Homog, xHomogenous.transpose())  # to camera frame
        xXYZ = np.transpose(x_tr / x_tr[2])  # to pixels (from millimeters)

        # xHomogenous = Homog * xHomogenous  # to camera frame
        # xXYZ = xHomogenous / xHomogenous[2]  # to pixels (from millimeters)
        return xXYZ[:, :2]
    else:
        raise ValueError('x should be 1d or 2d')


if __name__ == '__main__':
    head_pts = []  # np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    foot_pts = []  # np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    main_dir = 'C:/Users/Javad/Dropbox/PAMELA data/new_cut_video'

    # csv_file = main_dir + '/head_foot_points.txt'
    # out_file = main_dir + '/homog.txt'

    csv_file = main_dir + '/head_foot_points-robot.txt'
    out_file = main_dir + '/homog-robot.txt'


    with open(csv_file, 'r') as data_file:
        csv_reader = csv.reader(data_file, delimiter=',')

        for i, tokens in enumerate(csv_reader):
            if len(tokens) < 4 or tokens[0].startswith('#'): continue

            head_x = float(tokens[0])
            head_y = float(tokens[1])
            foot_x = float(tokens[2])
            foot_y = float(tokens[3])
            head_pts.append([head_x, head_y])
            foot_pts.append([foot_x, foot_y])

    head_pts = np.asarray(head_pts).astype(np.float32)
    foot_pts = np.asarray(foot_pts).astype(np.float32)
    M, mask = cv2.findHomography(head_pts, foot_pts, cv2.RANSAC, 5.0)
    # M, mask = cv2.findHomography(head_pts, foot_pts, cv2.LMEDS, 5.0)

    print(M)
    np.savetxt(out_file, M, fmt='%.6f')

    Homog = np.eye(3, dtype=np.float)
    homog_file = out_file
    if os.path.exists(homog_file):
        Homog = np.loadtxt(homog_file, dtype=float)


    scn_nbr = 8
    run_nbr = 4
    parser = PeTrackParser()
    cap = cv2.VideoCapture(main_dir + '/S%d/S%d_run%d0001-100000-undistort.mp4' %(scn_nbr, scn_nbr, run_nbr))
    p_data, t_data, ids = parser.load(main_dir + '/S%d/run%d/S%d_run%d-new.txt' % (scn_nbr, run_nbr, scn_nbr, run_nbr))

    max_t = -1
    for Ti in t_data:
        max_t = max(max(Ti), max_t)

    cv2.namedWindow('im', cv2.WINDOW_NORMAL)
    t = -1
    pause = False
    while True:
        if t < max_t and not pause:
            t += 1
            ret, im_raw = cap.read()
        im = np.copy(im_raw)

        head_locs = []
        foot_locs = []
        for ii, Ti in enumerate(t_data):
            if len(Ti) == 0 or t < Ti[0] or t > Ti[-1]: continue
            t_ind = Ti.index(t)
            head_i_t = p_data[ii][t_ind][3:5].astype(int)
            head_locs.append(head_i_t)
            foot_i_t = headToFoot(head_i_t, Homog)
            foot_locs.append(foot_i_t)
            cv2.circle(im, (int(head_i_t[0]), int(head_i_t[1])), 5, RED_COLOR, 2)
            cv2.circle(im, (int(foot_i_t[0]), int(foot_i_t[1])), 15, GREEN_COLOR, 2)

        cv2.imshow('im', im)
        key = cv2.waitKeyEx(20)
        if key & 0xFF == ord('q'):
            break
        elif key == 32:  # space
            pause = not pause
