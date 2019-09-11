import numpy as np
import cv2
import csv

head_pts = []  # np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
foot_pts = []  # np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

main_dir = 'C:/Users/Javad/Dropbox/PAMELA data/new_cut_video'

csv_file = main_dir + '/head_foot_points.txt'
out_file = main_dir + '/homog.txt'

with open(csv_file, 'r') as data_file:
    csv_reader = csv.reader(data_file, delimiter=',')

    for i, tokens in enumerate(csv_reader):
        if tokens[0].startswith('#') or len(tokens) < 4: continue

        head_x = float(tokens[0])
        head_y = float(tokens[1])
        foot_x = float(tokens[2])
        foot_y = float(tokens[3])
        head_pts.append([head_x, head_y])
        foot_pts.append([foot_x, foot_y])

head_pts = np.asarray(head_pts).astype(np.float32)
foot_pts = np.asarray(foot_pts).astype(np.float32)
M, mask = cv2.findHomography(head_pts, foot_pts, cv2.RANSAC, 5.0)
print(M)
np.savetxt(out_file, M, fmt='%.6f')
