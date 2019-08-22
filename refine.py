import numpy as np
import cv2
from parse_utils import PeTrackParser

parser = PeTrackParser()
# p_data, t_data = parser.load('/home/cyrus/Dropbox/PAMELA data/run1/S1_run1_red.txt')
p_data, t_data = parser.load('/home/cyrus/Dropbox/PAMELA data/run1/S1_run1_yellow.txt')
print(len(p_data[0]))

cap = cv2.VideoCapture('/home/cyrus/Dropbox/PAMELA data/run1/(1)-undistort.mp4')
frame_id = 0

while(True):

    # print('frame = ', frame_id)
    ped_inds_t = []
    time_inds_t = []
    for ind, Ti in enumerate(t_data):
        if frame_id in Ti:
            # print(ind)
            Ti_list = list(Ti)
            time_inds_t.append(Ti_list.index(frame_id))
            ped_inds_t.append(ind)
    # print('******************')

    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_id += 1

    for ii, p_ind in enumerate(ped_inds_t):
        t_ind = time_inds_t[ii]
        pi = p_data[p_ind][t_ind]
        pi[0] = pi[0] * 12000
        pi[1] = - pi[1] * 12000
        print(pi)
        cv2.circle(frame, (int(pi[0]), int(pi[1])), 5, (100, 0, 200), 2)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()