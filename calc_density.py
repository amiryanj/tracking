import os
import cv2
import numpy as np
from parse_utils import PeTrackParser
from scipy.spatial import Voronoi, voronoi_plot_2d
from DEFINITIONS import *
import matplotlib.pyplot as plt
from voronoi import voronoi_finite_polygons_2d, clip, PolyArea
from linear_models import MyKalman

scn_nbr = 1
run_nbr = 1
parser = PeTrackParser()
# main_dir = '/home/cyrus/Dropbox/PAMELA data/new_cut_video'
main_dir = 'C:/Users/Javad/Dropbox/PAMELA data/new_cut_video'
cap = cv2.VideoCapture(main_dir + '/S%d/S%d_run%d0001-100000-undistort.mp4' % (scn_nbr, scn_nbr, run_nbr))
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
    # im = np.ones((1200, 1600, 3), np.uint8) * 255
    points = []
    found_inds = []
    for ii, Ti in enumerate(t_data):
        if len(Ti) == 0 or t < Ti[0] or t > Ti[-1]: continue
        found_inds.append(ii)
        t_ind = Ti.index(t)
        head_i_t = p_data[ii][t][3:5].astype(int)
        points.append(head_i_t)

    vor = Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    rect_all = np.array([[200, 0], [1000, 0], [1000, 1000], [200, 1000]])
    rect_up = np.array([[655, 300], [963, 284], [985, 600], [836, 600], [836, 1000], [695, 1000], [695, 600], [622, 603]])
    rect_down = np.array([[593, 1020], [1000, 1050], [985, 600], [836, 600], [836, 200], [695, 200], [695, 600], [622, 603]])
    # cv2.fillPoly(im, [rect_up.astype(int)], RED_COLOR)
    # cv2.fillPoly(im, [rect_down.astype(int)], BLUE_COLOR)

    all_inds = []
    for ii, region in enumerate(regions):
        polygon = vertices[region]
        # polygon = clip(polygon, rect_poly)
        # if points[ii][1] < 600:  # up
        #     polygon = clip(rect_up, polygon)

        # else:
        #     polygon = clip(rect_down, polygon)
        # if len(polygon) == 0: continue

        all_inds.append(ii)

        polygon = np.stack(polygon)
        cv2.fillPoly(im, [polygon.astype(int)], (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))


        x = np.arange(0, 1, 0.001)
        y = np.sqrt(1 - x ** 2)
        area = PolyArea(polygon)
        # print('[%d] = %d ' %(ii, area))
        # print(polygon, '\n****************')
        # cv2.putText(im, '%d' %area, )

    print(all_inds)

    for ii, pnt in enumerate(points):
        cv2.circle(im, (int(pnt[0]), int(pnt[1])), 5, (100, 0, 0), -1)
        cv2.putText(im, '%d' % ii, (pnt[0], pnt[1]),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.8, RED_COLOR, 2)

    # im = np.flipud(im)
    cv2.imshow('im', im)
    key = cv2.waitKeyEx(30)
    if key & 0xFF == ord('q'):
        break
    elif key == 32:  # space
        pause = not pause

    # voronoi_plot_2d(vor)
    # plt.show()



peds = []
K = 3
for ped_i in peds:
    pass