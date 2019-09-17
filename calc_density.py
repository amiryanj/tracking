import os
import cv2
import numpy as np
from parse_utils import PeTrackParser
from scipy.spatial import Voronoi, voronoi_plot_2d
from DEFINITIONS import *
import matplotlib.pyplot as plt
from voronoi import voronoi_finite_polygons_2d, clip, PolyArea, get_ccw_contour

scn_nbr = 6
run_nbr = 3
parser = PeTrackParser()
# main_dir = '/home/cyrus/Dropbox/PAMELA data/new_cut_video'
main_dir = 'C:/Users/Javad/Dropbox/PAMELA data/new_cut_video'
cap = cv2.VideoCapture(main_dir + '/S%d/S%d_run%d0001-100000-undistort.mp4' % (scn_nbr, scn_nbr, run_nbr))
p_data, t_data, ids = parser.load(main_dir + '/S%d/run%d/S%d_run%d-feetcorrect.txt' % (scn_nbr, run_nbr, scn_nbr, run_nbr))

max_t = -1
for Ti in t_data:
    max_t = max(max(Ti), max_t)

cv2.namedWindow('im', cv2.WINDOW_NORMAL)
# rect_all = np.array([[655, 300], [963, 284], [1010, 1050], [593, 1020]])
rect_up = np.array([[655, 300], [963, 284], [985, 600], [836, 600], [836, 1000], [695, 1000], [695, 600], [622, 603]])
rect_down = np.array(
    [[593, 1020], [1000, 1050], [985, 600], [836, 600], [836, 200], [695, 200], [695, 600], [622, 603]])
# rect_comp = np.array([[630, 595], [645, 300], [963, 284], [985, 595],
#                       [836, 595], [836, 602],
#                       [985, 602], [1010, 1050], [593, 1020], [630, 606],
#                       [695, 600], [695, 594] ])

def getVoronois(frame_id):
    available_ids = []
    for ii, Ti in enumerate(t_data):
        if len(Ti) == 0 or frame_id < Ti[0] or frame_id > Ti[-1]: continue
        available_ids.append(ii)
        t_ind = Ti.index(frame_id)
        foot_i_t = p_data[ii][t_ind][0:2].astype(int)
        points.append(foot_i_t)

    ped_inds = []
    ped_cells = []
    if len(points) > 2:
        vor = Voronoi(points)
        regions, vertices = voronoi_finite_polygons_2d(vor)

        for ii, region in enumerate(regions):
            voronoi_cell = vertices[region]
            voronoi_cell = get_ccw_contour(voronoi_cell)
            # polygon = clip(rect_comp, voronoi_cell)
            if points[ii][1] < 600:  # up
                clipped_cell = clip(rect_up, voronoi_cell)
            else:
                clipped_cell = clip(rect_down, voronoi_cell)

            if len(clipped_cell) == 0:
                raise ValueError('voronoi cell invalid')

            clipped_cell = np.stack(clipped_cell)
            ped_cells.append(clipped_cell)
            ped_inds.append(available_ids[ii])
    return ped_cells, points, ped_inds

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

    ped_cells, centers, ped_inds = getVoronois(t)
    for ii, cell in enumerate(ped_cells):
        cv2.fillPoly(im, [cell.astype(int)],
                     (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
        area_i = PolyArea(cell)
        cntr = centers[ii].astype(int)
        cv2.circle(im, (cntr[0], cntr[1]), 5, (100, 0, 0), -1)
        cv2.putText(im, '%d' % ped_inds[ii], (cntr[0], cntr[1]),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.8, RED_COLOR, 2)



    im = np.flipud(im)
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