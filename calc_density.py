import os
import cv2
import numpy as np
from parse_utils import PeTrackParser
from scipy.spatial import Voronoi, voronoi_plot_2d
from DEFINITIONS import *
import matplotlib.pyplot as plt
from voronoi import voronoi_finite_polygons_2d, clip, PolyArea, get_ccw_contour
from homography import getFieldHomog
import matplotlib.cm as cm
from colormap import parula

scn_nbr = 8
run_nbr = 4
parser = PeTrackParser()
# main_dir = '/home/cyrus/Dropbox/PAMELA data/new_cut_video'
main_dir = 'C:/Users/Javad/Dropbox/PAMELA data/new_cut_video'
cap = cv2.VideoCapture(main_dir + '/S%d/S%d_run%d0001-100000-undistort.mp4' % (scn_nbr, scn_nbr, run_nbr))
p_data, t_data, ids = parser.load(main_dir + '/S%d/run%d/S%d_run%d-std.txt' % (scn_nbr, run_nbr, scn_nbr, run_nbr))

max_t = -1
for Ti in t_data:
    max_t = max(max(Ti), max_t)

cv2.namedWindow('im', cv2.WINDOW_NORMAL)
# rect_all = np.array([[655, 300], [963, 284], [1010, 1050], [593, 1020]])
# rect_up =np.array([[655, 300], [963, 284], [985, 600], [836, 600], [836, 1000], [695, 1000], [695, 600], [622, 603]])
# rect_down =np.array([[593, 1020], [1000, 1050], [985, 600], [836, 600], [836, 200], [695, 200], [695, 600], [622, 603]])
# rect_comp = np.array([[630, 595], [645, 300], [963, 284], [985, 595],
#                       [836, 595], [836, 602],
#                       [985, 602], [1010, 1050], [593, 1020], [630, 606],
#                       [695, 600], [695, 594] ])

rect_up =np.array([[0, 0], [600, 0], [600, 580], [400, 580], [400, 800], [200, 800], [200, 580], [0, 580]]) + np.array([200, 200])

rect_down =np.array([[0, 1200], [600, 1200], [600, 580], [400, 580], [400, 200], [200, 200], [200, 580], [0, 580]]) + np.array([200, 200])

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
        regions, vertices = voronoi_finite_polygons_2d(points)

        for ii, region in enumerate(regions):
            voronoi_cell = vertices[region]
            voronoi_cell = get_ccw_contour(voronoi_cell)
            # polygon = clip(rect_comp, voronoi_cell)
            if points[ii][1] < rect_up[2,1]:  # up
                clipped_cell = clip(rect_up, voronoi_cell)
            else:
                clipped_cell = clip(rect_down, voronoi_cell)

            if len(clipped_cell) == 0:
                continue
                # raise ValueError('voronoi cell invalid')

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
    HomogField, _ = getFieldHomog()
    warp_frame = cv2.warpPerspective(im_raw, HomogField, (1000, 1600), cv2.INTER_LINEAR)
    im = np.copy(warp_frame)
    im_colors = np.zeros(warp_frame.shape, dtype=warp_frame.dtype)
    points = []
    found_inds = []

    # cv2.fillPoly(im, [rect_up], RED_COLOR)

    ped_cells, centers, ped_inds = getVoronois(t)
    for ii, cell in enumerate(ped_cells):
        area_i = PolyArea(cell)

        area_normal = min(1, area_i / 200000.)
        color_i = parula[int(area_normal * (len(parula)-1))]

        # color_i = cm.hot(area_normal)
        color_i = [int(x * 255) for x in color_i[:3]]

        cv2.fillPoly(im_colors, [cell.astype(int)], color_i)
        cntr = centers[ii].astype(int)
        cv2.circle(im, (cntr[0], cntr[1]), 5, (100, 0, 0), -1)
        cv2.putText(im, '%d' % ped_inds[ii], (cntr[0], cntr[1]),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.8, RED_COLOR, 2)

    cv2.line(im, (200, 200+580), (800, 200+580), RED_COLOR, 3)
    im = cv2.addWeighted(im, 0.5, im_colors, 0.8, 0)
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