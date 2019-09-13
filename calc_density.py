import os
import cv2
import numpy as np
from parse_utils import PeTrackParser
from scipy.spatial import Voronoi, voronoi_plot_2d
from DEFINITIONS import *
import matplotlib.pyplot as plt
from voronoi3 import Voronoi3

from linear_models import MyKalman
# from shapely.geometry import MultiPoint, Point, Polygon


scn_nbr = 1
run_nbr = 1
parser = PeTrackParser()
# main_dir = '/home/cyrus/Dropbox/PAMELA data/new_cut_video'
main_dir = 'C:/Users/Javad/Dropbox/PAMELA data/new_cut_video'
p_data, t_data, ids = parser.load(main_dir + '/S%d/run%d/S%d_run%d-new.txt' % (scn_nbr, run_nbr, scn_nbr, run_nbr))

max_t = -1
for Ti in t_data:
    max_t = max(max(Ti), max_t)

cv2.namedWindow('im', cv2.WINDOW_NORMAL)
t = -1
pause = False
while True:
    if t < max_t and not pause: t += 1
    im = np.ones((1200, 1600, 3), np.uint8) * 255
    points = []
    found_inds = []
    for ii, Ti in enumerate(t_data):
        if len(Ti) == 0 or t < Ti[0] or t > Ti[-1]: continue
        found_inds.append(ii)
        t_ind = Ti.index(t)
        head_i_t = p_data[ii][t][3:5].astype(int)
        points.append(head_i_t)
        cv2.circle(im, (int(head_i_t[0]), int(head_i_t[1])), 5, (100,0,0), -1)


    segments = Voronoi3(points)
    for seg in segments:
        seg = np.stack(seg).astype(int)
        print(seg)
        cv2.line(im, (seg[0][0], seg[0][1]), (seg[1][0], seg[1][1]), RED_COLOR, 2)

    # plt.scatter(points[:, 0], points[:, 1], color="blue")
    # lines = matplotlib.collections.LineCollection(lines, color='red')
    # plt.gca().add_collection(lines)
    # plt.axis((-20, 120, -20, 120))
    # plt.show()


    vor = Voronoi(points)
    #
    # # RI = vor.point_region
    # # R = vor.regions
    # # V = vor.vertices
    # # RV = vor.ridge_vertices
    # # PR = vor.point_region
    #
    # segments=[]
    #
    # for i, (istart, iend) in enumerate(vor.ridge_vertices):
    #     if istart < 0 or iend <= 0:
    #         start = vor.vertices[istart] if istart >= 0 else vor.vertices[iend]
    #         # if check_outside(start, bbox): continue
    #         first, second = vor.ridge_points[i]
    #         first, second = vor.points[first], vor.points[second]
    #         edge = second - first
    #         vector = np.array([-edge[1], edge[0]])
    #         # midpoint = (second + first) / 2
    #         # orientation = np.sign(np.dot(midpoint - center, vector))
    #         # vector = orientation * vector
    #         # c = calc_shift(start, vector, bbox)
    #         # if c is not None:
    #         #     segments.append([start, start + c * vector])
    #
    #     else:
    #         start, end = vor.vertices[istart], vor.vertices[iend]
    #         # if check_outside(start, bbox):
    #         #     start = move_point(start, end, bbox)
    #         #     if start is None:
    #         #         continue
    #         # if check_outside(end, bbox):
    #         #     end = move_point(end, start, bbox)
    #         #     if end is None:
    #         #         continue
    #         segments.append([start, end])
    #
    # V = vor.vertices
    #
    # print(len(vor.ridge_vertices))
    # for ridge_inds in vor.ridge_vertices:
    #     vers_i = []
    #     print(len(ridge_inds))
    #     for ridge_ind in ridge_inds:
    #         ridge_ver = vor.vertices[ridge_ind]
    #         vers_i.append(ridge_ver)
    #     vers_i = np.stack(vers_i).astype(np.int32)
    #     # print(vers_i)
    #     cv2.fillPoly(im, [vers_i], (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
    #
    #
    # for ii, pnt in enumerate(points):
    #     reg_ind = vor.point_region[ii]
    #     ver_inds = vor.regions[reg_ind]
    #     vers_i = []
    #     for ver_ind in ver_inds:
    #         if ver_ind == -1: continue
    #         vers_i.append(vor.vertices[ver_ind])
    #
    #     ridge_inds = vor.ridge_vertices[ii]
    #     # print(ridge_inds)
    #     # vers_i.append(vor.vertices[ridge_inds])
    #     # for ridge_ind in ridge_inds:
    #     #     vers_i.append(vor.ridge_points[ridge_ind])
    #     cv2.line(im,
    #              (int(V[ridge_inds[0]][0]), int(V[ridge_inds[0]][1])),
    #              (int(V[ridge_inds[1]][0]), int(V[ridge_inds[1]][1])), RED_COLOR, 2)
    #
    #     # print(len(vers_i))
    #     vers_i = np.stack(vers_i).astype(np.int32)
    #     # print(vers_i)
    #     cv2.fillPoly(im, [vers_i], (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
    #     cv2.circle(im, (int(points[ii][0]), int(points[ii][1])), 5, (100,0,0), -1)
    #
    #


    # regions, vertices = voronoi_finite_polygons_2d(vor)

    im = np.flipud(im)
    cv2.imshow('im', im)
    key = cv2.waitKeyEx(30)
    if key & 0xFF == ord('q'):
        break
    elif key == 32:  # space
        pause = not pause

    voronoi_plot_2d(vor)
    plt.show()



peds = []
K = 3
for ped_i in peds:
    pass