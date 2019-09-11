import csv
import math
import sys
import numpy as np
import os
# from pandas import DataFrame, concat


class PeTrackParser:
    def __init__(self):
        self.actual_fps = 1.
        self.delimit = " "
        self.heigth = 1.80

    def load(self, filename, down_sample=1):
        #NOTICE
        # any line starting with # is a comment
        pos_data_list = list()
        time_data_list = list()

        fps = 30
        self.actual_fps = fps / down_sample

        with open(filename, 'r') as data_file:
            csv_reader = csv.reader(data_file, delimiter=' ')
            id_list = list()
            for i, tokens in enumerate(csv_reader):
                if len(tokens) == 0 or tokens[0].startswith('#'):
                    continue
                id = int(tokens[0])
                ts = int(tokens[1])
                if ts % down_sample != 0:
                    continue

                px = float(tokens[2]) * 100
                py = float(tokens[3]) * -100
                pz = float(tokens[4])

                hx, hy, hz = 0, 0, 0
                if len(tokens) == 8:
                    hx = float(tokens[5]) * 100
                    hy = float(tokens[6]) * -100
                    hz = float(tokens[7])

                if id not in id_list:
                    id_list.append(id)
                    pos_data_list.append(list())
                    time_data_list.append(list())
                pos_data_list[-1].append(np.array([px, py, pz, hx, hy, hz]))
                time_data_list[-1].append(ts)

        p_data = list()

        for i in range(len(pos_data_list)):
            poss_i = np.array(pos_data_list[i])
            p_data.append(poss_i)

        # t_data = np.array(time_data_list)
        return p_data, time_data_list, id_list

    def save(self, filename, p_data, t_data):
        with open(filename, 'w') as out_file:
            if len(p_data[-1][-1]) < 6:
                out_file.write('# id frame x/m y/m z/m\n')
                for ii, ped in enumerate(p_data):
                    for tt, pos in enumerate(ped):
                        out_file.write("%d %d %.5f %.5f %.2f\n" %(ii+1, t_data[ii][tt], pos[0]/100, pos[1]/-100, self.heigth))
                    # out_file.write('\n')
            else:
                out_file.write('# id frame foot_x/m foot_y/m foot_z/m head_x/m head_y/m head_z/m\n')
                out_file.write('# ids = %d\n' % len(t_data))
                out_file.write('# Robot id = %d\n' % 1)
                for ii, Ti in enumerate(t_data):
                    for kk, t in enumerate(Ti):
                        out_file.write("%d %d %.3f %.3f %.2f %.3f %.3f %.2f\n"
                                       % (ii + 1, t,
                                          p_data[ii][kk, 0] / 100, p_data[ii][kk, 1] / -100, p_data[ii][kk, 2],
                                          p_data[ii][kk, 3] / 100, p_data[ii][kk, 4] / -100, p_data[ii][kk, 5]))

