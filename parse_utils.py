import csv
import math
import sys
import numpy as np
import os
# from pandas import DataFrame, concat


class Scale(object):
    '''
    Given max and min of a rectangle it computes the scale and shift values to normalize data to [0,1]
    '''

    def __init__(self):
        self.min_x = +math.inf
        self.max_x = -math.inf
        self.min_y = +math.inf
        self.max_y = -math.inf
        self.sx, self.sy = 1, 1

    def calc_scale(self, keep_ratio=True):
        self.sx = 1 / (self.max_x - self.min_x)
        self.sy = 1 / (self.max_y - self.min_y)
        if keep_ratio:
            if self.sx > self.sy:
                self.sx = self.sy
            else:
                self.sy = self.sx

    def normalize(self, data, shift=True, inPlace=True):
        if inPlace:
            data_copy = data
        else:
            data_copy = np.copy(data)

        if data.ndim == 1:
            data_copy[0] = (data[0] - self.min_x * shift) * self.sx
            data_copy[1] = (data[1] - self.min_y * shift) * self.sy
        elif data.ndim == 2:
            data_copy[:, 0] = (data[:, 0] - self.min_x * shift) * self.sx
            data_copy[:, 1] = (data[:, 1] - self.min_y * shift) * self.sy
        elif data.ndim == 3:
            data_copy[:, :, 0] = (data[:, :, 0] - self.min_x * shift) * self.sx
            data_copy[:, :, 1] = (data[:, :, 1] - self.min_y * shift) * self.sy
        elif data.ndim == 4:
            data_copy[:, :, :, 0] = (data[:, :, :, 0] - self.min_x * shift) * self.sx
            data_copy[:, :, :, 1] = (data[:, :, :, 1] - self.min_y * shift) * self.sy
        else:
            return False
        return data_copy

    def denormalize(self, data, shift=True, inPlace=False):
        if inPlace:
            data_copy = data
        else:
            data_copy = np.copy(data)

        ndim = data.ndim
        if ndim == 1:
            data_copy[0] = data[0] / self.sx + self.min_x * shift
            data_copy[1] = data[1] / self.sy + self.min_y * shift
        elif ndim == 2:
            data_copy[:, 0] = data[:, 0] / self.sx + self.min_x * shift
            data_copy[:, 1] = data[:, 1] / self.sy + self.min_y * shift
        elif ndim == 3:
            data_copy[:, :, 0] = data[:, :, 0] / self.sx + self.min_x * shift
            data_copy[:, :, 1] = data[:, :, 1] / self.sy + self.min_y * shift
        elif ndim == 4:
            data_copy[:, :, :, 0] = data[:, :, :, 0] / self.sx + self.min_x * shift
            data_copy[:, :, :, 1] = data[:, :, :, 1] / self.sy + self.min_y * shift
        else:
            return False

        return data_copy


class BIWIParser:
    def __init__(self):
        self.scale = Scale()
        self.all_ids = list()
        self.actual_fps = 0.
        self.delimit = ' '
        self.p_data = []
        self.v_data = []
        self.t_data = []
        self.min_t = int(sys.maxsize)
        self.max_t = -1
        self.interval = 6


    def load(self, filename, down_sample=1):
        pos_data_dict = dict()
        vel_data_dict = dict()
        time_data_dict = dict()
        self.all_ids.clear()

        if 'zara' in filename:
            self.delimit = '\t'

        # to search for files in a folder?
        file_names = list()
        if '*' in filename:
            files_path = filename[:filename.index('*')]
            extension = filename[filename.index('*') + 1:]
            for file in os.listdir(files_path):
                if file.endswith(extension):
                    file_names.append(files_path + file)
        else:
            file_names.append(filename)

        self.actual_fps = 2.5
        for file in file_names:
            with open(file, 'r') as data_file:
                content = data_file.readlines()
                id_list = list()
                for i, row in enumerate(content):
                    row = row.split(self.delimit)
                    while '' in row: row.remove('')
                    if len(row) < 8: continue

                    ts = float(row[0])
                    id = round(float(row[1]))
                    if ts % down_sample != 0:
                        continue
                    if ts < self.min_t: self.min_t = ts
                    if ts > self.max_t: self.max_t = ts


                    px = float(row[2])
                    py = float(row[4])
                    vx = float(row[5])
                    vy = float(row[7])

                    if id not in id_list:
                        id_list.append(id)
                        pos_data_dict[id] = list()
                        vel_data_dict[id] = list()
                        time_data_dict[id] = np.empty(0, dtype=int)
                        last_t = ts
                    pos_data_dict[id].append(np.array([px, py]))
                    vel_data_dict[id].append(np.array([vx, vy]))
                    time_data_dict[id] = np.hstack((time_data_dict[id], np.array([ts])))
            self.all_ids += id_list

        for key, value in pos_data_dict.items():
            poss_i = np.array(value)
            self.p_data.append(poss_i)
            # TODO: you can apply a Kalman filter/smoother on v_data
            vels_i = np.array(vel_data_dict[key])
            self.v_data.append(vels_i)
            self.t_data.append(np.array(time_data_dict[key]).astype(np.int32))

        # calc scale
        for i in range(len(self.p_data)):
            poss_i = np.array(self.p_data[i])
            self.scale.min_x = min(self.scale.min_x, min(poss_i[:, 0]))
            self.scale.max_x = max(self.scale.max_x, max(poss_i[:, 0]))
            self.scale.min_y = min(self.scale.min_y, min(poss_i[:, 1]))
            self.scale.max_y = max(self.scale.max_y, max(poss_i[:, 1]))
        self.scale.calc_scale()


class PeTrackParser:
    def __init__(self):
        self.scale = Scale()
        self.actual_fps = 0.
        self.delimit = " "

    def load(self, filename, down_sample=1):
        '''
        any line starting with # is a comment
        '''
        pos_data_list = list()
        vel_data_list = list()
        time_data_list = list()

        fps = 30
        self.actual_fps = fps / down_sample

        with open(filename, 'r') as data_file:
            csv_reader = csv.reader(data_file, delimiter=' ')
            id_list = list()
            for i, tokens in enumerate(csv_reader):
                if len(tokens) == 0 or tokens[0] == '#':
                    continue
                id = int(tokens[0])
                ts = float(tokens[1])
                if ts % down_sample != 0:
                    continue

                px = float(tokens[2])/100.
                py = float(tokens[3])/100.
                pz = float(tokens[4])/100.
                if id not in id_list:
                    id_list.append(id)
                    pos_data_list.append(list())
                    time_data_list.append(np.empty((0), dtype=int))
                pos_data_list[-1].append(np.array([px, py]))
                time_data_list[-1] = np.hstack((time_data_list[-1], np.array([ts])))

        p_data = list()

        for i in range(len(pos_data_list)):
            poss_i = np.array(pos_data_list[i])
            p_data.append(poss_i)

        t_data = np.array(time_data_list)

        for i in range(len(pos_data_list)):
            poss_i = np.array(pos_data_list[i])
            self.scale.min_x = min(self.scale.min_x, min(poss_i[:, 0]))
            self.scale.max_x = max(self.scale.max_x, max(poss_i[:, 0]))
            self.scale.min_y = min(self.scale.min_y, min(poss_i[:, 1]))
            self.scale.max_y = max(self.scale.max_y, max(poss_i[:, 1]))
        self.scale.calc_scale()

        return p_data, t_data
