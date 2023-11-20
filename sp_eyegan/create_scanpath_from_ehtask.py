from __future__ import annotations

import argparse
import math
import os
import random
import socket
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from tqdm import tqdm

import config as config
from sp_eyegan.preprocessing import data_loader as data_loader
from sp_eyegan.preprocessing import event_detection as event_detection
from sp_eyegan.preprocessing import smoothing as smoothing


def get_avg_scanpath(global_fixation_list):
    scanpath = []
    for i in range(len(global_fixation_list[0])):
        frame_sum = 0
        x_dva_sum = 0
        y_dva_sum = 0

        for j in range(len(global_fixation_list)):
            frame_sum += global_fixation_list[j][i][0][6]
            x_dva_sum += np.sum(global_fixation_list[j][i][:, 0])
            y_dva_sum += np.sum(global_fixation_list[j][i][:, 1])

        # Calculate the average for frame_no, x_dva, and y_dva
        frame_avg = frame_sum / len(global_fixation_list)
        x_dva_avg = x_dva_sum / (10 * len(global_fixation_list))
        y_dva_avg = y_dva_sum / (10 * len(global_fixation_list))

        scanpath.append([int(frame_avg), x_dva_avg, y_dva_avg])
    scanpath = np.array(scanpath)
    return scanpath

def get_random_scanpath(global_fixation_list):
    scanpath = []
    for i in range(len(global_fixation_list[0])):
        idx = random.randint(0, len(global_fixation_list)-1)
        frame_no = global_fixation_list[idx][i][0][6]
        x_dva_sum = np.sum(global_fixation_list[idx][i][:, 0])
        y_dva_sum = np.sum(global_fixation_list[idx][i][:, 1])

        x_dva_avg = x_dva_sum / len(global_fixation_list[idx][i])
        y_dva_avg = y_dva_sum / len(global_fixation_list[idx][i])

        scanpath.append([int(frame_no), x_dva_avg, y_dva_avg])
    scanpath = np.array(scanpath)
    return scanpath


def main():
    # gloabal params
    sampling_rate = 100

    # params
    smoothing_window_length = 0.007
    disable = False
    min_fixation_length = 10
    max_fixation_dispersion = 2.7
    max_vel = 500

    parser = argparse.ArgumentParser()
    parser.add_argument('-target_sampling_rate', '--target_sampling_rate', type=int, default=100)
    parser.add_argument('-sac_window_size', '--sac_window_size', type=int, default=10)
    parser.add_argument('-fix_window_size', '--fix_window_size', type=int, default=10)
    parser.add_argument(
        '-type', '--type', type=str,
        choices=['random', 'average'], default='average'
    )
    parser.add_argument('-task', '--task', type=str, default='all')  # all | 1 | 2 | 3 | 4
    parser.add_argument('-video', '--video', type=str, default='all')  # all | 1 | 2 | 3 | 4 ...

    args = parser.parse_args()
    type = args.type
    task_ids = []
    task = args.task
    if task == 'all':
        task_ids = [1, 2, 3, 4]
    else:
        task_ids.append(int(task))

    video_no = args.video
    video_ids = []
    if video_no == 'all':
        video_ids = [i+1 for i in range(15)]
    else:
        video_ids.append(int(video_no))

    target_sampling_rate = args.target_sampling_rate
    sac_window_size = args.sac_window_size
    fix_window_size = args.fix_window_size

    gaze_data_list, gaze_feature_dict, gaze_label_matrix, gaze_label_dict = data_loader.load_ehtask_data(
        ehtask_dir=config.EHTASK_DIR,
        target_sampling_rate=target_sampling_rate,
        sampling_rate=sampling_rate,
        task_ids=task_ids,
        video_ids=video_ids
    )

    event_df_list = []
    list_dicts_list = []
    for i in tqdm(np.arange(len(gaze_data_list)), disable=False):
        x_dva = gaze_data_list[i][:, gaze_feature_dict['x_dva_left']]
        y_dva = gaze_data_list[i][:, gaze_feature_dict['y_dva_left']]
        x_pixel = gaze_data_list[i][:, gaze_feature_dict['x_left_px']]
        y_pixel = gaze_data_list[i][:, gaze_feature_dict['y_left_px']]
        corrupt = np.zeros([len(x_dva), ])
        corrupt_ids = np.where(np.logical_or(np.isnan(x_pixel),
                                             np.isnan(y_pixel)))[0]
        corrupt[corrupt_ids] = 1

        # apply smoothing like in https://digital.library.txstate.edu/handle/10877/6874
        smooth_vals = smoothing.smooth_data(x_dva, y_dva,
                                            n=2, smoothing_window_length=smoothing_window_length,
                                            sampling_rate=target_sampling_rate)

        x_smo = smooth_vals['x_smo']
        y_smo = smooth_vals['y_smo']
        vel_x = smooth_vals['vel_x']
        vel_y = smooth_vals['vel_y']
        vel = smooth_vals['vel']
        acc_x = smooth_vals['acc_x']
        acc_y = smooth_vals['acc_y']
        acc = smooth_vals['acc']

        corrupt_vels = []
        corrupt_vels += list(np.where(vel_x > max_vel)[0])
        corrupt_vels += list(np.where(vel_x < -max_vel)[0])
        corrupt_vels += list(np.where(vel_y > max_vel)[0])
        corrupt_vels += list(np.where(vel_y < -max_vel)[0])

        corrupt[corrupt_vels] = 1

        # dispersion
        list_dicts, event_df = event_detection.get_sacc_fix_lists_dispersion(
            x_smo, y_smo,
            corrupt=corrupt,
            sampling_rate=target_sampling_rate,
            min_duration=min_fixation_length,
            velocity_threshold=20,
            flag_skipNaNs=False,
            verbose=0,
            max_fixation_dispersion=max_fixation_dispersion,
        )

        event_df_list.append(event_df)
        list_dicts_list.append(list_dicts)

    print('number of lists: ' + str(len(event_df_list)))
    #####################################################

    global_fixation_list = []
    for i in tqdm(np.arange(len(event_df_list))):
        fixation_list = []
        list_dicts = list_dicts_list[i]
        event_df = event_df_list[i]
        fixations = list_dicts['fixations']
        x_dva = gaze_data_list[i][:, gaze_feature_dict['x_dva_left']]
        y_dva = gaze_data_list[i][:, gaze_feature_dict['y_dva_left']]
        x_pixel = gaze_data_list[i][:, gaze_feature_dict['x_left_px']]
        y_pixel = gaze_data_list[i][:, gaze_feature_dict['y_left_px']]

        # apply smoothing like in https://digital.library.txstate.edu/handle/10877/6874
        smooth_vals = smoothing.smooth_data(x_dva, y_dva,
                                            n=2, smoothing_window_length=smoothing_window_length,
                                            sampling_rate=target_sampling_rate)

        x_smo = smooth_vals['x_smo']
        y_smo = smooth_vals['y_smo']
        vel_x = smooth_vals['vel_x']
        vel_y = smooth_vals['vel_y']
        vel = smooth_vals['vel']
        acc_x = smooth_vals['acc_x']
        acc_y = smooth_vals['acc_y']
        acc = smooth_vals['acc']

        for f_i in range(len(fixations)):
            fixation_list.append(np.concatenate([
                np.expand_dims(x_smo[fixations[f_i]], axis=1),
                np.expand_dims(y_smo[fixations[f_i]], axis=1),
                np.expand_dims(x_pixel[fixations[f_i]], axis=1),
                np.expand_dims(y_pixel[fixations[f_i]], axis=1),
                np.expand_dims(vel_x[fixations[f_i]], axis=1) / target_sampling_rate,
                np.expand_dims(vel_y[fixations[f_i]], axis=1) / target_sampling_rate,
                np.expand_dims(fixations[f_i], axis=1),
            ], axis=1))

        print('number of fixations: ' + str(len(fixation_list)))

        filtered_fixation_list = []
        mx_dispersion = 0.0
        mn_len_x_dva = 10000
        for f_i in tqdm(np.arange(len(fixation_list))):
            cur_x_dva = fixation_list[f_i][:, 0]
            cur_y_dva = fixation_list[f_i][:, 1]
            x_amp = np.abs(np.max(cur_x_dva) - np.min(cur_x_dva))
            y_amp = np.abs(np.max(cur_y_dva) - np.min(cur_y_dva))
            cur_dispersion = x_amp + y_amp
            mn_len_x_dva = min(mn_len_x_dva, len(cur_x_dva))
            mx_dispersion = max(mx_dispersion, cur_dispersion)
            if cur_dispersion >= max_fixation_dispersion:
                # print(f"cur_dispersion: {cur_dispersion}")
                continue
            if len(cur_x_dva) <= fix_window_size:
                # print(f"len(cur_x_dva): {len(cur_x_dva)}")
                continue
            filtered_fixation_list.append(fixation_list[f_i])
        print('number of fixations after filtering: ' + str(len(filtered_fixation_list)))
        print(f"mx_dispersion: {mx_dispersion}")
        print(f"mn_len_x_dva: {mn_len_x_dva}")

        print('number of fixations: ' + str(len(filtered_fixation_list)))
        # store fixations and saccades
        column_dict = {'x_dva': 0,
                       'y_dva': 1,
                       'x_px': 2,
                       'y_px': 3,
                       'x_dva_vel': 4,
                       'y_dva_vel': 5,
                       'frame': 6
                       }

        # joblib.dump(column_dict, 'data/column_dict.joblib', compress=3, protocol=2)

        fix_lens = [filtered_fixation_list[a].shape[0] for a in range(len(filtered_fixation_list))]
        print('fix_lens: ' + str(np.max(fix_lens)))

        max_fix_len = fix_window_size
        fixation_matrix = np.ones([len(filtered_fixation_list), max_fix_len, len(column_dict)]) * -1

        for i in tqdm(np.arange(len(filtered_fixation_list))):
            cur_fix_len = np.min([max_fix_len, filtered_fixation_list[i].shape[0]])
            fixation_matrix[i, 0:cur_fix_len, :] = filtered_fixation_list[i][0:cur_fix_len, :]

        print('fixation_matrix.shape: ', fixation_matrix.shape)
        global_fixation_list.append(fixation_matrix)

    #####################################################

    min_length = min(len(item) for item in global_fixation_list)
    for i in range(len(global_fixation_list)):
        global_fixation_list[i] = global_fixation_list[i][:min_length]

    print('scanpath length: ' + str(min_length))

    if type == 'average':
        scanpath = get_avg_scanpath(global_fixation_list)
        np.save('data/scanpath/scanpath_ehtask_video_' + video_no + '_task_' + task + '_type_' + type, scanpath)
    elif type == 'random':
        for i in range(len(global_fixation_list)):
            scanpath = get_random_scanpath(global_fixation_list)
            np.save('data/scanpath/random/scanpath_ehtask_video_' + video_no + '_task_' + task + '_type_' + type + '_' + str(i+1), scanpath)

if __name__ == '__main__':
    # execute only if run as a script
    raise SystemExit(main())
