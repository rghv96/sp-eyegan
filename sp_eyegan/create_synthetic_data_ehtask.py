from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd

from sp_eyegan.model import eventGAN as eventGAN
from sp_eyegan.model import stat_scanpath_model as stat_scanpath_model

def get_user_ids(video_no, task):
    user_ids = []
    txt_files = []
    start_dir = 'data/EHTaskDataset/RawData/'
    list_dir = os.listdir(start_dir)
    for i in range(len(list_dir)):
        cur_dir = start_dir + list_dir[i]
        if list_dir[i].startswith('User'):
            txt_files.append(cur_dir)
    for txt_file in txt_files:
        file_name = txt_file.split('/')[-1]
        file_name_split = file_name.replace('.txt', '').split('_')
        curr_user = int(file_name_split[1])
        curr_video = int(file_name_split[3])
        curr_task = int(file_name_split[5])
        if curr_video == int(video_no) and curr_task == int(task):
            user_ids.append(curr_user)

    return user_ids




def replace_giw(input_file, output_file, x_locations, y_locations):
    # Read the input file
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Replace the specified column with values from the replacement list
    for i in range(len(lines)):
        if i >= len(x_locations):
            print('Exiting early')
            break

        columns = lines[i].split()
        columns[6] = str(x_locations[i])
        columns[7] = str(y_locations[i])
        lines[i] = ' '.join(columns) + '\n'

    # Write the modified content to the output file
    with open(output_file, 'w') as file:
        file.writelines(lines)

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-window_size', '--window_size', type=int, required=True)
    parser.add_argument('--GPU', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--flag_train_on_gpu', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--sac_window_size', type=int, default=10)
    parser.add_argument('--fix_window_size', type=int, default=10)
    parser.add_argument('-task', '--task', type=str, default='all')  # all | 1 | 2 | 3 | 4
    parser.add_argument('-video', '--video', type=str, default='all')  # all | 1 | 2 | 3 | 4 ...
    parser.add_argument(
        '-scanpath_model', '--scanpath_model', type=str,
        choices=['random', 'average'], default='random'
    )

    args = parser.parse_args()
    GPU = args.GPU
    data_dir = args.data_dir
    num_samples = args.num_samples
    # output_window_size = args.window_size
    sac_window_size = args.sac_window_size
    fix_window_size = args.fix_window_size
    scanpath_model = args.scanpath_model

    task_ids = []
    task = args.task
    if task == 'all':
        task_ids = [1, 2, 3, 4]
    else:
        task_ids.append(int(task))

    video_no = args.video
    video_ids = []
    if video_no == 'all':
        video_ids = [i + 1 for i in range(15)]
    else:
        video_ids.append(int(video_no))

    # params for stimulus
    expt_txt = {'px_x': 1680,
                'px_y': 1050,
                'max_dva_x': 30,
                'max_dva_y': 25
                }

    # params for NN
    random_size = 32
    window_size = 10
    gen_kernel_sizes_fixation = [fix_window_size, 8, 4, 2]
    gen_kernel_sizes_saccade = [sac_window_size, 8, 4, 2]
    gen_filter_sizes = [16, 8, 4, 2]
    channels = 2
    relu_in_last = False
    batch_size = 256

    dis_kernel_sizes = [8, 16, 32]
    dis_fiter_sizes = [32, 64, 128]
    dis_dropout = 0.3

    sample_size = 1000

    # params for generator
    window_size = 10
    random_size = 32
    channels = 2
    mean_sacc_len = 20
    std_sacc_len = 10

    mean_fix_len = 25
    std_fix_len = 15

    fixation_path = 'event_model/fixation_model_ehtask_giw_video_' + video_no + '_task_' + task
    saccade_path = 'event_model/saccade_model_ehtask_giw_video_' + video_no + '_task_' + task

    data_suffix = ''

    model_config_fixation = {'gen_kernel_sizes': gen_kernel_sizes_fixation,
                             'gen_filter_sizes': gen_filter_sizes,
                             'dis_kernel_sizes': dis_kernel_sizes,
                             'dis_fiter_sizes': dis_fiter_sizes,
                             'dis_dropout': dis_dropout,
                             'window_size': fix_window_size,
                             'channels': channels,
                             'batch_size': batch_size,
                             'random_size': random_size,
                             'relu_in_last': relu_in_last,
                             }

    model_config_saccade = {'gen_kernel_sizes': gen_kernel_sizes_saccade,
                            'gen_filter_sizes': gen_filter_sizes,
                            'dis_kernel_sizes': dis_kernel_sizes,
                            'dis_fiter_sizes': dis_fiter_sizes,
                            'dis_dropout': dis_dropout,
                            'window_size': sac_window_size,
                            'channels': channels,
                            'batch_size': batch_size,
                            'random_size': random_size,
                            'relu_in_last': relu_in_last,
                            }

    gan_config = {'window_size': window_size,
                  'random_size': random_size,
                  'channels': channels,
                  'mean_sacc_len': mean_sacc_len,
                  'std_sacc_len': std_sacc_len,
                  'mean_fix_len': mean_fix_len,
                  'std_fix_len': std_fix_len,
                  'fixation_path': fixation_path,
                  'saccade_path': saccade_path,
                  }

    flag_train_on_gpu = args.flag_train_on_gpu
    if flag_train_on_gpu == 1:
        flag_train_on_gpu = True
    else:
        flag_train_on_gpu = False

    # set up GPU
    if flag_train_on_gpu:
        import tensorflow as tf
        # select graphic card
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        config = tf.compat.v1.ConfigProto(log_device_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 1.
        config.gpu_options.allow_growth = True
        tf_session = tf.compat.v1.Session(config=config)
    else:
        import tensorflow as tf
        # select graphic card
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    data_generator = eventGAN.dataGenerator(gan_config,
                                            model_config_fixation,
                                            model_config_saccade,
                                            )

    if scanpath_model == 'random':
        user_ids = get_user_ids(video_no, task)
        print(user_ids)
        for i in range(len(user_ids)):
            # print(i)
            scanpath_file = data_dir + 'scanpath/random/scanpath_ehtask_video_' + video_no + '_task_' + task + '_type_' + scanpath_model + '_' + str(i+1) + '.npy'
            scanpath = np.load(scanpath_file)
            x_dva = scanpath[:, 1]
            y_dva = scanpath[:, 2]

            x_locations, y_locations, fix_x_loc, fix_y_loc, sac_x_loc, sac_y_loc = data_generator.sample_scanpath(
                x_fix_locations=x_dva,
                y_fix_locations=y_dva,
                num_sample_saccs=1000,
                dva_threshold=0.01,
                fixation_durations=None,
                saccade_durations=None,
            )
            print('x_locations.len: ', len(x_locations))

            user_id = str(user_ids[i])
            if len(user_id) == 1:
                user_id = '0' + user_id
            if len(video_no) == 1:
                video_str = '0' + video_no
            else:
                video_str = video_no

            input_file = data_dir + 'EHTaskDataset/RawData/User_' + user_id + '_Video_' + video_str + '_Task_' + task + '.txt'
            print(input_file)
            output_file = data_dir + 'EHTask_Synthetic/User_' + user_id + '_Video_' + video_str + '_Task_' + task + '.txt'
            replace_giw(input_file, output_file, x_locations, y_locations)

    elif scanpath_model == 'average':
        # scanpath_file = '../data/scanpath/scanpath_ehtask_video_' + video_no + '_task_' + task + '.npy'
        scanpath_file = data_dir +'scanpath/scanpath_ehtask_video_' + video_no + '_task_' + task + '_type_' + scanpath_model + '.npy'
        scanpath = np.load(scanpath_file)
        frame_no = scanpath[:, 0]
        x_dva = scanpath[:, 1]
        y_dva = scanpath[:, 2]

        # x_locations, y_locations = data_generator.sample_scanpath(
        #     x_fix_locations=x_dva,
        #     y_fix_locations=y_dva,
        #     num_sample_saccs=1000,
        #     dva_threshold=0.01,
        #     fixation_durations=None,
        #     saccade_durations=None,
        # )

if __name__ == '__main__':
    # execute only if run as a script
    raise SystemExit(main())
