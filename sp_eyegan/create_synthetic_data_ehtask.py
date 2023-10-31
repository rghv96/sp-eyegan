from __future__ import annotations

import argparse
import os
import random
import socket
import sys

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow
import tensorflow as tf
from sklearn import metrics
from tqdm.notebook import tqdm

from sp_eyegan.model import eventGAN as eventGAN
from sp_eyegan.model import stat_scanpath_model as stat_scanpath_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', type=int, required=True)
    parser.add_argument('--GPU', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=100000)
    parser.add_argument('--flag_train_on_gpu', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--sac_window_size', type=int, default=30)
    parser.add_argument('--fix_window_size', type=int, default=10)
    parser.add_argument(
        '--scanpath_model', type=str,
        choices=['random', 'stat_model'], default='random'
    )
    parser.add_argument('--fix_hp_path', type=str, default=None)
    parser.add_argument('--sac_hp_path', type=str, default=None)

    args = parser.parse_args()
    GPU = args.GPU
    data_dir = args.data_dir
    num_samples = args.num_samples
    output_window_size = args.window_size
    sac_window_size = args.sac_window_size
    fix_window_size = args.fix_window_size
    scanpath_model = args.scanpath_model
    fix_hp_path = args.fix_hp_path
    sac_hp_path = args.sac_hp_path

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

    fixation_path = 'event_model/fixation_model_ehtask_giw'
    saccade_path = 'event_model/saccade_model_ehtask_giw'

    data_suffix = ''

    if fix_hp_path is not None:
        fixation_path += '_optimized'
        hp_result_data = pd.read_csv(fix_hp_path)
        event_accs = list(hp_result_data['event_acc'])
        model_names = list(hp_result_data['model_name'])
        best_id = np.argmax(event_accs)
        best_model_name = model_names[best_id]

        gen_kernel_sizes = [int(a) for a in
                            np.array(best_model_name.split('_')[0].replace('[', '').replace(']', '').split(','))]
        gen_filter_sizes = [int(a) for a in
                            np.array(best_model_name.split('_')[1].replace('[', '').replace(']', '').split(','))]

        dis_kernel_sizes = [int(a) for a in
                            np.array(best_model_name.split('_')[2].replace('[', '').replace(']', '').split(','))]
        dis_fiter_sizes = [int(a) for a in
                           np.array(best_model_name.split('_')[3].replace('[', '').replace(']', '').split(','))]

        dis_dropout = float(best_model_name.split('_')[4])

        model_config_fixation = {'gen_kernel_sizes': gen_kernel_sizes,
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
        data_suffix = '_optimized'
    else:
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
    if sac_hp_path is not None:
        saccade_path += '_optimized'
        hp_result_data = pd.read_csv(sac_hp_path)
        event_accs = list(hp_result_data['event_acc'])
        model_names = list(hp_result_data['model_name'])
        best_id = np.argmax(event_accs)
        best_model_name = model_names[best_id]

        gen_kernel_sizes = [int(a) for a in
                            np.array(best_model_name.split('_')[0].replace('[', '').replace(']', '').split(','))]
        gen_filter_sizes = [int(a) for a in
                            np.array(best_model_name.split('_')[1].replace('[', '').replace(']', '').split(','))]

        dis_kernel_sizes = [int(a) for a in
                            np.array(best_model_name.split('_')[2].replace('[', '').replace(']', '').split(','))]
        dis_fiter_sizes = [int(a) for a in
                           np.array(best_model_name.split('_')[3].replace('[', '').replace(']', '').split(','))]

        dis_dropout = float(best_model_name.split('_')[4])

        model_config_saccade = {'gen_kernel_sizes': gen_kernel_sizes,
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
        data_suffix = '_optimized'
    else:
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
        syt_data = data_generator.sample_random_data(
            num_samples=num_samples,
            output_size=output_window_size,
        )
    elif scanpath_model == 'stat_model':
        stat_model = stat_scanpath_model.satisticalScanPath()
        # collect texts
        # TODO change to image data
        image_data_dir = 'data/GazeBase/ocr_detection/images/'
        text_data_csvs = []
        file_list = os.listdir(image_data_dir)
        text_lists = []
        for ii in range(len(file_list)):
            if file_list[ii].endswith('.csv'):
                ocr_data = pd.read_csv(image_data_dir + file_list[ii])
                confs = np.array(ocr_data['conf'])
                lefts = np.array(ocr_data['left'])
                tops = np.array(ocr_data['top'])
                widths = np.array(ocr_data['width'])
                heights = np.array(ocr_data['height'])
                words = np.array(ocr_data['text'])

                text_list = []

                for i in range(len(confs)):
                    cur_conf = confs[i]
                    if cur_conf != -1:
                        text_list.append((words[i],
                                          lefts[i] + (widths[i] / 2),
                                          tops[i] + (heights[i] / 2)))
                text_lists.append(text_list)
        expt_txts = [expt_txt for _ in range(len(text_lists))]
        data_dict = data_generator.sample_scanpath_dataset_stat_model(
            stat_model,
            text_lists,
            expt_txts,
            num_sample_saccs=250,
            dva_threshold=0.2,
            max_iter=10,
            num_scanpaths_per_text=10,
            num_samples=num_samples,
            output_size=output_window_size,
            store_dva_data=False,
        )
        syt_data = data_dict['vel_data']

    print('save data')
    if window_size != 5000:
        data_save_path = data_dir + 'synthetic_ehtask_data_' + str(scanpath_model) + '_' + str(
            output_window_size) + data_suffix
        np.save(data_save_path, syt_data)
    else:
        data_save_path = data_dir + 'synthetic_ehtask_data_'+ str(scanpath_model) + data_suffix
        np.save(data_save_path, syt_data)

    print('data saved to: ' + str(data_save_path))


if __name__ == '__main__':
    # execute only if run as a script
    raise SystemExit(main())
