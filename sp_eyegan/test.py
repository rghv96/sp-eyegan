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

target_sampling_rate = 1000
sampling_rate = 1000


def main():
    gaze_data_list, gaze_feature_dict, gaze_label_matrix, gaze_label_dict = data_loader.load_ehtask_data(
            ehtask_dir=config.EHTASK_DIR,
            target_sampling_rate=target_sampling_rate,
            sampling_rate=sampling_rate,
        )

    print(len(gaze_data_list))

if __name__ == '__main__':
    # execute only if run as a script
    raise SystemExit(main())