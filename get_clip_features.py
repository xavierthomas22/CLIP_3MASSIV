import clip
import yaml
from utils.tools import *
from utils.Augmentation import *
import torch
import argparse
import os
from helpers import *
import tempfile
import time

from clip_manager import ClipManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='')
    parser.add_argument('--temp_path', default='')
    parser.add_argument('--device', required=True)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--url', type=str, required=True)
    args = parser.parse_args()

    start = time.time()

    # create temp folders
    args = create_temp_folders(args) # created folders saved into args

    # get clip features
    clip_manager = ClipManager(args)
    features = clip_manager.extract_features()

    # cleanup
    remove(clip_manager.args.temp_vid_path)
    remove(clip_manager.args.temp_frames_folder)
    
    print(f'Total Time Taken: {time.time() - start}')



if __name__ == '__main__':
    main()