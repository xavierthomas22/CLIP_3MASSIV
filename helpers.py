import mmcv
import tempfile
import os
import wget
import shutil
import os.path as osp
import glob
from multiprocessing import Pool, current_process
import sys
import torch.utils.data as data
import random
from numpy.random import randint
import numpy as np
from PIL import Image, ImageOps
import functools
import time
import yaml
import pprint
from dotmap import DotMap

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print("\nFinished {} in {} secs".format(repr(func.__name__), round(run_time, 4)))
        return value

    return wrapper

@timer
def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))

@timer
def download_vid(url, temp_vid_path):
    wget.download(url, out=temp_vid_path)

@timer
def setup_config(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(' ' * 30, "Config")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    print('-' * 80)
    config = DotMap(config)
    return config

def dump_frames(video_path, frame_out_dir, vid_name):
    video_file = os.listdir(video_path)[0]
    video_file = video_path + '/' + video_file
    vr = mmcv.VideoReader(video_file)
    for i in range(len(vr)):
        if vr[i] is not None:
            mmcv.imwrite(
                vr[i], '{}/img_{:05d}.jpg'.format(frame_out_dir, i + 1))
        else:
            print('[Warning] length inconsistent!'
                  'Early stop with {} out of {} frames'.format(i + 1, len(vr)))
            break
    print('{} done with {} frames'.format(vid_name, len(vr)))
    sys.stdout.flush()
    return True

@timer
def get_frames(temp_vid_path, temp_frames_subfolder, vid_name):
    # pool = Pool(args.num_worker)
    # pool.map(dump_frames, full_path)
    dump_frames(temp_vid_path, temp_frames_subfolder, vid_name)

def get_clip_emb(model, image):
    return model(image)

@timer
def create_temp_folders(args):
    temp_dir = args.temp_path
    if not os.path.isdir(temp_dir + '/' + 'temp_vid'):
        os.mkdir(temp_dir + '/' + 'temp_vid')
    temp_vid_path = temp_dir + '/' + 'temp_vid'

    if not os.path.isdir(temp_dir + '/' + 'temp_frames'):
        os.mkdir(temp_dir + '/' + 'temp_frames')
    temp_frames_folder = temp_dir + '/' + 'temp_frames'

    url = args.url.rsplit('/', 1)[1]
    vid_name = url.replace('.mp4', '')
    if not os.path.isdir(temp_frames_folder + '/' + vid_name):
        os.mkdir(temp_frames_folder + '/' + vid_name)
    temp_frames_subfolder = temp_frames_folder + '/' + vid_name

    args.temp_vid_path = temp_vid_path
    args.temp_frames_folder = temp_frames_folder
    args.temp_frames_subfolder = temp_frames_subfolder

    return args