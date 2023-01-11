import tempfile
import os
import wget
import shutil
import os.path as osp
import glob
from multiprocessing import Pool, current_process
import sys
import functools
import time
import yaml
import pprint
import torch

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]


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
