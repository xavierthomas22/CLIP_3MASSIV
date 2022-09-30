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

def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))

def download_vid(url, temp_vid_path):
    wget.download(url, out=temp_vid_path)


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


def get_frames(temp_vid_path, temp_frames_subfolder, vid_name, args):
    # pool = Pool(args.num_worker)
    # pool.map(dump_frames, full_path)
    dump_frames(temp_vid_path, temp_frames_subfolder, vid_name)

class Frame_DATASET(data.Dataset):
    def __init__(self, frame_path, num_segments=1, new_length=1,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False, index_bias=1):

        self.num_segments = num_segments
        self.seg_length = new_length
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.loop=False
        self.index_bias = index_bias

        if self.index_bias is None:
            if self.image_tmpl == "frame{:d}.jpg":
                self.index_bias = 0
            else:
                self.index_bias = 1

        self.initialized = False

        self.frame_path = frame_path
        self.num_frames = sum(len(files) for _, _, files in os.walk(self.frame_path))

        self.total_length = self.num_segments * self.seg_length

    def load_image(self, directory, idx):
        return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        
    def sample_indices(self):
        if self.num_frames <= self.total_length:
            if self.loop:
                return np.mod(np.arange(
                    self.total_length) + randint(self.num_frames // 2),
                    self.num_frames) + self.index_bias
            offsets = np.concatenate((
                np.arange(self.num_frames),
                randint(self.num_frames,
                        size=self.total_length - self.num_frames)))
            return np.sort(offsets) + self.index_bias
        offsets = list()
        ticks = [i * self.num_frames // self.num_segments
                 for i in range(self.num_segments + 1)]

        for i in range(self.num_segments):
            tick_len = ticks[i + 1] - ticks[i]
            tick = ticks[i]
            if tick_len >= self.seg_length:
                tick += randint(tick_len - self.seg_length + 1)
            offsets.extend([j for j in range(tick, tick + self.seg_length)])
        return np.array(offsets) + self.index_bias

    def get(self, indices, frame_path):
        images = list()
        for i, seg_ind in enumerate(indices):
            p = int(seg_ind)
            try:
                seg_imgs = self.load_image(frame_path, p)
            except OSError:
                print('ERROR: Could not read image "{}"'.format(record.path))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(seg_imgs)
        process_data = self.transform(images)
        return process_data

    