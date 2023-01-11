import yaml
import torch
import argparse
import os
from helpers.caption_helpers import *
import tempfile
import time

from vid_caption_manager import VideoCapManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--temp_path', default='')
    parser.add_argument('--device', required=True)
    parser.add_argument('--model_folder', required=True)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--url', type=str, required=True)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    args.url_list = ['https://cdn3.sharechat.com/video_url_1652094529943_529943_d92eaca_1630204262838.mp4',
    'https://cdn3.sharechat.com/video_url_1652096703633_703633_e47c693_1629654460146.mp4',
    'https://cdn3.sharechat.com/video_url_1652094782745_782745_2ad5faec_1628551052910.mp4',
    'https://cdn3.sharechat.com/video_url_1652095537797_537797_117d370e_1631773945044.mp4',
    'https://cdn3.sharechat.com/video_url_1652094883823_883823_301c93c7_1628247683149.mp4',
    'https://cdn3.sharechat.com/video_url_1652095030524_30524_12b482ab_1629143614169.mp4',
    'https://cdn3.sharechat.com/video_url_1652097324935_324935_29fc3f57_1633653839687.mp4',
    'https://cdn3.sharechat.com/video_url_1652095898049_898049_134eed57_1633180237953.mp4',]

    start = time.time()
    
    # create temp folders
    args = create_temp_folders(args) # created folders saved into args

    # get clip features
    vcap_manager = VideoCapManager(args)
    vcap_manager.get_captions()

    # # cleanup
    remove(vcap_manager.args.temp_vid_path)
    remove(vcap_manager.args.temp_frames_folder)
    
    print(f'Total Time Taken: {time.time() - start}')



if __name__ == '__main__':
    main()
