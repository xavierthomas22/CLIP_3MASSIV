import clip
import yaml
from dotmap import DotMap
import pprint
from utils.tools import *
from utils.Augmentation import *
import torch
import argparse
import os
from helpers import *
import tempfile
import time

class TextCLIP(torch.nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self,text):
        return self.model.encode_text(text)

class ImageCLIP(torch.nn.Module):
    def __init__(self, model) :
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self,image):
        return self.model.encode_image(image)


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print("Finished {} in {} secs".format(repr(func.__name__), round(run_time, 3)))
        return value

    return wrapper

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='')
    parser.add_argument('--temp_path', default='')
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--url', type=str, required=True)
    args = parser.parse_args()

    start = time.time()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print(' ' * 30, "Config")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    print('-' * 80)

    config = DotMap(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, clip_state_dict = clip.load(config.network.arch,device=device) #Must set jit=False for training  ViT-B/32

    transform_image = get_augmentation(False,config) # False for val

    print('transforms: {}'.format(transform_image.transforms))

    # create temp folders
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


    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)
    model_text = torch.nn.DataParallel(model_text).cuda()
    model_image = torch.nn.DataParallel(model_image).cuda()

    if device == "cpu":
        model_text.float()
        model_image.float()
    else :
        clip.model.convert_weights(model_text) # Actually this line is unnecessary since clip by default already on float16
        clip.model.convert_weights(model_image)

    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain)
            model.load_state_dict(checkpoint['model_state_dict'])
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.resume)))

    if not os.listdir(temp_vid_path):
        print('\ndownloading video')
        download_vid(args.url, temp_vid_path)
    if not os.listdir(temp_frames_subfolder):
        print('\nprocessing frames')
        get_frames(temp_vid_path, temp_frames_subfolder, vid_name, args)
    
    frame_data = Frame_DATASET(frame_path=temp_frames_subfolder, num_segments=config.data.num_segments,image_tmpl=config.data.image_tmpl,random_shift=config.data.random_shift,
                       transform=transform_image)

    frame_indices = frame_data.sample_indices()

    processed_frames = frame_data.get(frame_indices, frame_path=temp_frames_subfolder)
    processed_frames = processed_frames.view((-1,config.data.num_segments,3)+processed_frames.size()[-2:])
    b,t,c,h,w = processed_frames.size()
    processed_frames = processed_frames.to(device).view(-1,c,h,w ) 

    frames_clip_emb = model_image(processed_frames).view(b, t, -1)
    mean_frames_clip_emb = torch.mean(frames_clip_emb, dim=1)

    print('CLIP Features extracted with dim: ', mean_frames_clip_emb.shape)

    # cleanup
    remove(temp_vid_path)
    remove(temp_frames_folder)
    
    print(f'Time: {time.time() - start}')



if __name__ == '__main__':
    main()