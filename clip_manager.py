import torch
import clip
import os
from utils.Augmentation import *
from helpers import *

class ImageCLIP(torch.nn.Module):
    def __init__(self, model) :
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self,image):
        return self.model.encode_image(image)

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

class ClipManager():
    def __init__(self, args, config) :
        super(ClipManager, self).__init__()

        self.args = args
        self.config = config
        self.device = self.args.device
        self.model, self.clip_state_dict = clip.load(config.network.arch,device=self.device) #Must set jit=False for training  ViT-B/32
        self.transform_image = get_augmentation(False, self.config)

        self.model_image = ImageCLIP(self.model)

        if self.device == "cpu":
            self.model_image.float()
        else :
            self.model_image = torch.nn.DataParallel(self.model_image).cuda()
            clip.model.convert_weights(self.model_image)

        self.load_pretrain()
        self.download_and_process_vid()

        self.frame_data = Frame_DATASET(frame_path=self.args.temp_frames_subfolder, num_segments=self.config.data.num_segments,image_tmpl=self.config.data.image_tmpl,random_shift=self.config.data.random_shift,
                       transform=self.transform_image)
    
    @timer
    def load_pretrain(self):

        if os.path.isfile(self.config.pretrain):
            print(("=> loading checkpoint '{}'".format(self.config.pretrain)))
            checkpoint = torch.load(self.config.pretrain)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(self.config.resume)))

    @timer
    def download_and_process_vid(self):

        if not os.listdir(self.args.temp_vid_path):
            print('\ndownloading video')
            download_vid(self.args.url, self.args.temp_vid_path)
        url = self.args.url.rsplit('/', 1)[1]
        vid_name = url.replace('.mp4', '')
        if not os.listdir(self.args.temp_frames_subfolder):
            print('\nprocessing frames')
            get_frames(self.args.temp_vid_path, self.args.temp_frames_subfolder, vid_name)

    @timer
    def extract_features(self):

        frame_indices = self.frame_data.sample_indices()

        processed_frames = self.frame_data.get(frame_indices, frame_path=self.args.temp_frames_subfolder)
        processed_frames = processed_frames.view((-1, self.config.data.num_segments,3)+processed_frames.size()[-2:])
        b,t,c,h,w = processed_frames.size()
        processed_frames = processed_frames.to(self.device).view(-1,c,h,w ) 

        frames_clip_emb = get_clip_emb(self.model_image, processed_frames).view(b, t, -1)
        mean_frames_clip_emb = torch.mean(frames_clip_emb, dim=1)

        return mean_frames_clip_emb

