import joblib
import torch
import gc
import urllib.request
from src.utils.comm import dist_init
from src.utils.miscellaneous import set_seed
from src.modeling.load_swin import get_swin_model
from src.modeling.load_bert import get_bert_model
from src.modeling.video_captioning_e2e_vid_swin_bert import VideoTransformer
from src.datasets.caption_tensorizer import build_tensorizer
from src.tasks.run_caption_VidSwinBert_inference import inference, _online_video_decode, _transforms

from helpers.caption_helpers import *

import json
from tqdm import tqdm

class VideoCapManager(object):
    def __init__(self, args):
            torch.manual_seed(1024)
            self.model_folder = args.model_folder
            f = open('/home/xavier.thomas/projects/SwinBERT_org/3massiv_output/log/args.json')
            # self.args = json.load(f)
            self.args_model = joblib.load(f'{self.model_folder}/args.pkl')
            self.args_model.model_name_or_path = f'{self.model_folder}/captioning/bert-base-uncased/'

            self.args_model.num_gpus = 4
            self.args_model.distributed=True
            self.args_model.device = torch.device('cuda')

            self.args = args
            
            dist_init(self.args_model)
            set_seed(self.args_model.seed, self.args_model.num_gpus)
            fp16_trainning = None
            
            swin_model = get_swin_model(self.args_model)
            bert_model, config, self.tokenizer = get_bert_model(self.args_model)
            
            self.vl_transformer = VideoTransformer(self.args_model, config, swin_model, bert_model) 
            self.vl_transformer.freeze_backbone(freeze=self.args_model.freeze_backbone)
            
            # cpu_device = torch.device('cpu')
            cpu_device = torch.device('cuda')
            pretrained_model = torch.load(f'{self.model_folder}/model.bin', map_location=cpu_device)
            # pretrained_model = torch.load(f'/home/xavier.thomas/projects/SwinBERT_org/3massiv_output/checkpoint-90/model.bin', map_location=cpu_device)

            if isinstance(pretrained_model, dict):
                self.vl_transformer.load_state_dict(pretrained_model, strict=False)
            else:
                self.vl_transformer.load_state_dict(pretrained_model.state_dict(), strict=False)

            self.vl_transformer.to(self.args_model.device)
            self.vl_transformer.eval()
            
            self.tensorizer = build_tensorizer(self.args_model, self.tokenizer, is_train=False)        

            self.download_and_process_vid()

    @timer
    def download_and_process_vid(self):

        for url in self.args.url_list:
            url_name = url.rsplit('/', 1)[1]
            vid_name = url_name.replace('.mp4', '')
            if not os.path.isfile(f'{self.args.temp_vid_path}/{url_name}'):
                print(f'\ndownloading video {vid_name}')
                download_vid(url, self.args.temp_vid_path)
    
        # if not os.listdir(self.args.temp_frames_subfolder):
        #     print('\nprocessing frames')
        #     get_frames(self.args.temp_vid_path, self.args.temp_frames_subfolder, vid_name)
    
    def get_single_caption(self,url):
        urllib.request.urlretrieve(url, f'{self.model_folder}/video.mp4') 
        caption, conf = inference(self.args_model, f'{self.model_folder}/video.mp4', self.vl_transformer, self.tokenizer, self.tensorizer)
        return caption

    @timer
    def get_captions(self):

        cls_token_id, sep_token_id, pad_token_id, mask_token_id, period_token_id = \
            self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token, self.tokenizer.sep_token,
            self.tokenizer.pad_token, self.tokenizer.mask_token, '.'])

        self.vl_transformer.float()
        self.vl_transformer.eval()

        dataset = []
        
        for video_path in os.listdir(self.args.temp_vid_path):
            video_path = f'{self.args.temp_vid_path}/{video_path}'

            frames = _online_video_decode(self.args_model, video_path)
            preproc_frames = _transforms(self.args_model, frames)
            data_sample = self.tensorizer.tensorize_example_e2e('', preproc_frames)
            data_sample = tuple(t.to(self.args_model.device) for t in data_sample)

            dataset.append(data_sample)

        print(len(dataset))
        dataset = CustomDataset(dataset)

        from torch.utils.data import TensorDataset, DataLoader
        my_dataloader = DataLoader(dataset, batch_size=4) # create your dataloader

        time_meter = 0
        all_res = []
        
        with torch.no_grad():
            args = self.args_model
            for step, (batch) in tqdm(enumerate(my_dataloader)):
            
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {'is_decode': True,
                    'input_ids': batch[0], 'attention_mask': batch[1],
                    'token_type_ids': batch[2], 'img_feats': batch[3],
                    'masked_pos': batch[4],
                    'do_sample': False,
                    'bos_token_id': cls_token_id,
                    'pad_token_id': pad_token_id,
                    'eos_token_ids': [sep_token_id],
                    'mask_token_id': mask_token_id,
                    # for adding od labels
                    'add_od_labels': args.add_od_labels, 'od_labels_start_posid': args.max_seq_a_length,
                    # hyperparameters of beam search
                    'max_length': args.max_gen_length,
                    'num_beams': args.num_beams,
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "top_p": args.top_p,
                    "repetition_penalty": args.repetition_penalty,
                    "length_penalty": args.length_penalty,
                    "num_return_sequences": args.num_return_sequences,
                    "num_keep_best": args.num_keep_best,
                }

                tic = time.time()
                outputs = self.vl_transformer(**inputs)

                time_meter += time.time() - tic
                all_caps = outputs[0]  # batch_size * num_keep_best * max_len
                all_confs = torch.exp(outputs[1])

                for caps, confs in zip(all_caps, all_confs):
                    res = []
                    for cap, conf in zip(caps, confs):
                        cap = self.tokenizer.decode(cap.tolist(), skip_special_tokens=True)
                        res.append({'caption': cap, 'conf': conf.item()})
                    all_res.append(res)
                    
        print(all_res)
                



        