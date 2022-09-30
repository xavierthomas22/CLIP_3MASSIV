#!/usr/bin/env bash

python3 get_clip_features.py --config /home/xavier.thomas/projects/CLIP_3MASSIV/configs/3MASSIV.yaml \
    --url https://cdn3.sharechat.com/video_url_1652095168243_168243_7b7e76e_1633008176553.mp4 \
    --temp_path /home/xavier.thomas/projects/CLIP_3MASSIV --device cuda