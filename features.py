import skvideo
import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset
import skvideo.io
from PIL import Image
import os
import h5py
import numpy as np
import random
import time
import clip

class VideoDataset(Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, videos_dir, video_names, score, video_format='RGB', width=None, height=None):

        super(VideoDataset, self).__init__()
        self.videos_dir = videos_dir
        self.video_names = video_names
        self.score = score
        self.format = video_format
        self.width = width
        self.height = height

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_name = self.video_names[idx]

        video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_name))
        video_score = self.score[idx]

        sample = {'name': video_name,
                  'video': video_data,
                  'score': video_score}
        return sample


def clip_feature(batch, text):
    print(batch.shape)
    if(batch.shape[1]==540):
        i = 0
        output = torch.Tensor().to(device)
        while (i< batch.shape[0]):
            h = 0
            output1 = torch.Tensor().to(device)
            while ((h + 224) <= 540):
            # print('h为{}'.format(h))
                w = 0
                while ((w + 224) <= 960):
                    b = batch[i, h:h + 224, w:w + 224,:]
                    b = Image.fromarray(b)
                  #  b.save('123.jpg')
                    image = preprocess(b).unsqueeze(0).to(device)
                    output1 = torch.cat((output1, image), 0)
                    w = w + 46
                h = h + 79

            logits_per_image, logits_per_text = clip_model(output1, text)
            o = logits_per_image.softmax(dim=-1)

            frame_feature = o.contiguous().view(-1).unsqueeze(0)
            output = torch.cat((output, frame_feature), 0)
            i = i+1
       # print("[540,960,3]")
        return output
    else:
        i = 0
        output = torch.Tensor().to(device)
        while (i < batch.shape[0]):
            h = 0
            output1 = torch.Tensor().to(device)
            while ((h + 224) <= 960):
                # print('h为{}'.format(h))
                w = 0
                while ((w + 224) <= 540):
                    b = batch[i, h:h + 224, w:w + 224, :]
                    b = Image.fromarray(b)
                    #  b.save('123.jpg')
                    image = preprocess(b).unsqueeze(0).to(device)
                    output1 = torch.cat((output1, image), 0)
                    w = w + 79
                h = h + 46

            logits_per_image, logits_per_text = clip_model(output1, text)
            o = logits_per_image.softmax(dim=-1)

            frame_feature = o.contiguous().view(-1).unsqueeze(0)
            output = torch.cat((output, frame_feature), 0)
            i = i + 1
      #  print("[960,540,3]")
        return output




def get_features(text, current_video, frame_batch_size=64, device= 'cuda:0'):
    """feature extraction"""
    video_length = current_video.shape[0]
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    output = torch.Tensor().to(device)
    with torch.no_grad():
        while frame_end < video_length:
            batch = current_video[frame_start:frame_end]
            clip_features = clip_feature(batch, text)
            output = torch.cat((output, clip_features), 0)
            frame_end += frame_batch_size
            frame_start += frame_batch_size

        last_batch = current_video[frame_start:video_length]
        clip_features = clip_feature(last_batch, text)
        output = torch.cat((output, clip_features), 0)
        print(output.shape)
    return output


if __name__ == "__main__":
    frame_batch_size = 64
    seed = 19920517
    torch.manual_seed(seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.utils.backcompat.broadcast_warning.enabled = True
    videos_dir = '../KoNViD_1k_videos'  # videos dir
    features_dir = '../../datasets/KoNViD/features/'  # features dir
    datainfo = '../data/KoNViD-1kinfo.mat'  # database info: video_names, scores; video format, width, height, index, ref_ids, max_len, etc.

    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    Info = h5py.File(datainfo, 'r')
    video_names = [Info[Info['video_names'][0, :][i]][()].tobytes()[::2].decode() for i in range(len(Info['video_names'][0, :]))]
    scores = Info['scores'][0, :]
    video_format = Info['video_format'][()].tobytes()[::2].decode()
    width = int(Info['width'][0])
    height = int(Info['height'][0])
    dataset = VideoDataset(videos_dir, video_names, scores, video_format, width, height)
    clip_model, preprocess = clip.load("../models/ViT-B-32.pt", device)
    text_list = ["good image block", "bad image block", "noisy image block", "hazy image block", "dark image block", "bright image block",
        "blurry image block", "over exposure image block", "sharp image block", "colorful image block", 'dull image block',
        "high contrast image block", "low contrast image block", "image block without noise", "image block without blur",
        "image block with additive gaussian noise", "image block with noise in color compression", "image block with spatially correlated noise",
        "image block with masked noise", "image block with high frequency noise", "image block with impulse noise",
        "image block with quantization noise", "image block with gaussian blur", "image block with motion blur", "image block with bokeh blur",
        "uniform color image block", "uneven color image block", "image block with chromatic aberration", "image block without chromatic aberration",
        "image block with distortions", "image block without distortions", "uniform illumination image block", "unevenly illuminated image block",
        "image block with sharpness loss", "image block without sharpness loss",
        "Light-hearted image block", "Depressing image block", "Comfortable image block", "Uncomfortable image block", "Sad image block",
        "Sentimental image block", "Fearful image block", "Exciting image block", "Satisfactory image block", "Calming image block",
        "Fascinating image block", "Interesting image block", "Impatient image block", "Tense image block", "Puzzling image block",
        "Delightful image block", "Outrageous image block", "Disgusting image block",
                 ]

    text = clip.tokenize(text_list).to(device)


    for i in range(0, len(dataset)):
        current_data = dataset[i]
        name = current_data['name'].split('_')[0]
        current_video = current_data['video']
        current_score = current_data['score']
        print('Video {}: length {}'.format(i, current_video1.shape[0]))
        start_time = time.time()
        features = get_features(text, current_video, frame_batch_size, device)
        end_time = time.time()
        run_time = end_time - start_time
        print(run_time)
        np.save(features_dir + str(i) + 'clip', features.to('cpu').numpy())
        np.save(features_dir + str(i) + '_score', current_score)
