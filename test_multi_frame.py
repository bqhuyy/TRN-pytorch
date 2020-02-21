# import the necessary packages
from __future__ import print_function
import argparse
import cv2
import pandas as pd
import os
import re
import functools
import subprocess
import numpy as np
from PIL import Image
import moviepy.editor as mpy
import torchvision
import torch.nn.parallel
import torch.optim
from torch.autograd import Variable
from models import TSN
import transforms
from torch.nn import functional as F

def putIterationsPerSec(frame, label):
    """
    Add iterations per second text to lower-left corner of a frame.
    """

    cv2.putText(frame, "class label: " + label,
                (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return frame

def load_frames(frames, num_frames=8):
    if len(frames) >= num_frames:
        return frames[::int(np.ceil(len(frames) / float(num_frames)))]
    else:
        raise (ValueError('Video must have at least {} frames'.format(num_frames)))

parser = argparse.ArgumentParser(description="test TRN on a single video")
# group = parser.add_mutually_exclusive_group(required=True)
parser.add_argument('--video_path', type=str, default='input/demo.mp4')
parser.add_argument('--modality', type=str, default='RGB',
                    choices=['RGB', 'Flow', 'RGBDiff'], )
parser.add_argument('--dataset', type=str, default='jester',
                    choices=['something', 'jester', 'moments', 'somethingv2'])
parser.add_argument('--rendered_output', type=str, default='test')
parser.add_argument('--arch', type=str, default="InceptionV3")
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--test_segments', type=int, default=8)
parser.add_argument('--img_feature_dim', type=int, default=256)
parser.add_argument('--consensus_type', type=str, default='TRNmultiscale')
parser.add_argument('--weights', type=str, default='model/TRN_jester_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar')
parser.add_argument('--output_path', type=str, default='output')
args = parser.parse_args()

# Get dataset categories.
categories_file = 'pretrain/{}_categories.txt'.format(args.dataset)
categories = [line.rstrip() for line in open(categories_file, 'r').readlines()]
num_class = len(categories)

args.arch = 'InceptionV3' if args.dataset == 'moments' else 'BNInception'

# Load model.
model = TSN(num_class,
          args.test_segments,
          args.modality,
          base_model=args.arch,
          consensus_type=args.consensus_type,
          img_feature_dim=args.img_feature_dim, print_spec=False)

# for idx, m in enumerate(model.modules()):
#     print(idx, '->', m)
# exit()

checkpoint = torch.load(args.weights)
base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}

model.load_state_dict(base_dict)
model.cuda().eval()

# Print model's state_dict
# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())


# for idx, m in enumerate(model.modules()):
#     print(idx, '->', m)
# exit()

# Initialize frame transforms.
transform = torchvision.transforms.Compose([
    transforms.GroupScale(model.scale_size),
    transforms.GroupCenterCrop(model.input_size),
    transforms.Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
    transforms.ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
    transforms.GroupNormalize(model.input_mean, model.input_std),
])

pred = "";
bufferf = []
video = cv2.VideoCapture(args.video_path)
img_array = []
filename = args.video_path.split('/')[-1].split('.')[0] 

# loop over some frames...this time using the threaded stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    ret, frame = video.read() # B, G, R order

    if not ret:
        break
    
    frame = putIterationsPerSec(frame, pred)

    # frame = cv2.resize(frame, (640, 480))

    crop_img = frame[0:720, 0:720]
    # h,w = crop_img.shape[:2]
    input_img = crop_img;
    input_pill = Image.fromarray(input_img)
    img_array.append(crop_img)

    if (len(bufferf) < 16):
        bufferf.append(input_pill)
    else:
        input_frames = load_frames(bufferf)
        print(input_frames)
        data = transform(input_frames)
        input = Variable(data.view(-1, 3, data.size(1), data.size(2)).unsqueeze(0).cuda(), volatile=True)
        
        # with torch.no_grad():
        logits = model(input)
        np.save(f'{args.output_path}/{filename}', logits.data.cpu().numpy().copy())
        print(logits.data.cpu().numpy().copy())
        exit()
        
        h_x = torch.mean(F.softmax(logits, 1), dim=0).data
        probs, idx = h_x.sort(0, True)
        pred = categories[idx[0]]
        bufferf[:-1] = bufferf[1:];
        bufferf[-1] = input_pill
        # check to see if the frame should be displayed to our screen
    
if not os.path.isdir(args.output_path):
    print(f'Outpur dir {args.output_path} does not exist. Creating...')
    os.mkdir(args.output_path)   

# height, width, layers = img_array[0].shape
# size = (width, height)
# output = cv2.VideoWriter(f'{args.output_path}/{filename}.avi', cv2.VideoWriter_fourcc(*'XVID'), 24, size)

# for i in range(len(img_array)):
#     output.write(img_array[i])
# output.release()

clip = mpy.ImageSequenceClip(img_array, fps=24)
clip.write_videofile(f'{args.output_path}/{filename}.mp4')  
