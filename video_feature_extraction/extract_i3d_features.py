"""
For video feature extraction, we use the public github repository https://github.com/v-iashin/video_features.
"""

import torch
import gc
import cv2
import time
import os
import sys
import numpy as np

import warnings
warnings.filterwarnings("ignore")

gc.collect()
torch.cuda.empty_cache()

sys.path.insert(0,".../video_features")
os.chdir(".../video_features")


from models.i3d.extract_i3d import ExtractI3D
from utils.utils import build_cfg_path
from omegaconf import OmegaConf

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#torch.cuda.get_device_name(0)


def count_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames


def extract_from_mp4(mp4_path):
    start_extraction = time.time()    
    feature_type = 'i3d'
    args = OmegaConf.load(build_cfg_path(feature_type))

    '''
    The number of frames from which to extract features (or window size).
    '''
    args.stack_size = 16

    '''
    The number of frames to step before extracting the next features.
    '''
    args.step_size = 16


    args.extraction_fps = 30

    '''
    By default, the flow-features of I3D will be calculated using optical from
    calculated with PWCNet (originally with TV-L1). Another supported model is raft.
    '''
    args.flow_type = 'raft'

    '''
    If print, the features are printed to the terminal.
    If save_numpy or save_pickle, the features are saved
    to either .npy file or .pkl.
    '''
    args.on_extraction = 'print' # save_numpy
    #args.output_path = './output'

    args.device = "cuda:0"

    # Load the model
    extractor = ExtractI3D(args)

    # Extract features
    num_frames = count_frames(mp4_path)
    print(f"{mp4_path.split(os.path.sep)[-1]}",f"Frames:", num_frames, "Expectation:", num_frames//16,'Remains',num_frames%16)
    feature_dict = extractor.extract(mp4_path)
    print(feature_dict["rgb"].shape, feature_dict["flow"].shape)
    elapsed_extraction = time.time() - start_extraction
    print(f"{elapsed_extraction/60} seconds")
    return feature_dict


# base path
path_base = ".../MP4-Files"
path_i3d_output = ".../I3D-Files"

print(path_base)

avi_files = [f for f in os.listdir(path_base)]

path_tuples = list()
for avi_file in avi_files:
    src_path = os.path.join(path_base, avi_file)
    #dst_path = os.path.join(path_i3d_output, avi_file + ".npy")
    dst_path = os.path.join(path_i3d_output, avi_file.replace(".mp4",".npy"))
    if not os.path.exists(dst_path):
        path_tuples.append([src_path, dst_path])
    else:
        print("already extracted:", src_path)

print("number of videos to proccess:", len(path_tuples))


print("num tuples", len(path_tuples))
for index_path_tuple, path_tuple in enumerate(path_tuples):
    src_path, dst_path = path_tuple
    print(index_path_tuple, "of", len(path_tuples))
    print("src:", src_path)
    print("dst:", dst_path)
    print()
    feature_dict = extract_from_mp4(src_path)
    np.save(dst_path, feature_dict)
