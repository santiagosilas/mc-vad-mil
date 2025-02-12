from __future__ import print_function
import os
import torch
from model import model_generater
from video_dataset_anomaly_balance_uni_sample import dataset
from torch.utils.data import DataLoader
from train import train
from tensorboard_logger import Logger
import options
import torch.optim as optim
import datetime
import time
import glob

from test import test_slf_n_cams
from eval import eval_p

def just_eval_slf_n_cams(parser):
    main_start = time.time()
    args = parser.parse_args() # args = options.parser.parse_args()
    torch.manual_seed(args.seed)
    
    


    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(int(args.device)))
        torch.cuda.set_device(int(args.device))

    mytime = datetime.datetime.now()

    models = {
        camera:model_generater(model_name=args.model_name, feature_size=args.feature_size, camera_list = args.camera_list).to(device)
        for camera in args.camera_list
    }

    ckpts = {
        camera:torch.load(pretrained_ckpt) for camera, pretrained_ckpt in zip(args.camera_list, args.pretrained_ckpts)
    }

    for camera in args.camera_list:
        models[camera].load_state_dict(ckpts[camera])

    test_dataset = dataset(args=args, train=False)
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, pin_memory=True,
                             num_workers=2, shuffle=False)

    test_result_dict = test_slf_n_cams(test_loader, models, device, args)
    
    results = eval_p(epoch = -1, itr=-1, 
                        camera_list = args.camera_list, 
                        predict_dict=test_result_dict, logger=None, 
                        save_path=None, plot=args.plot, args=args)
    
    epoch,itr,_auc,_fpr,_acc,_bacc,_prec,_rec,_f1, fpr, tpr, thresholds = results
