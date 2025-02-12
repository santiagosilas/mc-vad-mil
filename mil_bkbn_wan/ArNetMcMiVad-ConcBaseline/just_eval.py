from __future__ import print_function
import os, sys
import torch
from model import model_generater
from video_dataset_anomaly_balance_uni_sample import dataset, dataset_train2test  # For anomaly
from torch.utils.data import DataLoader
from train import train
from tensorboard_logger import Logger
import options
import torch.optim as optim
import datetime
import time
import glob

from test import test
from eval import eval_p


if __name__ == '__main__':
    
    args = options.parser.parse_args()
    torch.manual_seed(args.seed)

    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(int(args.device)))
        torch.cuda.set_device(int(args.device))

    mytime = datetime.datetime.now()

    
    print("model to load:", args.pretrained_ckpt)
    model = model_generater(model_name=args.model_name, feature_size=args.feature_size).to(device)
    ckpt = torch.load(args.pretrained_ckpt)
    model.load_state_dict(ckpt)


    test_dataset = dataset(args=args, train=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, pin_memory=True,
                             num_workers=2, shuffle=False)

    test_result_dict = test(test_loader, model, device, args)
    results = eval_p(epoch=-1, itr=-1, dataset=args.dataset_name, predict_dict=test_result_dict, logger=None, 
            save_path=None, plot=args.plot, args=args)