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


if __name__ == '__main__':
    main_start = time.time()
    args = options.parser.parse_args()
    torch.manual_seed(args.seed)
    


    if args.device == "cpu":
        print(f"set cpu device")
        device = torch.device("cpu")
    else:
        print(f"set gpu device", args.device)
        device = torch.device("cuda:{}".format(int(args.device)))
        torch.cuda.set_device(int(args.device))

    mytime = datetime.datetime.now()

    
    if args.model_name == "model_single":
        m_sufix = ""
    elif args.model_name == "model_single_mtv": 
        m_sufix = "-Mtv"
    elif args.model_name == "model_single_mtv2":
        m_sufix = "-Mtv2"
    else:
        raise
    
    save_path = os.path.join(args.model_name, args.feature_pretrain_model, f"{args.dataset_name_camA}-{args.dataset_name_camB}-{args.dataset_name_camC}-{args.dataset_name_camD}-{args.dataset_name_camE}", 
                             'k_{}'.format(args.k), '_Lambda_{}'.format(args.Lambda), 
                             args.feature_modal, 
                             '{}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}-SS{}-LC{}-LF{}{}'.format(mytime.year, mytime.month, mytime.day, mytime.hour,
                                                                                 mytime.minute, mytime.second, 
                                                                                 args.sample_size,
                                                                                 args.loss_combination,
                                                                                 args.late_fusion,
                                                                                 m_sufix
                                                                                 ))

    print("path to save", save_path)  

    model = model_generater(model_name=args.model_name, feature_size=args.feature_size).to(device)


    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    
    if args.pretrained_ckpt is not None:
        print("model.load_state_dict ..")
        model.load_state_dict(torch.load(args.pretrained_ckpt))


    train_dataset = dataset(args=args, train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, pin_memory=True,
                              num_workers=1, shuffle=True)
    test_dataset = dataset(args=args, train=False)
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, pin_memory=True,
                             num_workers=2, shuffle=False)

    ckpt_path = os.path.join(os.path.join(os.path.join(args.dataset_path, f"outputs-{args.dataset_name_camA}-{args.dataset_name_camB}-{args.dataset_name_camC}-{args.dataset_name_camD}-{args.dataset_name_camE}"), "ckpt"), save_path)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    print(ckpt_path)
    
    logs_path = os.path.join(os.path.join(os.path.join(args.dataset_path, f"outputs-{args.dataset_name_camA}-{args.dataset_name_camB}-{args.dataset_name_camC}-{args.dataset_name_camD}-{args.dataset_name_camE}"), "logs"), save_path)
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    print(logs_path)

    logger = Logger(logs_path)

    results_path = os.path.join(os.path.join(os.path.join(args.dataset_path, f"outputs-{args.dataset_name_camA}-{args.dataset_name_camB}-{args.dataset_name_camC}-{args.dataset_name_camD}-{args.dataset_name_camE}"), "result"), save_path)
    print(results_path)

    train(epochs=args.max_epoch, train_loader=train_loader, test_loader=test_loader, args=args, 
          model=model, optimizer=optimizer, logger=logger, device=device, save_path=save_path, 
          ckpt_path = ckpt_path,logs_path=logs_path, results_path = results_path)
            
    main_end = time.time()
    
    elapsed_time = main_end - main_start
    
    print("elapsed time:")
    print("\t", elapsed_time, "seconds.")
    print("\t", elapsed_time / 60, "mins.")
