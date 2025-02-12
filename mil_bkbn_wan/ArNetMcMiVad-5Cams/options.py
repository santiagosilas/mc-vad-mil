import argparse

parser = argparse.ArgumentParser(description='AR_Net')
parser.add_argument('--device', type=str, default="0", help='GPU ID')
parser.add_argument('--lr', type=float, default=0.0001,help='learning rate (default: 0.0001)')
parser.add_argument('--model_name', default='model_single', help=' ')
parser.add_argument('--loss_type', default='DMIL_C', type=str,  
                    help='the type of n_pair loss, max_min_2, max_min, attention, attention_median, attention_H_L or max')
parser.add_argument('--pretrain', type=int, default=0)

parser.add_argument('--pretrained_ckpt', default=None, help='ckpt for pretrained model')
parser.add_argument('--pretrained_ckpt1', default=None)
parser.add_argument('--pretrained_ckpt2', default=None)
parser.add_argument('--pretrained_ckpt3', default=None)
parser.add_argument('--pretrained_ckpt4', default=None)
parser.add_argument('--pretrained_ckpt5', default=None)

parser.add_argument('--testing_path', type=str, default=None, help='time file for test model')
parser.add_argument('--testing_model', type=str, default=None, help='iteration name for testing model')
parser.add_argument('--feature_size', type=int, default=2048, help='size of feature (default: 2048)')
parser.add_argument('--batch_size',  type=int, default=1, help='number of samples in one itration')
parser.add_argument('--sample_size',  type=int, default=30, help='number of samples in one itration')
parser.add_argument('--sample_step', type=int, default=1, help='')

parser.add_argument('--dataset_name_camA', type=str, default='001', help='')
parser.add_argument('--dataset_name_camB', type=str, default='002', help='')
parser.add_argument('--dataset_name_camC', type=str, default='003', help='')
parser.add_argument('--dataset_name_camD', type=str, default='004', help='')
parser.add_argument('--dataset_name_camE', type=str, default='005', help='')


# /Volumes/SILAS-HD/ACAD-2023/Datasets-Processed/single-cam/ICME-Original-Shanghaitech-Dataset
parser.add_argument('--dataset_path', type=str, default='/home/socialab/Downloads', help='path to dir contains anomaly datasets')
parser.add_argument('--save_every', type=int, default=5, help='save the model every X epochs')

parser.add_argument('--feature_modal', type=str, default='combine', help='features from different input, options contain rgb, flow , combine')
parser.add_argument('--max-seqlen', type=int, default=300, help='maximum sequence length during training (default: 750)')
parser.add_argument('--Lambda', type=str, default='1_20', help='')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

#parser.add_argument('--max_epoch', type=int, default=2, help='maximum iteration to train (default: 50000)')
#parser.add_argument('--max_epoch', type=int, default=20, help='maximum iteration to train (default: 50000)')
parser.add_argument('--max_epoch', type=int, default=20, help='maximum iteration to train (default: 50000)')

parser.add_argument('--feature_pretrain_model', type=str, default='i3d', help='type of feature to be used I3D or C3D (default: I3D)')
parser.add_argument('--feature_layer', type=str, default='fc6', help='fc6 or fc7')
parser.add_argument('--k', type=int, default=4, help='value of k')
parser.add_argument('--plot', type=int, default=0, help='whether plot the video anomalous map on testing')
# parser.add_argument('--rank', type=int, default=0, help='')
# parser.add_argument('--loss_instance_type', type=str, default='weight', help='mean, weight, weight_center or individual')
# parser.add_argument('--MIL_loss_type', type=str, default='CE', help='CE or MSE')
# parser.add_argument('--u_ratio', type=int, default=10, help='')
# parser.add_argument('--anomaly_smooth', type=int, default=1,
#                     help='type of smooth function, all or normal')
# parser.add_argument('--sparise_term', type=int, default=1,
#                     help='type of smooth function, all or normal')
# parser.add_argument('--attention_type', type=str, default='softmax',
#                     help='type of normalization of attention vector, softmax or sigmoid')
# parser.add_argument('--confidence', type=float, default=0, help='anomaly sample threshold')
parser.add_argument('--snapshot', type=int, default=200, help='anomaly sample threshold')
# parser.add_argument('--ps', type=str, default='normal_loss_mean')
parser.add_argument('--label_type', type=str, default='unary')

parser.add_argument('--loss_combination', type=str, default='Max(L_ca,L_cb,L_cc,L_cd)')
parser.add_argument('--late_fusion', type=str, default='maximum')


