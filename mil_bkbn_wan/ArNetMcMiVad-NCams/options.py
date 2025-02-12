import argparse


def parser_factory(
    device,
    seed,
    save_every,
    dataset_path,
    camera_list,
    feature_modal,
    feature_size,
    lambdas,
    alpha,
    max_epoch,
    model_name,
    sample_size,
    loss_combination,
    late_fusion,
    out_foldername,
    pretrained_ckpts = None, 
    ):
    parser = argparse.ArgumentParser(description='AR_Net')
    
    # customization
    parser.add_argument('--device', type=str, default=f"{device}", help='GPU ID')
    parser.add_argument('--dataset_path', type=str, default=f'{dataset_path}', help='path to dir contains anomaly datasets')
    parser.add_argument('--camera_list', type=list, default=camera_list, help='')
    parser.add_argument('--feature_modal', type=str, default=f'{feature_modal}', help='rgb, flow , combine')
    parser.add_argument('--Lambda', type=str, default=f'{lambdas}', help='')
    parser.add_argument('--k', type=int, default=alpha, help='value of k')
    parser.add_argument('--model_name', default=f"{model_name}", help=' ')
    parser.add_argument('--feature_size', type=int, default=feature_size, help='size of feature')
    parser.add_argument('--sample_size',  type=int, default=sample_size, help='number of samples in one itration')
    parser.add_argument('--max_epoch', type=int, default=max_epoch, help='maximum iteration to train')
    parser.add_argument('--save_every', type=int, default=save_every, help='save the model every X epochs')
    parser.add_argument('--loss_combination', type=str, default=f"{loss_combination}")
    parser.add_argument('--late_fusion', type=str, default=late_fusion)
    
    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--out_foldername', type=str, default=out_foldername)

    #parser.add_argument('--pretrained_ckpts', default=None)
    parser.add_argument('--pretrained_ckpts', type=list, default=pretrained_ckpts, help='')    

    
    parser.add_argument('--lr', type=float, default=0.0001,help='learning rate (default: 0.0001)')
    parser.add_argument('--loss_type', default='DMIL_C', type=str,  
                        help='the type of n_pair loss, max_min_2, max_min, attention, attention_median, attention_H_L or max')
    parser.add_argument('--pretrain', type=int, default=0)

    parser.add_argument('--testing_path', type=str, default=None, help='time file for test model')
    parser.add_argument('--testing_model', type=str, default=None, help='iteration name for testing model')
    parser.add_argument('--batch_size',  type=int, default=1, help='number of samples in one itration')
    parser.add_argument('--sample_step', type=int, default=1, help='')
    
    parser.add_argument('--max-seqlen', type=int, default=300, help='maximum sequence length during training (default: 750)')
    
    parser.add_argument('--feature_pretrain_model', type=str, default='i3d', help='type of feature to be used I3D or C3D (default: I3D)')
    parser.add_argument('--feature_layer', type=str, default='fc6', help='fc6 or fc7')
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
    
    return parser




