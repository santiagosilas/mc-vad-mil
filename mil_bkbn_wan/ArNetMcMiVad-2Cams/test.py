import torch
# import torch.nn.functional as F
# import utils
# import numpy as np
# from torch.autograd import Variable
# import scipy.io as sio

def test(test_loader, model, device, args):
    result = {}
    for i, data in enumerate(test_loader):

        feature_camA, feature_camB, data_video_name = data

        feature_camA = feature_camA.to(device)
        feature_camB = feature_camB.to(device) 
        
        #print("test-loader>>test video", data_video_name)

        with torch.no_grad():
            #if args.model_name == 'model_lstm':
            #    _, element_logits = model(feature, seq_len=None, is_training=False)
            #else:

            if args.model_name == 'model_single':
                _, element_logits_camA = model(feature_camA.float(), is_training=False)
            else:
                # *.forward_cam1
                _, element_logits_camA = model.forward_camA(feature_camA.float(), is_training=False)
            
            if args.model_name == 'model_single':
                _, element_logits_camB = model(feature_camB.float(), is_training=False)
            else:
                # *.forward_cam2
                _, element_logits_camB = model.forward_camB(feature_camB.float(), is_training=False)

        element_logits_camA = element_logits_camA.cpu().data.numpy().reshape(-1)
        element_logits_camB = element_logits_camB.cpu().data.numpy().reshape(-1)
        
        result[data_video_name[0]] = {"camA":element_logits_camA, "camB":element_logits_camB}
    
    return result


def test_slf_2cams(test_loader, model1, model2, device, args):
    result = {}
    for i, data in enumerate(test_loader):

        feature_camA, feature_camB, data_video_name = data

        feature_camA = feature_camA.to(device)
        feature_camB = feature_camB.to(device) 
        
        #print("test-loader>>test video", data_video_name)

        with torch.no_grad():
            #if args.model_name == 'model_lstm':
            #    _, element_logits = model(feature, seq_len=None, is_training=False)
            #else:

            if args.model_name == 'model_single':
                _, element_logits_camA = model1(feature_camA.float(), is_training=False)
            else:
                # *.forward_cam1
                _, element_logits_camA = model1.forward_camA(feature_camA.float(), is_training=False)
            
            if args.model_name == 'model_single':
                _, element_logits_camB = model2(feature_camB.float(), is_training=False)
            else:
                # *.forward_cam2
                _, element_logits_camB = model2.forward_camB(feature_camB.float(), is_training=False)

        element_logits_camA = element_logits_camA.cpu().data.numpy().reshape(-1)
        element_logits_camB = element_logits_camB.cpu().data.numpy().reshape(-1)
        
        result[data_video_name[0]] = {"camA":element_logits_camA, "camB":element_logits_camB}
    
    return result





