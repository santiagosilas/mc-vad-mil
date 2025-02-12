import torch
# import torch.nn.functional as F
# import utils
# import numpy as np
# from torch.autograd import Variable
# import scipy.io as sio

def test(test_loader, model, device, args):
    result = {}
    for i, data in enumerate(test_loader):

        feature_camA, feature_camB, feature_camC, feature_camD, data_video_name = data

        feature_camA = feature_camA.to(device)
        feature_camB = feature_camB.to(device)
        feature_camC = feature_camC.to(device) 
        feature_camD = feature_camD.to(device) 
        
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


            if args.model_name == 'model_single':
                _, element_logits_camC = model(feature_camC.float(), is_training=False)
            else:
                # *.forward_cam2
                _, element_logits_camC = model.forward_camC(feature_camC.float(), is_training=False)

            if args.model_name == 'model_single':
                _, element_logits_camD = model(feature_camD.float(), is_training=False)
            else:
                # *.forward_cam2
                _, element_logits_camD = model.forward_camD(feature_camD.float(), is_training=False)


        element_logits_camA = element_logits_camA.cpu().data.numpy().reshape(-1)
        element_logits_camB = element_logits_camB.cpu().data.numpy().reshape(-1)
        element_logits_camC = element_logits_camC.cpu().data.numpy().reshape(-1)
        element_logits_camD = element_logits_camD.cpu().data.numpy().reshape(-1)
        
        result[data_video_name[0]] = {
            "camA":element_logits_camA, 
            "camB":element_logits_camB, 
            "camC":element_logits_camC, 
            "camD":element_logits_camD}
    
    return result


def test_slf_4cams(test_loader, model1, model2,model3,model4, device, args):
    result = {}
    for i, data in enumerate(test_loader):

        feature_camA, feature_camB, feature_camC,feature_camD, data_video_name = data

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


            if args.model_name == 'model_single':
                _, element_logits_camC = model3(feature_camC.float(), is_training=False)
            else:
                # *.forward_cam2
                _, element_logits_camC = model3.forward_camC(feature_camC.float(), is_training=False)

            if args.model_name == 'model_single':
                _, element_logits_camD = model4(feature_camD.float(), is_training=False)
            else:
                # *.forward_cam2
                _, element_logits_camD = model4.forward_camD(feature_camD.float(), is_training=False)




        element_logits_camA = element_logits_camA.cpu().data.numpy().reshape(-1)
        element_logits_camB = element_logits_camB.cpu().data.numpy().reshape(-1)
        element_logits_camC = element_logits_camC.cpu().data.numpy().reshape(-1)
        element_logits_camD = element_logits_camD.cpu().data.numpy().reshape(-1)
        
        result[data_video_name[0]] = {
            "camA":element_logits_camA, 
            "camB":element_logits_camB,
            "camC":element_logits_camC,
            "camD":element_logits_camD,
        }
    
    return result


def test_slf_3cams(test_loader, model1, model2, model3, device, args):
    result = {}
    for i, data in enumerate(test_loader):

        feature_camA, feature_camB, feature_camC, data_video_name = data

        feature_camA = feature_camA.to(device)
        feature_camB = feature_camB.to(device)
        feature_camC = feature_camC.to(device) 
        
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

            if args.model_name == 'model_single':
                _, element_logits_camC = model2(feature_camC.float(), is_training=False)
            else:
                # *.forward_cam2
                _, element_logits_camC = model2.forward_camC(feature_camC.float(), is_training=False)

        element_logits_camA = element_logits_camA.cpu().data.numpy().reshape(-1)
        element_logits_camB = element_logits_camB.cpu().data.numpy().reshape(-1)
        element_logits_camC = element_logits_camC.cpu().data.numpy().reshape(-1)
        element_logits_camD = element_logits_camD.cpu().data.numpy().reshape(-1)
        
        result[data_video_name[0]] = {"camA":element_logits_camA, "camB":element_logits_camB, "camC":element_logits_camC, "camD":element_logits_camD}
        
    return result


def test_slf_4cams(test_loader, model1, model2, model3, model4, device, args):
    result = {}
    for i, data in enumerate(test_loader):

        feature_camA, feature_camB, feature_camC,feature_camD, data_video_name = data

        feature_camA = feature_camA.to(device)
        feature_camB = feature_camB.to(device)
        feature_camC = feature_camC.to(device)
        feature_camD = feature_camD.to(device) 
        
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

            if args.model_name == 'model_single':
                _, element_logits_camC = model2(feature_camC.float(), is_training=False)
            else:
                # *.forward_cam2
                _, element_logits_camC = model2.forward_camC(feature_camC.float(), is_training=False)


            if args.model_name == 'model_single':
                _, element_logits_camD = model2(feature_camD.float(), is_training=False)
            else:
                # *.forward_cam2
                _, element_logits_camD = model2.forward_camD(feature_camD.float(), is_training=False)


        element_logits_camA = element_logits_camA.cpu().data.numpy().reshape(-1)
        element_logits_camB = element_logits_camB.cpu().data.numpy().reshape(-1)
        element_logits_camC = element_logits_camC.cpu().data.numpy().reshape(-1)
        element_logits_camD = element_logits_camD.cpu().data.numpy().reshape(-1)
        
        result[data_video_name[0]] = {
            "camA":element_logits_camA, 
            "camB":element_logits_camB, 
            "camC":element_logits_camC, 
            "camD":element_logits_camD}
    
    return result


