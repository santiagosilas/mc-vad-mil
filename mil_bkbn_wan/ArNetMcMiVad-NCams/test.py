import torch

def test(test_loader, model, device, args):
    result = {}
    for i, data in enumerate(test_loader):

        features_cams, data_video_name = data
        for ds in args.camera_list:
            features_cams[ds] = features_cams[ds].to(device) 
        
        element_logits_cams = dict()
        with torch.no_grad():

            for ds in args.camera_list:
                if args.model_name == 'model_single':
                    _, element_logits_cams[ds] = model(features_cams[ds].float(), is_training=False)
                else:
                    _, element_logits_cams[ds] = model.forward_with_cam(ds, features_cams[ds].float(), is_training=False)

        for ds in args.camera_list:
            element_logits_cams[ds] = element_logits_cams[ds].cpu().data.numpy().reshape(-1)
        
        result[data_video_name[0]] = {ds:element_logits_cams[ds] for ds in args.camera_list }

    return result

def test_slf_n_cams(test_loader, models, device, args):
    result = {}
    for i, data in enumerate(test_loader):
        
        features_cams, data_video_name = data

        for ds in args.camera_list:
            features_cams[ds] = features_cams[ds].to(device) 
        
        element_logits_cams = dict()
        with torch.no_grad():
            for ds in args.camera_list:
                if args.model_name == 'model_single':
                    _, element_logits_cams[ds] = models[ds](features_cams[ds].float(), is_training=False)
                else:
                    _, element_logits_cams[ds] = models[ds].forward_with_cam(ds, features_cams[ds].float(), is_training=False)

        for ds in args.camera_list:
            element_logits_cams[ds] = element_logits_cams[ds].cpu().data.numpy().reshape(-1)
            print(ds)
            print(element_logits_cams[ds])

        result[data_video_name[0]] = {ds:element_logits_cams[ds] for ds in args.camera_list }

    return result