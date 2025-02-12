import torch
import numpy as np
from test import test
from eval import eval_p
import os
import pickle
import time
from losses import KMXMILL_individual, normal_smooth

def train(epochs, train_loader, test_loader, args, model, optimizer, logger, device, save_path, 
          ckpt_path,logs_path, results_path):
    
    print(args)

    itr = 0
    if os.path.exists(results_path) == 0:
        os.makedirs(results_path)
    with open(file=os.path.join(results_path, 'result.txt'), mode='w') as f:
        for key, value in vars(args).items():
            f.write('%s:%s\n' % (key, value))
    log_statics = {}
    
    #if args.pretrained_ckpt:
    #    checkpoint = torch.load(args.pretrained_ckpt)
    #    model.load_state_dict(checkpoint)
    #    print('model load weights from {}'.format(args.pretrained_ckpt))
    #else:
    #    print('model is trained from scratch')
    start_loop = time.time()

    with open(file=os.path.join(ckpt_path, 'params.txt'), mode='a+') as f:
        f.write(str(args))

    performance_results = list()
    for epoch in range(1, epochs+1):
        #print("epoch", epoch)
        for i, data in enumerate(train_loader):
            itr += 1
            anomaly_features_cams, normaly_features_cams, anomaly_label, normaly_label, train_video_name, start_index_cams, len_index_cams = data
            
            if args.loss_combination == "PairwiseMeanOfClipScoresInBags":
                
                features_cams = dict()
                videolabels_cams = dict()
                seq_len_cams = dict()
                element_logits_cams = dict()
                final_features_cams = dict()

                for ds in args.camera_list:
                    features_cams[ds] = torch.cat((anomaly_features_cams[ds].squeeze(0), normaly_features_cams[ds].squeeze(0)), dim=0)
                    videolabels_cams[ds] = torch.cat((anomaly_label.squeeze(0), normaly_label.squeeze(0)), dim=0)
                    seq_len_cams[ds] = torch.sum(torch.max(features_cams[ds].abs(), dim=2)[0] > 0, dim=1).numpy()
                    features_cams[ds] = features_cams[ds][:, :np.max(seq_len_cams[ds]), :]
                    features_cams[ds] = features_cams[ds].float().to(device)
                    videolabels_cams[ds] = videolabels_cams[ds].float().to(device)
                    if args.model_name == 'model_single':
                        final_features_cams[ds], element_logits_cams[ds] = model(features_cams[ds])
                    else:
                        final_features_cams[ds], element_logits_cams[ds] = model.forward_with_cam(ds, features_cams[ds])

                weights = args.Lambda.split('_')
                
                m = torch.stack([element_logits_cams[ds] for ds in args.camera_list])
                element_logits = torch.mean(m, axis = 0)

                ds0 = args.camera_list[0]
                m_loss = KMXMILL_individual(element_logits=element_logits, seq_len=seq_len_cams[ds0], 
                    labels=videolabels_cams[ds0], device=device, loss_type='CE', args=args)
                n_loss = normal_smooth(element_logits=element_logits, labels=videolabels_cams[ds0], device=device)
                total_loss = float(weights[0]) * m_loss + float(weights[1]) * n_loss

                logger.log_value('m_loss', m_loss, itr)
                logger.log_value('n_loss', n_loss, itr)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                continue

            
            m_loss_cams = dict()
            n_loss_cams = dict()
            total_loss_cams = dict()
            features_cams = dict()
            videolabels_cams = dict()
            seq_len_cams = dict()
            final_features_cams = dict()
            element_logits_cams = dict()
            for ds in args.camera_list:
                features_cams[ds] = torch.cat((anomaly_features_cams[ds].squeeze(0), normaly_features_cams[ds].squeeze(0)), dim=0)
                videolabels_cams[ds] = torch.cat((anomaly_label.squeeze(0), normaly_label.squeeze(0)), dim=0)
                seq_len_cams[ds] = torch.sum(torch.max(features_cams[ds].abs(), dim=2)[0] > 0, dim=1).numpy()
                features_cams[ds] = features_cams[ds][:, :np.max(seq_len_cams[ds]), :]
                features_cams[ds] = features_cams[ds].float().to(device)
                videolabels_cams[ds] = videolabels_cams[ds].float().to(device)
                if args.model_name == 'model_single':
                    final_features_cams[ds], element_logits_cams[ds] = model(features_cams[ds])
                else:
                    final_features_cams[ds], element_logits_cams[ds] = model.forward_camA(features_cams[ds])

                weights = args.Lambda.split('_')
                m_loss_cams[ds] = KMXMILL_individual(element_logits=element_logits_cams[ds], seq_len=seq_len_cams[ds], labels=videolabels_cams[ds], device=device, loss_type='CE', args=args)
                n_loss_cams[ds] = normal_smooth(element_logits=element_logits_cams[ds], labels=videolabels_cams[ds], device=device)
                total_loss_cams[ds] = float(weights[0]) * m_loss_cams[ds] + float(weights[1]) * n_loss_cams[ds]
                logger.log_value('m_loss', m_loss_cams[ds], itr)
                logger.log_value('n_loss', n_loss_cams[ds], itr)



            # max
            if args.loss_combination == "Max":
                ds0 = args.camera_list[0]
                total_loss = total_loss_cams[ds0]
                for ds in args.camera_list:
                    total_loss = torch.max(total_loss, total_loss_cams[ds])


            elif args.loss_combination == "Min":
                raise
                total_loss = torch.min(total_loss_camA, torch.min(total_loss_camB, torch.min(total_loss_camC, total_loss_camD)))
            elif args.loss_combination == "LC":
                raise
                total_loss = (total_loss_camA + total_loss_camB + total_loss_camC+ total_loss_camD)/4
            elif args.loss_combination == "OnlyLossCamA":
                raise
                total_loss = total_loss_camA
            elif args.loss_combination == "OnlyLossCamB":
                raise
                total_loss = total_loss_camB
            elif args.loss_combination == "OnlyLossCamC":
                raise
                total_loss = total_loss_camC
            elif args.loss_combination == "OnlyLossCamD":
                raise
                total_loss = total_loss_camD
            else:
                raise

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            

        if epoch % args.save_every == 0:
            print('Iteration:{}, Loss: {}'.format(itr,total_loss.data.cpu().detach().numpy()))    
            model_path_saved_on = os.path.join(ckpt_path, 'epoch_{}_iter_{}'.format(epoch,itr) + '.pkl')

            print("model path", model_path_saved_on)
            torch.save(model.state_dict(), model_path_saved_on)
            test_result_dict = test(test_loader, model, device, args)
            
            results = eval_p(epoch = epoch, itr=itr, 
                             camera_list = args.camera_list, 
                             predict_dict=test_result_dict, logger=logger, 
                             save_path=save_path, plot=args.plot, args=args)
            epoch,itr,_auc,_fpr,_acc,_bacc,_prec,_rec,_f1, fpr, tpr, thresholds = results
            performance_results.append([epoch,itr,_auc,_fpr,_acc,_bacc,_prec,_rec,_f1])

            print("Epoch\tIter\tAUC\tFPR\tACC\tBACC\tPREC\tREC\tF1")
            for epoch,itr,_auc,_fpr,_acc,_bacc,_prec,_rec,_f1 in performance_results:
                print(f"{epoch}\t{itr}\t{_auc:.2f}\t{_fpr:.2f}\t{_acc:.2f}\t{_bacc:.2f}\t{_prec:.2f}\t{_rec:.2f}\t{_f1:.2f}")

            elapsed_time = time.time() - start_loop
            print("elapsed time", elapsed_time)