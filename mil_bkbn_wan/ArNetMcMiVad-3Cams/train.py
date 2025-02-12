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
    if args.pretrained_ckpt:
        checkpoint = torch.load(args.pretrained_ckpt)
        model.load_state_dict(checkpoint)
        print('model load weights from {}'.format(args.pretrained_ckpt))
    else:
        print('model is trained from scratch')
    start_loop = time.time()

    with open(file=os.path.join(ckpt_path, 'params.txt'), mode='a+') as f:
        f.write(str(args))

    performance_results = list()
    for epoch in range(1, epochs+1):
        #print("epoch", epoch)
        for i, data in enumerate(train_loader):
            itr += 1
            [anomaly_features_camA, normaly_features_camA],[anomaly_features_camB, normaly_features_camB],[anomaly_features_camC, normaly_features_camC], [anomaly_label, normaly_label], stastics_data = data
            
            if args.loss_combination == "PairwiseMeanOfClipScoresInBags":
                
                # camA
                features_camA = torch.cat((anomaly_features_camA.squeeze(0), normaly_features_camA.squeeze(0)), dim=0)
                videolabels_camA = torch.cat((anomaly_label.squeeze(0), normaly_label.squeeze(0)), dim=0)
                seq_len_camA = torch.sum(torch.max(features_camA.abs(), dim=2)[0] > 0, dim=1).numpy()
                features_camA = features_camA[:, :np.max(seq_len_camA), :]
                features_camA = features_camA.float().to(device)
                videolabels_camA = videolabels_camA.float().to(device)
                if args.model_name == 'model_single':
                    final_features_camA, element_logits_camA = model(features_camA)
                else:
                    # *.forward_cam1
                    final_features_camA, element_logits_camA = model.forward_camA(features_camA)

                # camB
                features_camB = torch.cat((anomaly_features_camB.squeeze(0), normaly_features_camB.squeeze(0)), dim=0)
                videolabels_camB = torch.cat((anomaly_label.squeeze(0), normaly_label.squeeze(0)), dim=0)
                seq_len_camB = torch.sum(torch.max(features_camB.abs(), dim=2)[0] > 0, dim=1).numpy()
                features_camB = features_camB[:, :np.max(seq_len_camB), :]
                features_camB = features_camB.float().to(device)
                videolabels_camB = videolabels_camB.float().to(device)
                if args.model_name == 'model_single':
                    final_features_camB, element_logits_camB = model(features_camB)
                else:
                    # *.forward_cam2
                    final_features_camB, element_logits_camB = model.forward_camB(features_camB)

                # camC
                features_camC = torch.cat((anomaly_features_camC.squeeze(0), normaly_features_camC.squeeze(0)), dim=0)
                videolabels_camC = torch.cat((anomaly_label.squeeze(0), normaly_label.squeeze(0)), dim=0)
                seq_len_camC = torch.sum(torch.max(features_camC.abs(), dim=2)[0] > 0, dim=1).numpy()
                features_camC = features_camC[:, :np.max(seq_len_camC), :]
                features_camC = features_camC.float().to(device)
                videolabels_camC = videolabels_camC.float().to(device)
                if args.model_name == 'model_single':
                    final_features_camC, element_logits_camC = model(features_camC)
                else:
                    # *.forward_cam2
                    final_features_camC, element_logits_camC = model.forward_camB(features_camC)



                weights = args.Lambda.split('_')

                element_logits = (element_logits_camA+element_logits_camB+element_logits_camC)/3
                
                m_loss = KMXMILL_individual(element_logits=element_logits, seq_len=seq_len_camA, 
                    labels=videolabels_camA, device=device, loss_type='CE', args=args)
                n_loss = normal_smooth(element_logits=element_logits, labels=videolabels_camA, device=device)
                total_loss = float(weights[0]) * m_loss + float(weights[1]) * n_loss
                
                logger.log_value('m_loss', m_loss, itr)
                logger.log_value('n_loss', n_loss, itr)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                continue

            # camA
            features_camA = torch.cat((anomaly_features_camA.squeeze(0), normaly_features_camA.squeeze(0)), dim=0)
            videolabels_camA = torch.cat((anomaly_label.squeeze(0), normaly_label.squeeze(0)), dim=0)
            seq_len_camA = torch.sum(torch.max(features_camA.abs(), dim=2)[0] > 0, dim=1).numpy()
            features_camA = features_camA[:, :np.max(seq_len_camA), :]
            features_camA = features_camA.float().to(device)
            videolabels_camA = videolabels_camA.float().to(device)
            if args.model_name == 'model_single':
                final_features_camA, element_logits_camA = model(features_camA)
            else:
                # *.forward_cam1
                final_features_camA, element_logits_camA = model.forward_camA(features_camA)

            weights = args.Lambda.split('_')
            m_loss_camA = KMXMILL_individual(element_logits=element_logits_camA, seq_len=seq_len_camA, labels=videolabels_camA, device=device, loss_type='CE', args=args)
            n_loss_camA = normal_smooth(element_logits=element_logits_camA, labels=videolabels_camA, device=device)
            total_loss_camA = float(weights[0]) * m_loss_camA + float(weights[1]) * n_loss_camA
            logger.log_value('m_loss', m_loss_camA, itr)
            logger.log_value('n_loss', n_loss_camA, itr)

            # camB
            features_camB = torch.cat((anomaly_features_camB.squeeze(0), normaly_features_camB.squeeze(0)), dim=0)
            videolabels_camB = torch.cat((anomaly_label.squeeze(0), normaly_label.squeeze(0)), dim=0)
            seq_len_camB = torch.sum(torch.max(features_camB.abs(), dim=2)[0] > 0, dim=1).numpy()
            features_camB = features_camB[:, :np.max(seq_len_camB), :]
            features_camB = features_camB.float().to(device)
            videolabels_camB = videolabels_camB.float().to(device)
            if args.model_name == 'model_single':
                final_features_camB, element_logits_camB = model(features_camB)
            else:
                # *.forward_cam2
                final_features_camB, element_logits_camB = model.forward_camB(features_camB)

            weights = args.Lambda.split('_')
            m_loss_camB = KMXMILL_individual(element_logits=element_logits_camB, seq_len=seq_len_camB, labels=videolabels_camB, device=device, loss_type='CE', args=args)
            n_loss_camB = normal_smooth(element_logits=element_logits_camB, labels=videolabels_camB, device=device)
            total_loss_camB = float(weights[0]) * m_loss_camB + float(weights[1]) * n_loss_camB
            logger.log_value('m_loss', m_loss_camB, itr)
            logger.log_value('n_loss', n_loss_camB, itr)

            # camC
            features_camC = torch.cat((anomaly_features_camC.squeeze(0), normaly_features_camC.squeeze(0)), dim=0)
            videolabels_camC = torch.cat((anomaly_label.squeeze(0), normaly_label.squeeze(0)), dim=0)
            seq_len_camC = torch.sum(torch.max(features_camC.abs(), dim=2)[0] > 0, dim=1).numpy()
            features_camC = features_camC[:, :np.max(seq_len_camC), :]
            features_camC = features_camC.float().to(device)
            videolabels_camC = videolabels_camC.float().to(device)
            if args.model_name == 'model_single':
                final_features_camC, element_logits_camC = model(features_camC)
            else:
                # *.forward_cam2
                final_features_camC, element_logits_camC = model.forward_camC(features_camC)

            weights = args.Lambda.split('_')
            m_loss_camC = KMXMILL_individual(element_logits=element_logits_camC, seq_len=seq_len_camC, labels=videolabels_camC, device=device, loss_type='CE', args=args)
            n_loss_camC = normal_smooth(element_logits=element_logits_camC, labels=videolabels_camC, device=device)
            total_loss_camC = float(weights[0]) * m_loss_camC + float(weights[1]) * n_loss_camC
            logger.log_value('m_loss', m_loss_camC, itr)
            logger.log_value('n_loss', n_loss_camC, itr)


            # max
            if args.loss_combination == "Max":
                total_loss = torch.max(total_loss_camA, torch.max(total_loss_camB, total_loss_camC))
            elif args.loss_combination == "Min":
                total_loss = torch.min(total_loss_camA, torch.min(total_loss_camB, total_loss_camC))
            elif args.loss_combination == "LC":
                total_loss = (total_loss_camA + total_loss_camB + total_loss_camC)/3
            elif args.loss_combination == "OnlyLossCamA":
                total_loss = total_loss_camA
            elif args.loss_combination == "OnlyLossCamB":
                total_loss = total_loss_camB
            elif args.loss_combination == "OnlyLossCamC":
                total_loss = total_loss_camC
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
            
            results = eval_p(epoch = epoch, itr=itr, dataset_camA = args.dataset_name_camA, dataset_camB = args.dataset_name_camB, dataset_camC = args.dataset_name_camC, 
                predict_dict=test_result_dict, logger=logger, 
                save_path=save_path, plot=args.plot, args=args)
            epoch,itr,_auc,_fpr,_acc,_bacc,_prec,_rec,_f1, fpr, tpr, thresholds = results
            performance_results.append([epoch,itr,_auc,_fpr,_acc,_bacc,_prec,_rec,_f1])

            print("Epoch\tIter\tAUC\tFPR\tACC\tBACC\tPREC\tREC\tF1")
            for epoch,itr,_auc,_fpr,_acc,_bacc,_prec,_rec,_f1 in performance_results:
                print(f"{epoch}\t{itr}\t{_auc:.2f}\t{_fpr:.2f}\t{_acc:.2f}\t{_bacc:.2f}\t{_prec:.2f}\t{_rec:.2f}\t{_f1:.2f}")

            elapsed_time = time.time() - start_loop
            print("elapsed time", elapsed_time)

