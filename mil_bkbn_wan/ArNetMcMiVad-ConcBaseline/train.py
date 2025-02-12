import torch
import numpy as np
from test import test
from eval import eval_p
import os
import pickle
import time
from losses import KMXMILL_individual, normal_smooth

def train(epochs, train_loader, all_test_loader, args, model, optimizer, logger, device, save_path, 
          ckpt_path,logs_path, results_path):
    
    [train2test_loader, test_loader] = all_test_loader
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
    performance_results = list()

    with open(file=os.path.join(ckpt_path, 'params.txt'), mode='a+') as f:
        f.write(str(args))

    for epoch in range(1, epochs+1):
        
        for i, data in enumerate(train_loader):
            itr += 1
            [anomaly_features, normaly_features], [anomaly_label, normaly_label], stastics_data = data
            features = torch.cat((anomaly_features.squeeze(0), normaly_features.squeeze(0)), dim=0)
            videolabels = torch.cat((anomaly_label.squeeze(0), normaly_label.squeeze(0)), dim=0)
            seq_len = torch.sum(torch.max(features.abs(), dim=2)[0] > 0, dim=1).numpy()
            features = features[:, :np.max(seq_len), :]

            features = features.float().to(device)
            videolabels = videolabels.float().to(device)
            final_features, element_logits = model(features)
            weights = args.Lambda.split('_')
            m_loss = KMXMILL_individual(element_logits=element_logits,
                                        seq_len=seq_len,
                                        labels=videolabels,
                                        device=device,
                                        loss_type='CE',
                                        args=args)
            n_loss = normal_smooth(element_logits=element_logits,
                                   labels=videolabels,
                                   device=device)

            total_loss = float(weights[0]) * m_loss + float(weights[1]) * n_loss
            logger.log_value('m_loss', m_loss, itr)
            logger.log_value('n_loss', n_loss, itr)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        if epoch % args.save_every == 0: 
            
            print('Epoch:{}, Iteration:{}, Loss: {}'.format(epoch,itr,total_loss.data.cpu().detach().numpy()))
            model_path_saved_on = os.path.join(ckpt_path, 'epoch_{}_iter_{}'.format(epoch,itr) + '.pkl')
            print("model path", model_path_saved_on)
            torch.save(model.state_dict(), model_path_saved_on)
            test_result_dict = test(test_loader, model, device, args)
            results = eval_p(epoch=epoch, itr=itr, dataset=args.dataset_name, predict_dict=test_result_dict, logger=logger, 
                    save_path=save_path, plot=args.plot, args=args)
            
            epoch,itr,_auc,_fpr,_acc,_bacc,_prec,_rec,_f1, fpr, tpr, thresholds = results
            performance_results.append([epoch,itr,_auc,_fpr,_acc,_bacc,_prec,_rec,_f1])

            print("Epoch\tIter\tAUC\tFPR\tACC\tBACC\tPREC\tREC\tF1")
            for epoch,itr,_auc,_fpr,_acc,_bacc,_prec,_rec,_f1 in performance_results:
                print(f"{epoch}\t{itr}\t{_auc:.2f}\t{_fpr:.2f}\t{_acc:.2f}\t{_bacc:.2f}\t{_prec:.2f}\t{_rec:.2f}\t{_f1:.2f}")


        elapsed_time = time.time() - start_loop
        print(f"epoch: {epoch} iter:{itr} loss: {total_loss.data.cpu().detach().numpy()} elapsed time: {elapsed_time} seconds ({(elapsed_time/60):.2f} minutes)" )

