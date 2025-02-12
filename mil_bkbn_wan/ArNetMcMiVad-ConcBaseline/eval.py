import pickle
import os
import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import sys
from utils import scorebinary, anomap
from collections import Counter

#"""
# HQFS Dataset Relabeled
handcraft_labeling = {
    "Fall1":(1410,3930),
    "Fall10":(720,5160),
    "Fall15":(326,3226),
    "Fall22":(570,3960),
    "Fall25":(705,3030),
    "Fall28":(2040,4131),
    "Fall36":(1004,4656),
    "Fall42":(2370,2976),
    "Fall45":(600,1049),
    "Fall50":(483,3999),
    "Fall51":(878,4370),
    "Fall52":(1309,4986),
    "Fall54":(1344,5679),
}
#"""

def eval_p(epoch, itr, dataset, predict_dict, logger, save_path, args, plot=False, zip=False, manual=False):
    global label_dict_path
    if manual:
        save_root = './manul_test_result'
    else:
        save_root = os.path.join(os.path.join(args.dataset_path, args.dataset_name), "result")
    
    if dataset == 'shanghaitech':
        label_dict_path = '{}/shanghaitech/GT'.format(args.dataset_path)
    else:
        label_dict_path = os.path.join(os.path.join(args.dataset_path, args.dataset_name), "GT")


    #with open(file=os.path.join(label_dict_path, 'frame_label.pickle'), mode='rb') as f:
    #    frame_label_dict = pickle.load(f)
    #with open(file=os.path.join(label_dict_path, 'video_label.pickle'), mode='rb') as f:
    #    video_label_dict = pickle.load(f)
    frame_label_dict = np.load(os.path.join(label_dict_path, 'frame_label.pickle'), allow_pickle = True)
    video_label_dict = np.load(os.path.join(label_dict_path, 'video_label.pickle'), allow_pickle = True)

    all_predict_np = np.zeros(0)
    all_label_np = np.zeros(0)
    normal_predict_np = np.zeros(0)
    normal_label_np = np.zeros(0)
    abnormal_predict_np = np.zeros(0)
    abnormal_label_np = np.zeros(0)
    for k, v in predict_dict.items():
        
        #print("test video", k, "cat", video_label_dict.item().get(k))

        try:
            video_label_dict_key_k = video_label_dict.item().get(k)
        except:
            video_label_dict_key_k = video_label_dict.get(k)

        if video_label_dict_key_k == [1.]:
            try:
                frame_labels = frame_label_dict.item().get(k)
            except:
                frame_labels = frame_label_dict.get(k)

            #"""
            if k in handcraft_labeling.keys():
                dim_orig = frame_labels.shape
                frame_labels = frame_labels.ravel()
                interval = handcraft_labeling[k]
                for frame_pos in range(*interval):
                    frame_labels[frame_pos] = 1.0
                frame_labels = frame_labels.reshape(dim_orig)
            #"""


            #print("debug", k);import sys;print(frame_labels.shape);sys.exit(1);
            all_predict_np = np.concatenate((all_predict_np, v.repeat(16)))
            all_label_np = np.concatenate((all_label_np, frame_labels[:len(v.repeat(16))]))
            abnormal_predict_np = np.concatenate((abnormal_predict_np, v.repeat(16)))
            abnormal_label_np = np.concatenate((abnormal_label_np, frame_labels[:len(v.repeat(16))]))
        elif video_label_dict_key_k == [0.]:
            try:
                frame_labels = frame_label_dict.item().get(k)
            except:
                frame_labels = frame_label_dict.get(k)
            all_predict_np = np.concatenate((all_predict_np, v.repeat(16)))
            all_label_np = np.concatenate((all_label_np, frame_labels[:len(v.repeat(16))]))
            normal_predict_np = np.concatenate((normal_predict_np, v.repeat(16)))
            normal_label_np = np.concatenate((normal_label_np, frame_labels[:len(v.repeat(16))]))

    all_label_np[np.isnan(all_label_np)] = 0 # Up Fall contains NaN in y_true, originally.

    all_label_np = all_label_np[:min(len(all_label_np),len(all_predict_np))]
    all_predict_np = all_predict_np[:min(len(all_label_np),len(all_predict_np))]

    all_auc_score = roc_auc_score(y_true=all_label_np, y_score=all_predict_np)
    binary_all_predict_np = scorebinary(all_predict_np, threshold=0.5)
    tn, fp, fn, tp = confusion_matrix(y_true=all_label_np, y_pred=binary_all_predict_np).ravel()
    all_ano_false_alarm = fp / (fp + tn)
    binary_normal_predict_np = scorebinary(normal_predict_np, threshold=0.5)

    # tn, fp, fn, tp = confusion_matrix(y_true=normal_label_np, y_pred=binary_normal_predict_np).ravel()
    #fp_n = binary_normal_predict_np.sum()
    #normal_count = normal_label_np.shape[0]
    #normal_ano_false_alarm = fp_n / normal_count

    #abnormal_auc_score = roc_auc_score(y_true=abnormal_label_np, y_score=abnormal_predict_np)
    #binary_abnormal_predict_np = scorebinary(abnormal_predict_np, threshold=0.5)
    #tn, fp, fn, tp = confusion_matrix(y_true=abnormal_label_np, y_pred=binary_abnormal_predict_np).ravel()
    #abnormal_ano_false_alarm = fp / (fp + tn)

    #print('Iteration: {} AUC_score_all_video is {}'.format(itr, all_auc_score))
    #print('Iteration: {} AUC_score_abnormal_video is {}'.format(itr, abnormal_auc_score))
    #print('Iteration: {} ano_false_alarm_all_video is {}'.format(itr, all_ano_false_alarm))
    #print('Iteration: {} ano_false_alarm_normal_video is {}'.format(itr, normal_ano_false_alarm))
    #print('Iteration: {} ano_false_alarm_abnormal_video is {}'.format(itr, abnormal_ano_false_alarm))
    #if plot:
    #    anomap(predict_dict, frame_label_dict, save_path, epoch, itr, save_root, zip)
    #if logger:
    #    logger.log_value('Test_AUC_all_video', all_auc_score, itr)
    #    logger.log_value('Test_AUC_abnormal_video', abnormal_auc_score, itr)
    #    logger.log_value('Test_false_alarm_all_video', all_ano_false_alarm, itr)
    #    logger.log_value('Test_false_alarm_normal_video', normal_ano_false_alarm, itr)
    #    logger.log_value('Test_false_alarm_abnormal_video', abnormal_ano_false_alarm, itr)
    if save_path!=None and os.path.exists(os.path.join(save_root,save_path)) == 0:
        os.makedirs(os.path.join(save_root,save_path))
    #if save_path!=None:
    #    with open(file=os.path.join(save_root, save_path, 'result.txt'), mode='a+') as f:
    #        f.write('itration_{}_AUC_Score_all_video is {}\n'.format(itr, all_auc_score))
    #        f.write('itration_{}_AUC_Score_abnormal_video is {}\n'.format(itr, abnormal_auc_score))
    #        f.write('itration_{}_ano_false_alarm_all_video is {}\n'.format(itr, all_ano_false_alarm))
    #        f.write('itration_{}_ano_false_alarm_normal_video is {}\n'.format(itr, normal_ano_false_alarm))
    #        f.write('itration_{}_ano_false_alarm_abnormal_video is {}\n'.format(itr, abnormal_ano_false_alarm))

    # re-computes performance metrics for conference !
    all_y_true = all_label_np
    all_y_pred = all_predict_np
    
    fpr, tpr, thresholds = roc_curve(all_y_true, all_y_pred)
    _auc = round(sklearn.metrics.auc(fpr, tpr) * 100, 2)

    cm = confusion_matrix(all_y_true, np.round(all_y_pred))
    tn, fp, fn, tp = confusion_matrix(all_y_true, np.round(all_y_pred)).ravel()
    fpr = fp/(fp + tn)
    fpr = round(fpr * 100, 2)
    _fpr = fpr
    
    _acc = (tp+tn)/(tp+tn+fp+fn) * 100
    _bacc = sklearn.metrics.balanced_accuracy_score(all_y_true, np.round(all_y_pred))  * 100
    
    _prec = tp/(tp+fp) * 100
    _rec = tp/(tp+fn) * 100
    _f1 = 2 * (_prec * _rec) / (_prec + _rec)
    

    print("Epoch\tIter\tAUC\tFPR\tACC\tBACC\tPREC\tREC\tF1")
    print(f"{epoch}\t{itr}\t{_auc:.2f}\t{_fpr:.2f}\t{_acc:.2f}\t{_bacc:.2f}\t{_prec:.2f}\t{_rec:.2f}\t{_f1:.2f}")

    if save_path != None:
        with open(file=os.path.join(save_root, save_path, 'result_metrics.txt'), mode='a+') as f:
            f.write("Epoch\tIter\tAUC\tFPR\tACC\tBACC\tPREC\tREC\tF1\n")
            f.write(f"{epoch}\t{itr}\t{_auc:.2f}\t{_fpr:.2f}\t{_acc:.2f}\t{_bacc:.2f}\t{_prec:.2f}\t{_rec:.2f}\t{_f1:.2f}\n")

    return epoch,itr,_auc,_fpr,_acc,_bacc,_prec,_rec,_f1, fpr, tpr, thresholds


