import pickle
import os
import joblib
import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import sys
from utils import scorebinary, anomap



def eval_p(epoch, itr, dataset_camA,dataset_camB, predict_dict, logger, save_path, args, plot=False, zip=False, manual=False):
    global label_dict_path
    if manual:
        save_root = './manul_test_result'
    else:
        save_root = os.path.join(os.path.join(args.dataset_path, f"outputs-{args.dataset_name_camA}-{args.dataset_name_camB}"), "result")
    
    # GTs for the both camera views (the GTs bust be the same for the two camera views)
    label_dict_path = os.path.join(os.path.join(args.dataset_path, args.dataset_name_camA), "GT")

    frame_label_dict = np.load(os.path.join(label_dict_path, 'frame_label.pickle'), allow_pickle = True)
    video_label_dict = np.load(os.path.join(label_dict_path, 'video_label.pickle'), allow_pickle = True)

    all_predict_np_camA, all_predict_np_camB,  all_label_np = np.zeros(0), np.zeros(0), np.zeros(0)
    #print("eval ..", predict_dict.keys())
    dct_outputs = dict()
    for k, v in predict_dict.items(): # v["camA"]
        dct_outputs[k] = dict()
        dct_outputs[k]["y_pred"] = dict()        
        #print(f"test video {k} cat {video_label_dict.item().get(k)}")

        if video_label_dict.item().get(k) == [1.]:
            dct_outputs[k]["y"] = 1.
            frame_labels = frame_label_dict.item().get(k)

        if video_label_dict.item().get(k) == [1.]:
            dct_outputs[k]["y"] = 1.
            frame_labels = frame_label_dict.item().get(k)



            all_predict_np_camA = np.concatenate((all_predict_np_camA, v["camA"].repeat(16)))
            all_predict_np_camB = np.concatenate((all_predict_np_camB, v["camB"].repeat(16)))
            all_label_np = np.concatenate((all_label_np, frame_labels[:len(v["camA"].repeat(16))]))

            dct_outputs[k]["y_pred"]["camA"] = v["camA"].repeat(16)
            dct_outputs[k]["y_pred"]["camB"] = v["camB"].repeat(16)
            dct_outputs[k]["y_true"] = frame_labels

        elif video_label_dict.item().get(k) == [0.]:
            dct_outputs[k]["y"] = 0.
            frame_labels = frame_label_dict.item().get(k)
            all_predict_np_camA = np.concatenate((all_predict_np_camA, v["camA"].repeat(16)))
            all_predict_np_camB = np.concatenate((all_predict_np_camB, v["camB"].repeat(16)))
            all_label_np = np.concatenate((all_label_np, frame_labels[:len(v["camA"].repeat(16))]))

            dct_outputs[k]["y_pred"]["camA"] = v["camA"].repeat(16)
            dct_outputs[k]["y_pred"]["camB"] = v["camB"].repeat(16)
            dct_outputs[k]["y_true"] = frame_labels

    all_label_np[np.isnan(all_label_np)] = 0.0 # Up Fall contains NaN in y_true, originally.

    if args.late_fusion == "Max":
        all_predict_np = np.maximum(all_predict_np_camA, all_predict_np_camB)

        for k in dct_outputs:
            lim_vidk = min( len(dct_outputs[k]["y_pred"]["camA"]), len(dct_outputs[k]["y_pred"]["camB"]))
            dct_outputs[k]["max"] = np.maximum(dct_outputs[k]["y_pred"]["camA"][:lim_vidk], dct_outputs[k]["y_pred"]["camB"][:lim_vidk])

    elif args.late_fusion == "Min":
        all_predict_np = np.minimum(all_predict_np_camA, all_predict_np_camB)
    elif args.late_fusion == "LC":
        all_predict_np = 0.5 * all_predict_np_camA + 0.5 * all_predict_np_camB
    elif args.late_fusion == "OnlyCamA":
        all_predict_np = all_predict_np_camA
    elif args.late_fusion == "OnlyCamB":
        all_predict_np = all_predict_np_camB
    else:
        raise

    # re-computes performance metrics for conference !
    all_y_true = all_label_np
    all_y_pred = all_predict_np
    
    fpr, tpr, thresholds = roc_curve(all_y_true, all_y_pred)
    fpr_roccurve, tpr_roccurve = fpr, tpr
    _auc = round(sklearn.metrics.auc(fpr, tpr) * 100, 2)

    cm = confusion_matrix(all_y_true, np.round(all_y_pred))
    tn, fp, fn, tp = confusion_matrix(all_y_true, np.round(all_y_pred), labels=[0,1]).ravel()
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

        joblib.dump([fpr_roccurve, tpr_roccurve, thresholds,_auc], os.path.join(save_root, save_path, 'fpr-tpr-thresholds-auc.joblib'))
        joblib.dump([all_y_true, all_y_pred], os.path.join(save_root, save_path, 'ytrue-ypred.joblib'))
        joblib.dump(dct_outputs, os.path.join(save_root, save_path, 'outputs-per-vid.joblib'))
        print(os.path.join(save_root, save_path))

    return epoch,itr,_auc,_fpr,_acc,_bacc,_prec,_rec,_f1, fpr, tpr, thresholds




