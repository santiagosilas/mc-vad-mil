import pickle
import os
import joblib
import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import sys
from utils import scorebinary, anomap

def eval_p(epoch, itr, camera_list, predict_dict, logger, save_path, args, plot=False, zip=False, manual=False):
    
    global label_dict_path
    if manual:
        save_root = './manul_test_result'
    else:
        out_name = args.camera_list
        out_name = str(out_name).replace("'","").replace("[","").replace("]","").replace(",","-").replace(" ","").replace(")","").replace("(","")
        save_root = os.path.join(os.path.join(args.dataset_path, f"outputs-{out_name}"), "result")
    
    # GTs for the both camera views (the GTs bust be the same for the two camera views)
    label_dict_path = os.path.join(os.path.join(args.dataset_path, args.camera_list[0]), "GT")

    frame_label_dict = np.load(os.path.join(label_dict_path, 'frame_label.pickle'), allow_pickle = True)
    video_label_dict = np.load(os.path.join(label_dict_path, 'video_label.pickle'), allow_pickle = True)

    all_predict_np_cams = {ds:np.zeros(0) for ds in args.camera_list }
    all_label_np = np.zeros(0)
    
    dct_outputs = dict()
    for k, v in predict_dict.items():
        dct_outputs[k] = dict()
        dct_outputs[k]["y_pred"] = dict()

        if video_label_dict.item().get(k) == [1.]:
            dct_outputs[k]["y"] = 1.
            frame_labels = frame_label_dict.item().get(k)

            for ds in args.camera_list:
                all_predict_np_cams[ds] = np.concatenate((all_predict_np_cams[ds], v[ds].repeat(16)))
                dct_outputs[k]["y_pred"][ds] = v[ds].repeat(16)
                dct_outputs[k]["y_true"] = frame_labels

            ds0 = args.camera_list[0]
            all_label_np = np.concatenate((all_label_np, frame_labels[:len(v[ds0].repeat(16))]))

        elif video_label_dict.item().get(k) == [0.]:
            dct_outputs[k]["y"] = 0.
            frame_labels = frame_label_dict.item().get(k)


            for ds in args.camera_list:
                all_predict_np_cams[ds] = np.concatenate((all_predict_np_cams[ds], v[ds].repeat(16)))
                dct_outputs[k]["y_pred"][ds] = v[ds].repeat(16)
                dct_outputs[k]["y_true"] = frame_labels

            ds0 = args.camera_list[0]
            all_label_np = np.concatenate((all_label_np, frame_labels[:len(v[ds0].repeat(16))]))

    all_label_np[np.isnan(all_label_np)] = 0.0 

    if args.late_fusion == "Max":
        #ds0 = next(iter(all_predict_np_cams.keys())) 
        ds0 = args.camera_list[0]
        all_predict_np = all_predict_np_cams[ds0]
        for ds in args.camera_list:
            lim = min( len(all_predict_np), len(all_predict_np_cams[ds]))
            all_predict_np = np.maximum(all_predict_np[:lim] , all_predict_np_cams[ds][:lim])
        
        for k in dct_outputs:
            dct_outputs[k]["max"] = dct_outputs[k]["y_pred"][ds0]
            for ds in args.camera_list:
                lim_vidk = min( len(dct_outputs[k]["max"]), len(dct_outputs[k]["y_pred"][ds]))
                dct_outputs[k]["max"] = np.maximum(dct_outputs[k]["max"][:lim_vidk] , dct_outputs[k]["y_pred"][ds][:lim_vidk])

        
    elif args.late_fusion == "Min":
        ds0 = next(iter(all_predict_np_cams.keys()))
        all_predict_np = all_predict_np_cams[ds0]
        for ds in args.camera_list:
            all_predict_np = np.minimum(all_predict_np, all_predict_np_cams[ds])

    elif args.late_fusion == "LC":
        all_predict_np = np.mean([all_predict_np_cams[ds] for ds in args.camera_list],axis = 0)
    elif "OnlyCam-" in args.late_fusion:
        all_predict_np = all_predict_np_cams[args.late_fusion.replace("OnlyCam-","")]
    else:
        raise

    
    all_y_true, all_y_pred = list(), list()
    for k in dct_outputs:
        lim_vidk = min( len(dct_outputs[k]["max"]), len(dct_outputs[k]["y_true"]))
        all_y_pred.append(np.array(dct_outputs[k]["max"][:lim_vidk]))
        all_y_true.append(np.array(dct_outputs[k]["y_true"][:lim_vidk]))
    all_y_pred = np.concatenate(all_y_pred, axis=0)
    all_y_true = np.concatenate(all_y_true, axis=0) 
    print("eval.py attempt", all_y_pred.shape, all_y_true.shape)
    
    all_y_true = all_y_true[:lim]
    all_y_pred = all_y_pred[:lim]

    
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

        
        


    return epoch,itr,_auc,_fpr,_acc,_bacc,_prec,_rec,_f1, fpr, tpr, thresholds
"""
import pickle
import os
import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import sys
from utils import scorebinary, anomap

def eval_p(epoch, itr, 
           dataset_camA,dataset_camB, dataset_camC, dataset_camD,  dataset_camE, 
           predict_dict, logger, save_path, args, plot=False, zip=False, manual=False):
    global label_dict_path
    if manual:
        save_root = './manul_test_result'
    else:
        save_root = os.path.join(os.path.join(args.dataset_path, 
                                              f"outputs-{args.dataset_name_camA}-{args.dataset_name_camB}-{args.dataset_name_camC}-{args.dataset_name_camD}-{args.dataset_name_camE}"), "result")
    
    label_dict_path = os.path.join(os.path.join(args.dataset_path, args.dataset_name_camA), "GT")

    frame_label_dict = np.load(os.path.join(label_dict_path, 'frame_label.pickle'), allow_pickle = True)
    video_label_dict = np.load(os.path.join(label_dict_path, 'video_label.pickle'), allow_pickle = True)

    all_predict_np_camA, all_predict_np_camB,all_predict_np_camC,all_predict_np_camD, all_predict_np_camE,  all_label_np = np.zeros(0),np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
    for k, v in predict_dict.items(): # v["camA"]
        if video_label_dict.item().get(k) == [1.]:
            frame_labels = frame_label_dict.item().get(k)

            all_predict_np_camA = np.concatenate((all_predict_np_camA, v["camA"].repeat(16)))
            all_predict_np_camB = np.concatenate((all_predict_np_camB, v["camB"].repeat(16)))
            all_predict_np_camC = np.concatenate((all_predict_np_camC, v["camC"].repeat(16)))
            all_predict_np_camD = np.concatenate((all_predict_np_camD, v["camD"].repeat(16)))
            all_predict_np_camE = np.concatenate((all_predict_np_camE, v["camE"].repeat(16)))
            all_label_np = np.concatenate((all_label_np, frame_labels[:len(v["camA"].repeat(16))]))
        elif video_label_dict.item().get(k) == [0.]:
            frame_labels = frame_label_dict.item().get(k)
            all_predict_np_camA = np.concatenate((all_predict_np_camA, v["camA"].repeat(16)))
            all_predict_np_camB = np.concatenate((all_predict_np_camB, v["camB"].repeat(16)))
            all_predict_np_camC = np.concatenate((all_predict_np_camC, v["camC"].repeat(16)))
            all_predict_np_camD = np.concatenate((all_predict_np_camD, v["camD"].repeat(16)))
            all_predict_np_camE = np.concatenate((all_predict_np_camE, v["camE"].repeat(16)))
            all_label_np = np.concatenate((all_label_np, frame_labels[:len(v["camA"].repeat(16))]))

    all_label_np[np.isnan(all_label_np)] = 0.0 

    if args.late_fusion == "Max":
        all_predict_np = np.maximum(all_predict_np_camA, 
                                    np.maximum(all_predict_np_camB, 
                                               np.maximum(all_predict_np_camC, 
                                                          np.maximum(all_predict_np_camD,all_predict_np_camE)
                        )))
    elif args.late_fusion == "Min":
        raise
        all_predict_np = np.minimum(all_predict_np_camA, np.minimum(all_predict_np_camB, np.minimum(all_predict_np_camC, all_predict_np_camD)))
    elif args.late_fusion == "LC":
        raise
        all_predict_np = (all_predict_np_camA + all_predict_np_camB + all_predict_np_camC + all_predict_np_camD)/4
    elif args.late_fusion == "OnlyCamA":
        raise
        all_predict_np = all_predict_np_camA
    elif args.late_fusion == "OnlyCamB":
        raise
        all_predict_np = all_predict_np_camB
    elif args.late_fusion == "OnlyCamC":
        raise
        all_predict_np = all_predict_np_camC
    elif args.late_fusion == "OnlyCamD":
        raise
        all_predict_np = all_predict_np_camD
    else:
        raise
    
    all_y_true = all_label_np
    all_y_pred = all_predict_np
    
    fpr, tpr, thresholds = roc_curve(all_y_true, all_y_pred)
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


    return epoch,itr,_auc,_fpr,_acc,_bacc,_prec,_rec,_f1, fpr, tpr, thresholds
"""




