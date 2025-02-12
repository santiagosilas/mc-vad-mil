"""
This source code is based on the github repository 
of of Eitan Kosman (https://orcid.org/0000-0002-5538-0616) 
with title "Pytorch implementation of Real-World Anomaly Detection in Surveillance Videos"
available at https://github.com/ekosman/AnomalyDetectionCVPR2018-Pytorch. 

Our code extends the original work by implementing multi-camera algorithms for VAD in overlapping scenarios.
"""

import joblib
import os, sys
import random
import numpy as np
import pandas as pd
import munch
import torch

from collections import Counter
from abc import ABC, abstractmethod
from torch.optim import Optimizer
from torch import Tensor, nn
from torch.utils.data import DataLoader
from os import path, mkdir
from typing import Dict, List, Tuple, Union
from torch.backends import cudnn
from torch.utils.data import DataLoader
from os import path, makedirs
from typing import Union
from torch import device
from torch.backends import cudnn
from typing import List, Tuple
from torch import Tensor
from torch.utils import data
from torchvision.datasets.video_utils import VideoClips
from typing import List, Tuple
from torch.utils import data
from os import path
from typing import Tuple, Union, Callable

import sklearn
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from collections import Counter
from IPython.display import display, HTML
from torch.backends import cudnn
from munch import munchify

def train_mcmil2(list_data_paths, iterations = 100, batch_size = 60, feature_dim = 1024):
  
  class RegressionNetwork(nn.Module):
    def __init__(self, input_dim, count_views) -> None:
        super().__init__()

        self.fc0 = nn.ModuleDict()
        for i in range(count_views):
            self.fc0[f"{i+1}"] = nn.Linear(input_dim, 512).to(device)

        self.fc0_cam2 = nn.Linear(input_dim, 512)
        
        self.fc1 = nn.Linear(512, 512)

        self.fc2 = nn.Linear(512, 32)

        self.fc3 = nn.Linear(32, 1)
        self.sig = nn.Sigmoid()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.6)
        
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, x: Tensor, view_index: int) -> Tensor:
        x = self.dropout(self.relu(self.fc0[f"{view_index}"](x)))
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        x = self.sig(self.fc3(x))
        return x


  print("start multi-camera training ..")

  def data_to_device(data: Tensor, device: str) -> Tensor:
    if isinstance(data, list):
        data = [d.to(device) for d in data]
    elif isinstance(data, tuple):
        data = tuple([d.to(device) for d in data])
    else:
        data = data.to(device)
    return data

  def read_features(file_path, feature_dim: int, cache: Dict = None) -> np.ndarray:
          if cache is not None and file_path in cache:
              return cache[file_path]

          if not os.path.exists(file_path):
              raise FileNotFoundError(f"Feature doesn't exist: `{file_path}`")

          features = None
          with open(file_path, "r") as fp:
              data = fp.read().splitlines(keepends=False)
              features = np.zeros((len(data), feature_dim))
              for i, line in enumerate(data):
                  features[i, :] = [float(x) for x in line.split(" ")]

          features = torch.from_numpy(features).float()
          if cache is not None:
              cache[file_path] = features
          return features

  class FeaturesLoader:
          def __init__(
              self,
              features_path: str,
              feature_dim: int,
              annotation_path: str,
              bucket_size: int = batch_size//2,
              iterations: int = 10000,
          ) -> None:

              super().__init__()
              self._features_path = features_path
              self._feature_dim = feature_dim
              self._bucket_size = bucket_size

              (
                  self.features_list_normal,
                  self.features_list_anomaly,
              ) = FeaturesLoader._get_features_list(
                  features_path=self._features_path, annotation_path=annotation_path
              )

              self._iterations = iterations
              self._features_cache = {}
              self._i = 0

          def __len__(self) -> int:
              return self._iterations

          def __getitem__(self, index: int):
              if self._i == len(self):
                  self._i = 0
                  raise StopIteration

              succ = False
              while not succ:
                  try:
                      feature, label = self.get_features()
                      succ = True
                  except Exception as e:
                      index = np.random.choice(range(0, self.__len__()))

              self._i += 1
              return feature, label


          def get_positions(self):

              normal_positions = np.random.choice(
                  len(self.features_list_normal), size=self._bucket_size
              )
              abnormal_positions = np.random.choice(
                  len(self.features_list_anomaly), size=self._bucket_size
              )
              return normal_positions, abnormal_positions

          def get_features_by_positions(self, normal_positions, abnormal_positions):

              normal_paths = [self.features_list_normal[pos] for pos in normal_positions]
              abnormal_paths = [self.features_list_anomaly[pos] for pos in abnormal_positions]

              all_paths = np.concatenate([normal_paths, abnormal_paths])

              features_to_stack = [
                      read_features(
                          f"{feature_subpath}.txt", self._feature_dim, self._features_cache
                      ).shape[0]
                      for feature_subpath in all_paths
              ]

              features = [
                      read_features(
                          f"{feature_subpath}.txt", self._feature_dim, self._features_cache
                      )
                      for feature_subpath in all_paths
              ]
              return (
                  features,
                  torch.cat([torch.zeros(self._bucket_size), torch.ones(self._bucket_size)]),
              )



          def get_features(self):
              normal_paths = np.random.choice(
                  self.features_list_normal, size=self._bucket_size
              )
              abnormal_paths = np.random.choice(
                  self.features_list_anomaly, size=self._bucket_size
              )
              all_paths = np.concatenate([normal_paths, abnormal_paths])

              features_to_stack = [
                      read_features(
                          f"{feature_subpath}.txt", self._feature_dim, self._features_cache
                      ).shape[0]
                      for feature_subpath in all_paths
              ]

              features = [
                      read_features(
                          f"{feature_subpath}.txt", self._feature_dim, self._features_cache
                      )
                      for feature_subpath in all_paths
              ]
              return (
                  features,
                  torch.cat([torch.zeros(self._bucket_size), torch.ones(self._bucket_size)]),
              )

          @staticmethod
          def _get_features_list(
              features_path: str, annotation_path: str
          ) -> Tuple[List[str], List[str]]:

              assert os.path.exists(features_path)
              features_list_normal = []
              features_list_anomaly = []
              with open(annotation_path, "r") as f:
                  lines = f.read().splitlines(keepends=False)

                  for line in lines:
                      items = line.split()
                      file = items[0].split(".")[0]
                      file = file.replace("/", os.sep)
                      feature_path = os.path.join(features_path, file)
                      if "Normal" in feature_path:
                          features_list_normal.append(feature_path)
                      else:
                          features_list_anomaly.append(feature_path)
              return features_list_normal, features_list_anomaly

  class RegularizedLoss(torch.nn.Module):
          def __init__(self, model: nn.Module, lambdas: float = 0.001) -> None:
              super().__init__()
              self.lambdas = lambdas
              self.model = model

          def forward(self, y_pred: Tensor, y_true: Tensor):

              lambdas = 8e-5
              normal_vids_indices = torch.where(y_true == 0)[0]
              anomal_vids_indices = torch.where(y_true == 1)[0]

              normal_segments_scores = [y_pred[i] for i in range(len(y_pred)) if i in normal_vids_indices]
              anomal_segments_scores = [y_pred[i] for i in range(len(y_pred)) if i in anomal_vids_indices]

              normal_segments_scores_maxes = [torch.max(normal_segments_scores[i]).reshape([1]) for i in range(len(normal_segments_scores))]
              anomal_segments_scores_maxes = [torch.max(anomal_segments_scores[i]).reshape([1]) for i in range(len(anomal_segments_scores))]

              hinge_loss = [1 - anomal_segments_scores_maxes[i] + normal_segments_scores_maxes[i] for i in range(batch_size//2)]
              hinge_loss = torch.cat(hinge_loss)

              hinge_loss = torch.max(hinge_loss, torch.zeros_like(hinge_loss))

              smoothed_scores = [
                  [ anomal_segments_scores[j][i]-anomal_segments_scores[j][i-1] for i in list(range(1, len(anomal_segments_scores[j])))]
                  for j in range(len(anomal_segments_scores))
              ]

              smoothed_scores_sum_squared = [
                  torch.stack([ torch.pow(smoothed_scores[j][i], 2) for i in list(range(1, len(smoothed_scores[j])))  ]).sum() for j in range(batch_size//2)
              ]

              sparsity_loss =  [
                  anomal_segments_scores[j].sum()
                  for j in range(len(anomal_segments_scores))
              ]

              final_loss = torch.stack([
                  hinge_loss[i] +
                  lambdas * smoothed_scores_sum_squared[i] +
                  lambdas * sparsity_loss[i] for i in range(batch_size//2)]).mean()


              fc1_params = torch.cat(tuple([x.view(-1) for x in self.model.fc1.parameters()]))
              fc2_params = torch.cat(tuple([x.view(-1) for x in self.model.fc2.parameters()]))
              fc3_params = torch.cat(tuple([x.view(-1) for x in self.model.fc3.parameters()]))

              l1_regularization = self.lambdas * torch.norm(fc1_params, p=2)
              l2_regularization = self.lambdas * torch.norm(fc2_params, p=2)
              l3_regularization = self.lambdas * torch.norm(fc3_params, p=2)

              return (final_loss+ l1_regularization + l2_regularization + l3_regularization)




  evaluation_mode = False
  args = munch.munchify(dict(
          list_data_paths = list_data_paths,
          feature_dim = feature_dim,
          iterations_per_epoch = iterations,
          batch_size = batch_size,
          lr_base = 0.01,
  ))

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  cudnn.benchmark = True

  train_loaders = [FeaturesLoader(
      features_path=list_data_path,
      feature_dim=args.feature_dim,
      annotation_path=f"{list_data_path}/train-annot.txt",
      iterations=args.iterations_per_epoch,
  ) for list_data_path in list_data_paths]

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
  model = RegressionNetwork(feature_dim, count_views=len(list_data_paths))
  model = model.to(device)
  model = model.train()
  optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr_base, eps=1e-8)

  criterion = RegularizedLoss(model).to(device)

  
  iteration = 0
  total_loss = 0


  for i, data_path in enumerate(list_data_paths):
    print(f"input data (camera {i})", list_data_paths[i])

  print(model)
  for iteration, (data_cams) in enumerate(zip(*train_loaders)):

      normal_positions, abnormal_positions = train_loaders[0].get_positions()

      glob_cis_n, glob_cis_a = list(),list()
      for i in range(len(train_loaders)):  
        batch_ci, targets_ci = train_loaders[i].get_features_by_positions(normal_positions, abnormal_positions)
        batch_ci = data_to_device(batch_ci, device)
        targets_ci = data_to_device(targets_ci, device)
        outputs_ci = [model.forward(b.to(device), view_index = i+1 ) for b in batch_ci]
        glob_ci_n, glob_ci_a = outputs_ci[:30], outputs_ci[30:]
        glob_cis_n.append(glob_ci_n)
        glob_cis_a.append(glob_ci_a)
      
    
      glob_n = [torch.mean(torch.cat(sbs, axis=1),axis=1) for sbs in zip(*glob_cis_n) ]
      glob_a = [torch.mean(torch.cat(sbs, axis=1),axis=1) for sbs in zip(*glob_cis_a) ]
      glob = glob_n + glob_a # concatena as bags de scores em uma unica bag 
      loss_unique = criterion(glob, targets_ci)
      print(len(glob),len(targets_ci))
      optimizer.zero_grad()
      loss_final = loss_unique
      loss_final.backward()
      optimizer.step()

      total_loss += loss_final.item()
      print(iteration, loss_final.item())

  loss = total_loss / len(train_loaders[0])
  train_loss = loss
  return model

def evaluation(model, Views, path_cams):
    loss_aggr_name = '-'
    model = model.to('cpu').eval()

    for i in range(len(path_cams)):
        model.fc0[f"{i+1}"] = model.fc0[f"{i+1}"].to("cpu")


    dct_test_data = {'data':{},}
    cudnn.benchmark = True
    for feat in ['data',]:
        for cam in Views:
            test_path = path_cams[cam] #f'../data/test-view-{cam}.npy'
            dct_test_data[feat][cam] = np.load(test_path, allow_pickle=True)

    auc_plot_data = dict()
    auc_plot_data['MAX'] = dict()


    formatted_results = list()
    all_y_true = list()
    all_y_pred_cams =  [[] for view in Views] 
    for bags_cams in zip(*[dct_test_data[feat][view] for view in Views]):

        for i in range(len(bags_cams)):
            b_cam_i = munchify(bags_cams[i])
            features_cam_i = torch.tensor(b_cam_i.X_i).reshape([1,-1, b_cam_i.X_i.shape[1]  ])
            out_cam_i = model.forward(features_cam_i.float(), view_index = i+1).squeeze(-1)
            y_pred_cam_i = out_cam_i.detach().numpy().ravel()
            y_pred_cam_i = np.array([[_y]*16 for _y in y_pred_cam_i ]).ravel()    
            all_y_pred_cams[i] += list(y_pred_cam_i.ravel())

            if i == 0:
                y_true = bags_cams[0]["y_fi"].ravel()
                all_y_true += list(y_true.ravel())
        
    
    all_y_true = np.array(all_y_true).ravel()
    for i, view in enumerate(Views):
        all_y_pred_cams[i] = np.array(all_y_pred_cams[i]).ravel()

    '''
    Overal Cameras MAX
    '''

    all_y_pred_fus_max = np.array([
        all_y_pred_cams[i] for i, view in enumerate(Views)
    ])
    print(all_y_true.shape, all_y_pred_fus_max.shape)


    all_y_pred_fus_max = all_y_pred_fus_max.max(axis=0)

    print(all_y_true.shape, all_y_pred_fus_max.shape)

    cm_fus_max = confusion_matrix(all_y_true, np.round(all_y_pred_fus_max), labels=[0, 1])
    cm_fus_max_html = pd.DataFrame(cm_fus_max).to_html()
    all_y_pred_ = all_y_pred_fus_max
    fpr_, tpr_, thresholds_ = roc_curve(all_y_true, all_y_pred_)
    _auc = auc(fpr_, tpr_) * 100
    cm = confusion_matrix(all_y_true, np.round(all_y_pred_), labels=[0, 1])
    tn, fp, fn, tp = confusion_matrix(all_y_true, np.round(all_y_pred_), labels=[0,1]).ravel()
    _fpr = fp/(fp + tn) * 100
    _acc = (tp+tn)/(tp+tn+fp+fn) * 100
    _bacc = sklearn.metrics.balanced_accuracy_score(all_y_true, np.round(all_y_pred_)) * 100
    _prec = tp/(tp+fp) * 100
    _rec = tp/(tp+fn) * 100
    _f1 = 2 * (_prec * _rec) / (_prec + _rec)
    str_fus_max = f'{feat.upper()}\tMAX\tAUC:{_auc:.2f}%\tFPR:{_fpr:.2f}%'
    df_metrics_max = pd.DataFrame([[_auc, _fpr, _acc,_prec,_rec,_f1]], columns=['AUC','FPR','ACC','PREC','REC','F1'])
    auc_plot_data['MAX'] = [fpr_, tpr_, thresholds_, _auc]

    df_metrics_all = pd.concat([df_metrics_max])

    df_metrics_all.index = ['MAX']

    df_metrics_all.insert(0, 'Decision', ['MAX',])
    df_metrics_all.insert(0, 'Cameras', f'CAMS')
    df_metrics_all = df_metrics_all.round(2)

    return auc_plot_data, df_metrics_all

if __name__ == "__main__":
    
    trained_model = train_mcmil2(
        list_data_paths = [
            "/media/dev/LaCie/MC-VAD-MIL-OUT/pets2009-sultani-format-i3d/pets2009-rgb-seed-1-train-view-002-ekos-input",
            "/media/dev/LaCie/MC-VAD-MIL-OUT/pets2009-sultani-format-i3d/pets2009-rgb-seed-1-train-view-003-ekos-input", 
        ],
        iterations=10000,
        batch_size=60,
    )

    dfs_list = list()
    auc_data, df = evaluation(
        model=trained_model,
        Views=["002","003"],
        path_cams = {
            "002":"/media/dev/LaCie/MC-VAD-MIL-DATA/center-crop/pets2009-relabeled-i3d/pets2009-rgb-seed-1-test-view-002.npy",
            "003":"/media/dev/LaCie/MC-VAD-MIL-DATA/center-crop/pets2009-relabeled-i3d/pets2009-rgb-seed-1-test-view-003.npy",
        })
    display(df)