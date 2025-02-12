import numpy as np
from torch.utils.data import Dataset, DataLoader
import utils
import options
import os
import pickle
import random
import torch


class dataset(Dataset):
    def __init__(self, args, train=True, trainlist=None, testlist=None):
        """
        :param args:
        self.dataset_path: path to dir contains anomaly datasets
        self.dataset_name: name of dataset which use now
        self.feature_modal: features froCset for testing
        self.train: boolen type, if it is True, the dataset class return training data
        self.t_max: the max of sampling in training
        """
        self.args = args
        self.dataset_path = args.dataset_path

        self.feature_modal = args.feature_modal
        
        self.feature_pretrain_model = args.feature_pretrain_model
        
        self.feature_path_cams = dict()
        self.videoname_cams = dict()
        for view in args.camera_list:
            self.feature_path_cams[view] = os.path.join(self.dataset_path, view, 'features_video',
                                                self.feature_pretrain_model, self.feature_modal)
            self.videoname_cams = os.listdir(self.feature_path_cams[view])

        if trainlist:
            self.trainlist = self.txt2list(trainlist)
            self.testlist = self.txt2list(testlist)
        else:
            self.trainlist = self.txt2list(
                txtpath=os.path.join(self.dataset_path, args.camera_list[0], 'train_split.txt'))
            self.testlist = self.txt2list(txtpath=os.path.join(self.dataset_path, args.camera_list[0], 'test_split.txt'))
        
        self.video_label_dict = self.pickle_reader(
            file=os.path.join(self.dataset_path, args.camera_list[0], 'GT', 'video_label.pickle'))
        
        self.normal_video_train, self.anomaly_video_train = self.p_n_split_dataset(self.video_label_dict, self.trainlist)
        self.train = train
        self.t_max = args.max_seqlen
    

    def txt2list(self, txtpath=''):
        """
        use for generating list from text file
        :param txtpath: path of text file
        :return: list of text file
        """
        with open(file=txtpath, mode='r') as f:
            filelist = f.readlines()
        return filelist

    def pickle_reader(self, file=''):
        with open(file=file, mode='rb') as f:
            #video_label_dict = pickle.load(f)
            video_label_dict = np.load(f, allow_pickle = True)
        return video_label_dict

    def p_n_split_dataset(self, video_label_dict, trainlist):
        normal_video_train = []
        anomaly_video_train = []
        for t in trainlist:
            t = t.replace("\n","")
            #print(video_label_dict)
            #print(f"key:{t}.")
            #print(type(video_label_dict[0]))
            #print(video_label_dict.keys())
            if video_label_dict.item().get(t) == [1.0]:
                anomaly_video_train.append(t.replace('\n', ''))
            else:
                normal_video_train.append(t.replace('\n', ''))
        return normal_video_train, anomaly_video_train

        # for k, v in video_label_dict.items():
        #     if v[0] == 1.:
        #         anomaly_video_train.append(k)
        #     else:
        #         normal_video_train.append(k)
        # return normal_video_train, anomaly_video_train

    def __getitem__(self, index):

        if self.train:
            anomaly_train_video_name = []
            normaly_train_video_name = []
            
            anomaly_start_index_cams = {ds:[] for ds in self.args.camera_list}
            anomaly_len_index_cams = {ds:[] for ds in self.args.camera_list}

            normaly_start_index_cams = {ds:[] for ds in self.args.camera_list}
            normaly_len_index_cams = {ds:[] for ds in self.args.camera_list}


            anomaly_indexs = random.sample(self.anomaly_video_train, self.args.sample_size)
            normaly_indexs = random.sample(self.normal_video_train, self.args.sample_size)

            anomaly_features_cams = {ds:torch.zeros(0) for ds in self.args.camera_list} 
            normaly_features_cams = {ds:torch.zeros(0) for ds in self.args.camera_list}

            anomaly_feature_cams = dict()
            normaly_feature_cams = dict()

            for a_i, n_i in zip(anomaly_indexs, normaly_indexs):
                anomaly_data_video_name = a_i.replace('\n', '')
                normaly_data_video_name = n_i.replace('\n', '')
                anomaly_train_video_name.append(anomaly_data_video_name)
                normaly_train_video_name.append(normaly_data_video_name)
                
                for ds in self.args.camera_list:
                    anomaly_feature_cams[ds] = np.load(file=os.path.join(self.feature_path_cams[ds], anomaly_data_video_name, 'feature.npy'))
                    anomaly_len_index_cams[ds].append(anomaly_feature_cams[ds].shape[0])
                    anomaly_feature_cams[ds], r = utils.process_feat_sample(anomaly_feature_cams[ds], self.t_max)
                    anomaly_start_index_cams[ds].append(r)
                    anomaly_feature_cams[ds] = torch.from_numpy(anomaly_feature_cams[ds]).unsqueeze(0)
                    
                    normaly_feature_cams[ds] = np.load(file=os.path.join(self.feature_path_cams[ds], normaly_data_video_name, 'feature.npy'))
                    normaly_len_index_cams[ds].append(normaly_feature_cams[ds].shape[0])
                    normaly_feature_cams[ds], r = utils.process_feat(normaly_feature_cams[ds], self.t_max, self.args.sample_step)
                    normaly_feature_cams[ds] = torch.from_numpy(normaly_feature_cams[ds]).unsqueeze(0)
                    normaly_start_index_cams[ds].append(r)
                    
                    anomaly_features_cams[ds] = torch.cat((anomaly_features_cams[ds], anomaly_feature_cams[ds]), dim=0)  # combine anomaly_feature of different a_i
                    normaly_features_cams[ds] = torch.cat((normaly_features_cams[ds], normaly_feature_cams[ds]), dim=0)  # combine normaly_feature of different n_i


            if self.args.label_type == 'binary':
                normaly_label = torch.cat((torch.ones((self.args.sample_size, 1)), torch.zeros((self.args.sample_size, 1))), dim=1)
                anomaly_label = torch.cat((torch.ones((self.args.sample_size, 1)), torch.ones((self.args.sample_size, 1))), dim=1)
            elif self.args.label_type == 'unary':
                normaly_label = torch.zeros((self.args.sample_size, 1))
                anomaly_label = torch.ones((self.args.sample_size, 1))
            else:
                normaly_label = torch.cat((torch.ones((self.args.sample_size, 1)), torch.zeros((self.args.sample_size, 1))), dim=1)
                anomaly_label = torch.cat((torch.zeros((self.args.sample_size, 1)), torch.ones((self.args.sample_size, 1))), dim=1)

            train_video_name = anomaly_train_video_name + normaly_train_video_name

            start_index_cams = dict()
            len_index_cams = dict()
            for ds in self.args.camera_list:
                start_index_cams[ds] = anomaly_start_index_cams[ds] + normaly_start_index_cams[ds]
                len_index_cams[ds] = anomaly_len_index_cams[ds] + normaly_len_index_cams[ds]
            
            ds0 = self.args.camera_list[0]
            return anomaly_features_cams, normaly_features_cams, anomaly_label, normaly_label, train_video_name, start_index_cams[ds0], len_index_cams[ds0]
        
        else:
            #data_video_name = self.testlist[index].replace('\n', '').replace('Ped', 'ped')
            #single_return = list()
            #for ds in self.args.camera_list:
            #    feature_cam_ds = np.load(file=os.path.join(self.feature_path_cams[ds], data_video_name, 'feature.npy'))
            #    single_return.append(feature_cam_ds)
            #single_return.append(data_video_name)
            #return single_return
            data_video_name = self.testlist[index].replace('\n', '').replace('Ped', 'ped')
            feature_cams = dict()
            for ds in self.args.camera_list:
                feature_cams[ds] = np.load(
                    file=os.path.join(self.feature_path_cams[ds], data_video_name, 'feature.npy'))
            return feature_cams, data_video_name

    def __len__(self):
        if self.train:
            return len(self.trainlist)

        else:
            return len(self.testlist)