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
        self.feature_modal: features from different input, contain rgb, flow or combine of above type
        self.feature_pretrain_model: the model name of feature extraction
        self.feature_path: the dir contain all features, use for training and testing
        self.videoname: videonames of dataset
        self.trainlist: videonames of dataset for training
        self.testlist: videonames of dataset for testing
        self.train: boolen type, if it is True, the dataset class return training data
        self.t_max: the max of sampling in training
        """
        self.args = args
        self.dataset_path = args.dataset_path

        self.dataset_name_camA = args.dataset_name_camA
        self.dataset_name_camB = args.dataset_name_camB

        self.feature_modal = args.feature_modal
        
        self.feature_pretrain_model = args.feature_pretrain_model
        
        if self.feature_pretrain_model == 'c3d' or self.feature_pretrain_model == 'c3d_ucf':
            self.feature_layer = args.feature_layer
            self.feature_path_camA = os.path.join(self.dataset_path, self.dataset_name_camA, 'features_video',
                                             self.feature_pretrain_model, self.feature_layer, self.feature_modal)
            self.feature_path_camB = os.path.join(self.dataset_path, self.dataset_name_camB, 'features_video',
                                             self.feature_pretrain_model, self.feature_layer, self.feature_modal)
        else:
            self.feature_path_camA = os.path.join(self.dataset_path, self.dataset_name_camA, 'features_video',
                                             self.feature_pretrain_model, self.feature_modal)
            self.feature_path_camB = os.path.join(self.dataset_path, self.dataset_name_camB, 'features_video',
                                             self.feature_pretrain_model, self.feature_modal)
            
        self.videoname_camA = os.listdir(self.feature_path_camA)
        self.videoname_camB = os.listdir(self.feature_path_camB)


        if trainlist:
            self.trainlist = self.txt2list(trainlist)
            self.testlist = self.txt2list(testlist)
        else:
            self.trainlist = self.txt2list(
                txtpath=os.path.join(self.dataset_path, self.dataset_name_camA, 'train_split.txt'))
            self.testlist = self.txt2list(txtpath=os.path.join(self.dataset_path, self.dataset_name_camA, 'test_split.txt'))
        
        self.video_label_dict = self.pickle_reader(
            file=os.path.join(self.dataset_path, self.dataset_name_camA, 'GT', 'video_label.pickle'))
        
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
            
            anomaly_start_index_camA = []
            anomaly_len_index_camA = []
            anomaly_start_index_camB = []
            anomaly_len_index_camB = []
            
            normaly_start_index_camA = []
            normaly_len_index_camA = []
            normaly_start_index_camB = []
            normaly_len_index_camB = []
            
            anomaly_indexs = random.sample(self.anomaly_video_train, self.args.sample_size)
            normaly_indexs = random.sample(self.normal_video_train, self.args.sample_size)

            #print("batch", "A:", anomaly_indexs, "N:", normaly_indexs)
            
            anomaly_features_camA = torch.zeros(0)
            normaly_features_camA = torch.zeros(0)

            anomaly_features_camB = torch.zeros(0)
            normaly_features_camB = torch.zeros(0)

            for a_i, n_i in zip(anomaly_indexs, normaly_indexs):
                anomaly_data_video_name = a_i.replace('\n', '')
                normaly_data_video_name = n_i.replace('\n', '')
                anomaly_train_video_name.append(anomaly_data_video_name)
                normaly_train_video_name.append(normaly_data_video_name)
                
                # cam A
                anomaly_feature = np.load(file=os.path.join(self.feature_path_camA, anomaly_data_video_name, 'feature.npy'))
                anomaly_len_index_camA.append(anomaly_feature.shape[0])
                anomaly_feature, r = utils.process_feat_sample(anomaly_feature, self.t_max)
                anomaly_start_index_camA.append(r)
                anomaly_feature = torch.from_numpy(anomaly_feature).unsqueeze(0)
                #print("debug", os.path.join(self.feature_path_camA, normaly_data_video_name, 'feature.npy'))
                normaly_feature = np.load(
                    file=os.path.join(self.feature_path_camA, normaly_data_video_name, 'feature.npy'))
                normaly_len_index_camA.append(normaly_feature.shape[0])
                normaly_feature, r = utils.process_feat(normaly_feature, self.t_max, self.args.sample_step)
                normaly_feature = torch.from_numpy(normaly_feature).unsqueeze(0)
                normaly_start_index_camA.append(r)
                anomaly_features_camA = torch.cat((anomaly_features_camA, anomaly_feature),
                                                dim=0)  # combine anomaly_feature of different a_i
                normaly_features_camA = torch.cat((normaly_features_camA, normaly_feature),
                                                dim=0)  # combine normaly_feature of different n_i

                # cam B
                anomaly_feature = np.load(file=os.path.join(self.feature_path_camB, anomaly_data_video_name, 'feature.npy'))
                anomaly_len_index_camB.append(anomaly_feature.shape[0])
                anomaly_feature, r = utils.process_feat_sample(anomaly_feature, self.t_max)
                anomaly_start_index_camB.append(r)
                anomaly_feature = torch.from_numpy(anomaly_feature).unsqueeze(0)
                normaly_feature = np.load(
                    file=os.path.join(self.feature_path_camB, normaly_data_video_name, 'feature.npy'))
                normaly_len_index_camB.append(normaly_feature.shape[0])
                normaly_feature, r = utils.process_feat(normaly_feature, self.t_max, self.args.sample_step)
                normaly_feature = torch.from_numpy(normaly_feature).unsqueeze(0)
                normaly_start_index_camB.append(r)
                anomaly_features_camB = torch.cat((anomaly_features_camB, anomaly_feature),
                                                dim=0)  # combine anomaly_feature of different a_i
                normaly_features_camB = torch.cat((normaly_features_camB, normaly_feature),
                                                dim=0)  # combine normaly_feature of different n_i


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
            start_index_camA = anomaly_start_index_camA + normaly_start_index_camA
            len_index_camA = anomaly_len_index_camA + normaly_len_index_camA
            start_index_camB = anomaly_start_index_camB + normaly_start_index_camB
            len_index_camB = anomaly_len_index_camB + normaly_len_index_camB

            return [anomaly_features_camA, normaly_features_camA],[anomaly_features_camB, normaly_features_camB], [anomaly_label, normaly_label], [train_video_name, start_index_camA, len_index_camA]
        else:
            data_video_name = self.testlist[index].replace('\n', '').replace('Ped', 'ped')
            self.feature_camA = np.load(file=os.path.join(self.feature_path_camA, data_video_name, 'feature.npy'))
            self.feature_camB = np.load(file=os.path.join(self.feature_path_camB, data_video_name, 'feature.npy'))
            return self.feature_camA,self.feature_camB, data_video_name

    def __len__(self):
        if self.train:
            return len(self.trainlist)

        else:
            return len(self.testlist)