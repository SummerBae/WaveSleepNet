import os
import glob
import torch
import numpy as np
from transform import *
from torch.utils.data import Dataset


class EEGDataLoader(Dataset):

    def __init__(self, config, fold, set='train'):

        self.set = set
        self.fold = fold

        # self.sr = 100        
        self.data_mode = config['mode']
        self.model_name = config['backbone']['name']
        self.dset_cfg = config['dataset']
        
        self.root_dir = self.dset_cfg['root_dir']
        self.dset_name = self.dset_cfg['name']
        self.sr = self.dset_cfg['sampleing_rate']
        self.num_splits = self.dset_cfg['num_splits']
        self.eeg_channel = self.dset_cfg['eeg_channel']
        
        self.seq_len = self.dset_cfg['seq_len']
        self.target_idx = self.dset_cfg['target_idx']
        
        self.training_mode = config['training_params']['mode']

        self.dataset_path = os.path.join(self.root_dir, 'dset', self.dset_name, 'npz')
        if self.dset_name == 'shhs':
            self.epochs = self.split_dataset()      
        else:
            self.inputs, self.labels, self.epochs = self.split_dataset()
        
        self.transform = Compose(
            transforms=[
                RandomAmplitudeScale(),
                RandomTimeShift(),
                RandomDCShift(),
                RandomZeroMasking(),
                RandomAdditiveGaussianNoise(),
                RandomBandStopFilter(),
            ]
        )
        if self.training_mode == 'pretrain':
            self.two_transform = TwoTransform(self.transform)
        
    def __len__(self):
        return len(self.epochs)

    def __getitem__(self, idx):
        n_sample = 30 * self.sr * self.seq_len
        if self.dset_name == 'shhs':
            file_data = np.load('/mnt/d/summer/shhs_dataset_{}/'.format(self.set) + self.epochs[idx], allow_pickle=True).item()
            inputs = file_data['x']
            labels = np.array(file_data['y'])
            labels = torch.from_numpy(labels).long()
        else:
            file_idx, idx, seq_len = self.epochs[idx]
            inputs = self.inputs[file_idx][idx:idx+seq_len]
            labels = self.labels[file_idx][idx:idx+seq_len]
            labels = torch.from_numpy(labels).long()
            labels = labels[self.target_idx]

        if self.set == 'train':
            if self.training_mode == 'pretrain':
                assert seq_len == 1
                input_a, input_b = self.two_transform(inputs)
                input_a = torch.from_numpy(input_a).float()
                input_b = torch.from_numpy(input_b).float()
                inputs = [input_a, input_b]
            elif self.training_mode in ['scratch', 'fullyfinetune', 'freezefinetune']:
                inputs = inputs.reshape(1, n_sample)
                inputs = torch.from_numpy(inputs).float()
            else:
                raise NotImplementedError
        else:
            if not self.training_mode == 'pretrain':
                inputs = inputs.reshape(1, n_sample)
            inputs = torch.from_numpy(inputs).float()
        return inputs, labels

    def split_dataset(self):

        file_idx = 0
        inputs, labels, epochs = [], [], []
        data_root = os.path.join(self.dataset_path, self.eeg_channel)
        data_fname_list = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(data_root, '*.npz')))]
        data_fname_dict = {'train': [], 'test': [], 'val': []}
        # data_fname_dict = {'train': [], 'test': []}
        split_idx_list = np.load(os.path.join('/summer/ProtoPSleep/split_idx', 'idx_{}.npy'.format(self.dset_name)), allow_pickle=True)

        assert len(split_idx_list) == self.num_splits
    
        if self.dset_name == 'Sleep-EDF-2013':
            for i in range(len(data_fname_list)):
                subject_idx = int(data_fname_list[i][3:5])
                if subject_idx == self.fold - 1:
                    data_fname_dict['test'].append(data_fname_list[i])
                elif subject_idx in split_idx_list[self.fold - 1]:
                    data_fname_dict['val'].append(data_fname_list[i])
                else:
                    data_fname_dict['train'].append(data_fname_list[i])    

        elif self.dset_name == 'Sleep-EDF-2018':
            for i in range(len(data_fname_list)):
                subject_idx = int(data_fname_list[i][3:5])
                if subject_idx in split_idx_list[self.fold - 1][self.set]:
                    data_fname_dict[self.set].append(data_fname_list[i])
                    
        elif self.dset_name == 'MASS' or self.dset_name == 'Physio2018' or self.dset_name == 'shhs':
            for i in range(len(data_fname_list)):
                if i in split_idx_list[self.fold - 1][self.set]:
                    data_fname_dict[self.set].append(data_fname_list[i])
        else:
            raise NameError("dataset '{}' cannot be found.".format(self.dataset))
            
        if self.dset_name == 'shhs':
            if self.set == 'train':
                epochs = [os.path.basename(x) for x in sorted(glob.glob(os.path.join('/mnt/d/summer/shhs_dataset_train/', '*.npy')))]
            elif self.set == 'val':
                epochs = [os.path.basename(x) for x in sorted(glob.glob(os.path.join('/mnt/d/summer/shhs_dataset_val/', '*.npy')))]
            elif self.set == 'test':
                epochs = [os.path.basename(x) for x in sorted(glob.glob(os.path.join('/mnt/d/summer/shhs_dataset_test/', '*.npy')))]
            return epochs
        else:
            for data_fname in data_fname_dict[self.set]:
                npz_file = np.load(os.path.join(data_root, data_fname))
                inputs.append(npz_file['x'])
                labels.append(npz_file['y'])
                seq_len = self.seq_len
                if self.dset_name== 'MASS' and ('-02-' in data_fname or '-04-' in data_fname or '-05-' in data_fname):
                    seq_len = int(self.seq_len * 1.5)
                for i in range(len(npz_file['y']) - seq_len + 1):
                    epochs.append([file_idx, i, seq_len])
                file_idx += 1
            return inputs, labels, epochs
