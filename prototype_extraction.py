import os, sys
sys.path.append('/summer/ProtoPSleep/')
import json
import argparse
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from utils import *
from loader import EEGDataLoader
from train_mtcl import OneFoldTrainer
from models.protop import ProtoPNet      
from models.main_model import MainModel
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


class OneFoldEvaluator(OneFoldTrainer):
    def __init__(self, args, fold, config):
        self.args = args
        self.fold = fold
        
        self.cfg = config
        self.ds_cfg = config['dataset']
        self.tp_cfg = config['training_params']
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('[INFO] Config name: {}'.format(config['name']))

        self.model = self.build_model()
        self.loader_dict = self.build_dataloader()
        self.criterion = nn.CrossEntropyLoss()
        ## Choose the checkpoint of the model which you want to explain
        self.ckpt_path = ''
        self.ckpt_name = 'ckpt_fold-{0:02d}.pth'.format(self.fold)
        
    def build_model(self):
        if self.cfg['backbone']['name'] == 'ProtoP':
            model = ProtoPNet(self.cfg) 
        else:
            model = MainModel(self.cfg)
        print('[INFO] Number of params of model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
        model = torch.nn.DataParallel(model, device_ids=list(range(len(self.args.gpu.split(",")))))
        model.to(self.device)
        print('[INFO] Model prepared, Device used: {} GPU:{}'.format(self.device, self.args.gpu))

        return model

    def build_dataloader(self):
        train_dataset = EEGDataLoader(self.cfg, self.fold, set='train')
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.tp_cfg['batch_size'], shuffle=False, num_workers=4*len(self.args.gpu.split(",")), pin_memory=True)
        test_dataset = EEGDataLoader(self.cfg, self.fold, set='test')
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.tp_cfg['batch_size'], shuffle=False, num_workers=4*len(self.args.gpu.split(",")), pin_memory=True)
        print('[INFO] Dataloader prepared')

        return {'train': train_loader, 'test':test_loader} 
 
    def prototype_perturbation(self, fold):
        fs = 100
        repeat_num =  4
        time_duration = 3
        length = int(time_duration * fs)
        print('\n[INFO] Fold: {}'.format(self.fold))
        self.model.load_state_dict(torch.load(os.path.join(self.ckpt_path, self.ckpt_name)))
        argmin, features, targets = self.distance(repeat_num)     ##[B, proto_num, K]
        proto_result = {}
        for j in range(10):
            proto_result[str(j)] = {}
            proto_result[str(j)]['wavelet'] = []
            proto_result[str(j)]['signal'] = []
            proto_result[str(j)]['targets'] = []
            proto_result[str(j)]['plot'] = []
            for k in range(repeat_num):
                ## split signal segments
                signal = np.expand_dims(features[str(j)][k], 0)        ## [1, 30000]
                idx = argmin[str(j)][k]
                perturbation_dataset = PerturbationSignal(signal, length)
                perturbation_loader = DataLoader(dataset=perturbation_dataset, batch_size=self.tp_cfg['batch_size'], shuffle=False, num_workers=4*len(self.args.gpu.split(",")), pin_memory=True)
                pert_dist = self.perturbation_dist(perturbation_loader)     ##[B, proto_num, K]
                pert_dist = pert_dist[1:, j, idx]
                protop_wavelet = np.squeeze(signal)[np.argmax(pert_dist): np.argmax(pert_dist) + length]
                proto_plot = np.empty(np.squeeze(signal).shape)
                proto_plot.fill(None)
                proto_plot[np.argmax(pert_dist) : np.argmax(pert_dist) + length] = np.squeeze(protop_wavelet) 
                proto_result[str(j)]['wavelet'].append(protop_wavelet)
                proto_result[str(j)]['signal'].append(signal)
                proto_result[str(j)]['targets'].append(targets[str(j)][k])
                proto_result[str(j)]['plot'].append(proto_plot)
                ## plot heat map
                prototype_plot(protop_wavelet, proto_plot, signal, targets[str(j)][k], j)
                print('P{}, repeat_id{} show!!'.format(j, k))
        return


    def per_prototype_perturbation(self, p_idx, repeat_num, trg_cls=None):
        fs = 100
        time_duration = 3
        length = int(time_duration * fs)
        print('\n[INFO] Fold: {}'.format(self.fold))
        self.model.load_state_dict(torch.load(os.path.join(self.ckpt_path, self.ckpt_name)))
        self.model.eval()
        distance = []
        signals = []
        label = []
        trgs = []
        for i, (inputs, labels) in enumerate(self.loader_dict['train']):
            inputs = inputs.to(self.device)
            labels = labels.view(-1).to(self.device)
            outputs = self.model(inputs)
            distance.extend(self.model.module.distance.cpu().detach().numpy())
            signals.extend(inputs.cpu().detach().numpy())
            label.extend(labels.cpu().detach().numpy())
            trgs.extend(np.argmax(outputs.cpu().detach().numpy(), 1))
        distance = np.array(distance)
        label = np.array(label)
        trgs = np.array(trgs)
        if trg_cls != None:
            trg_cls_idx = np.intersect1d(np.argwhere(label == trg_cls), np.argwhere(trgs == trg_cls))
            proto_dist_j = distance[trg_cls_idx, p_idx, :]
            signals = np.array(signals)[trg_cls_idx]
            label = label[trg_cls_idx]
        else:
            proto_dist_j = distance[:, p_idx, :]
        proto_dist_j_sort = np.sort(proto_dist_j.reshape(-1))
        for k in range(repeat_num):
            global_argmin_protop_dist_j = np.argwhere(proto_dist_j == proto_dist_j_sort[k])[0]
            print('Distance:{}'.format(proto_dist_j_sort[k]))
            if trg_cls != None:
                global_argmin_protop_dist_j[0] = trg_cls_idx[global_argmin_protop_dist_j[0]]
            signal = signals[global_argmin_protop_dist_j[0]]
            targets = label[global_argmin_protop_dist_j[0]]
            idx = global_argmin_protop_dist_j[1]
            ## split signal segments
            perturbation_dataset = PerturbationSignal(signal, length)
            perturbation_loader = DataLoader(dataset=perturbation_dataset, batch_size=self.tp_cfg['batch_size'], shuffle=False, num_workers=4*len(self.args.gpu.split(",")), pin_memory=True)
            pert_dist = self.perturbation_dist(perturbation_loader)     ##[B, proto_num, K]
            pert_dist = pert_dist[1:, p_idx, idx]
            protop_wavelet = np.squeeze(signal)[np.argmax(pert_dist): np.argmax(pert_dist) + length]
            proto_plot = np.empty(np.squeeze(signal).shape)
            proto_plot.fill(None)
            proto_plot[np.argmax(pert_dist) : np.argmax(pert_dist) + length] = np.squeeze(protop_wavelet) 
            ## plot heat map
            prototype_plot(protop_wavelet, proto_plot, signal, targets, p_idx)
 

    def prototype_classify_dependence(self, plot=True):
        weight_result = {}
        print('\n[INFO] Fold: {}'.format(self.fold))
        self.model.load_state_dict(torch.load(os.path.join(self.ckpt_path, self.ckpt_name)))
        ave_weight_cls, min_weight_cls, ave_weight_cls_err, min_weight_cls_err = self._get_similairty() 
        weight_result['ave'] = ave_weight_cls
        weight_result['min'] = min_weight_cls
        weight_result['ave_err'] = ave_weight_cls_err
        weight_result['min_err'] = min_weight_cls_err
        ## save result
        if plot:
            x = np.arange(10)
            total_width, n = 0.8, 5
            width = total_width / n
            x = x - (total_width - width) / 2
            show_weight_cls = ave_weight_cls
            fig, ax = plt.subplots(1, 2, figsize=(16, 5))
            ax[0].bar(x, show_weight_cls[0], width = width, label = 'W')
            ax[0].bar(x+ width, show_weight_cls[1], width = width, label = 'N1')
            ax[0].bar(x+ 2 * width, show_weight_cls[2], width = width, label = 'N2')
            ax[0].bar(x+ 3 * width, show_weight_cls[3], width = width, label = 'N3')
            ax[0].bar(x+ 4 * width, show_weight_cls[4], width = width, label = 'R')
            ax[1].bar(x, min_weight_cls[0], width = width, label = 'W')
            ax[1].bar(x + width, min_weight_cls[1], width = width, label = 'N1')
            ax[1].bar(x +2 * width, min_weight_cls[2], width = width, label = 'N2')
            ax[1].bar(x +3 * width, min_weight_cls[3], width = width, label = 'N3')
            ax[1].bar(x +4 * width, min_weight_cls[4], width = width, label = 'R')
            plt.legend()
            plt.show()

        np.save('/summer/ProtoPSleep/exp2/weight_result_{}.npy'.format(self.fold), weight_result)
        print('Weight Result Saved Done!!')
        

    def distance(self, repeat_times):
        self.model.eval()
        distance = []
        signals = []
        label = []
        for i, (inputs, labels) in enumerate(self.loader_dict['train']):
            inputs = inputs.to(self.device)
            labels = labels.view(-1).to(self.device)
            outputs = self.model(inputs)
            distance.extend(self.model.module.distance.cpu().detach().numpy())
            signals.extend(inputs.cpu().detach().numpy())
            label.extend(labels.cpu().detach().numpy())
        features = {}
        targets = {}
        argmin = {}
        distance = np.array(distance)
        for i in range(10):
            features[str(i)] = []
            targets[str(i)] = []
            argmin[str(i)] = []
            proto_dist_j = distance[:, i, :]
            proto_dist_j_sort = np.sort(proto_dist_j.reshape(-1))
            for k in range(repeat_times):
                global_argmin_protop_dist_j = np.argwhere(proto_dist_j == proto_dist_j_sort[k])[0]
                features[str(i)].extend(signals[global_argmin_protop_dist_j[0]])
                targets[str(i)].append(label[global_argmin_protop_dist_j[0]])
                argmin[str(i)].append(global_argmin_protop_dist_j[1])
        print(targets)
        return argmin, features, targets

    def _get_similairty(self):
        self.model.eval()
        outputs = []
        trgs = []
        proportion_similarity = []
        wave_similarity = []
        for i, (inputs, labels) in enumerate(self.loader_dict['train']):
            inputs = inputs.to(self.device)
            labels = labels.view(-1).to(self.device)
            outputs.extend(self.model(inputs).cpu().detach().numpy())
            trgs.extend(labels.cpu().detach().numpy())
            proportion_similarity.extend(self.model.module.proportion.cpu().detach().numpy())        ##[B, P]
            wave_similarity.extend(self.model.module.similarity.cpu().detach().numpy())   #[B, P]
        cls_idx = {}
        cls_idx_err = {}
        cls_idx_correct = {}
        weight = self.model.module.fc.weight.cpu().detach().numpy()    #[5, 2 * P]
        ave_weight_cls = {}
        min_weight_cls = {}
        ave_weight_cls_err = {}
        min_weight_cls_err = {}
        for j in range(5):
            cls_idx[j] = np.argwhere(np.argmax(np.array(outputs), 1) == j)
            cls_idx_err[j] = np.intersect1d(np.argwhere(np.array(trgs) == j), np.argwhere(np.argmax(np.array(outputs), 1) != j))
            cls_idx_correct[j] = np.intersect1d(np.argwhere(np.array(trgs) == j), np.argwhere(np.argmax(np.array(outputs), 1) == j))
            ave_weight_ = weight[j, ::2]     #[P]
            min_weight_ = weight[j, 1::2]     #[P]
            ave_weight_cls[j] = np.squeeze(np.array(proportion_similarity)[cls_idx_correct[j]] * ave_weight_).mean(0)
            min_weight_cls[j] = np.squeeze(np.array(wave_similarity)[cls_idx_correct[j]] * min_weight_).mean(0)
            ave_weight_cls_err[j] = np.squeeze(np.array(proportion_similarity)[cls_idx_err[j]] * ave_weight_).mean(0)
            min_weight_cls_err[j] = np.squeeze(np.array(wave_similarity)[cls_idx_err[j]] * min_weight_).mean(0)
        return ave_weight_cls, min_weight_cls, ave_weight_cls_err, min_weight_cls_err         


    def perturbation_dist(self, dataloader):
        self.model.eval()
        distance = []
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            labels = labels.view(-1).to(self.device)
            outputs = self.model(inputs)
            distance.extend(self.model.module.distance.cpu().detach().numpy())
        return np.array(distance)


def prototype_plot(prototype_wavelet, proto_plot, feature, target, p_num):
    fs = 100
    scale = np.max(np.abs(feature))
    fig, ax = plt.subplots(1, 3, figsize=(24, 3))
    fft_visualize(ax[0], np.squeeze(prototype_wavelet), fs=fs)
    ax[1].plot(np.arange(prototype_wavelet.shape[0]) / fs, np.squeeze(prototype_wavelet))
    ax[1].set_ylim([-120, 120])
    ax[2].plot(np.arange(proto_plot.shape[0]) / fs, np.squeeze(feature), alpha = 0.5)
    ax[2].set_yticks([])
    ax[2].set_xticks([])
    plt.title('P{}==>>label{}'.format(p_num, target))
    plt.show()


def fft_visualize(ax, signal, fs):
    ## signal centralize
    signal = signal - signal.mean(0)
    length = signal.shape[0] // 2
    fft_data= 20*(np.log10(2*np.fft.fft(signal)/signal.shape[0]))[1:length]
    fft_x = [fs/signal.shape[0]*i for i in range(signal.shape[0])][1:length]
    ax.plot(fft_x, fft_data.real)
    return 


class PerturbationSignal(Dataset):

    def __init__(self, signal, length):
        self.signal = signal
        self.length = length
        self.features, self.targets = self._get_signal()

    def _get_signal(self):
        perturbation = np.zeros([self.length])
        perturbation_signal = []
        for i in range(self.signal.shape[1] - self.length):
            replace = np.copy(self.signal)
            if i ==0:
                pass
            else:
                replace[:, i-1 : i+self.length -1] = perturbation
            perturbation_signal.append(replace)
        perturbation_signal = np.array(perturbation_signal, dtype=np.float32)
        perturbation_trgs = np.ones([perturbation_signal.shape[0]])
        return perturbation_signal, perturbation_trgs
        
    def __len__(self):
        return self.signal.shape[1] - self.length

    def __getitem__(self, idx):
        inputs = self.features[idx]
        inputs = torch.Tensor(inputs).float()
        labels = self.targets[idx]
        labels = torch.from_numpy(np.array(labels))
        return inputs, labels


def show_prototype_result(fold):
    proto_result = np.load('/summer/ProtoPSleep/exp2/proto_result_{}.npy'.format(fold), allow_pickle=True).item()
    weight_result = np.load('/summer/ProtoPSleep/exp2/weight_result_{}.npy'.format(fold), allow_pickle=True).item()
    ## weight plot
    ave_weight_cls = weight_result['ave']
    min_weight_cls = weight_result['min']
    x = np.arange(10)
    total_width, n = 0.8, 5
    width = total_width / n
    x = x - (total_width - width) / 2
    show_weight_cls = ave_weight_cls
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    ax[0].bar(x, show_weight_cls[0], width = width, label = 'W')
    ax[0].bar(x+ width, show_weight_cls[1], width = width, label = 'N1')
    ax[0].bar(x+ 2 * width, show_weight_cls[2], width = width, label = 'N2')
    ax[0].bar(x+ 3 * width, show_weight_cls[3], width = width, label = 'N3')
    ax[0].bar(x+ 4 * width, show_weight_cls[4], width = width, label = 'R')
    ax[1].bar(x, min_weight_cls[0], width = width, label = 'W')
    ax[1].bar(x + width, min_weight_cls[1], width = width, label = 'N1')
    ax[1].bar(x +2 * width, min_weight_cls[2], width = width, label = 'N2')
    ax[1].bar(x +3 * width, min_weight_cls[3], width = width, label = 'N3')
    ax[1].bar(x +4 * width, min_weight_cls[4], width = width, label = 'R')
    plt.legend()
    plt.show()
    ## proto plot
    for j in range(10):
        for k in range(4):
            proto_wavelet = proto_result[str(j)]['wavelet'][k]
            signal = proto_result[str(j)]['signal'][k]
            targets = proto_result[str(j)]['targets'][k]
            proto_plot = proto_result[str(j)]['plot'][k]
            prototype_plot(proto_wavelet, proto_plot, signal, targets, j)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = ListedColormap(cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def plot_scatter_importance(fold):
    weight_result = np.load('/summer/ProtoPSleep/exp2/weight_result_{}.npy'.format(fold), allow_pickle=True).item()
    ## weight plot
    ave_weight_cls = np.array(list(weight_result['ave'].values()))
    min_weight_cls = np.array(list(weight_result['min'].values()))
    weight_sum = np.stack([ave_weight_cls, min_weight_cls], 0)
    weight_sum_select = [weight_sum[:, :, 0], weight_sum[:, :, 1],weight_sum[:, :, 2],weight_sum[:, :, 3],weight_sum[:, :, 4],weight_sum[:, :, 5],weight_sum[:, :, 6],weight_sum[:, :, 9]]   
    x1 = 0.5* np.array([0, 1, 2.2, 3.2, 4.4, 5.4, 6.6, 7.6, 8.8, 9.8])
    x = np.stack([x1, x1, x1, x1, x1, x1, x1, x1], 0)
    y = np.stack([14*np.ones_like(x1), 12*np.ones_like(x1), 10*np.ones_like(x1), 8*np.ones_like(x1), 6*np.ones_like(x1), 4*np.ones_like(x1), 2*np.ones_like(x1), 0*np.ones_like(x1)], 0)
    z = []
    for i in range(8):
        z.append([item for pair in zip (weight_sum_select[i][0], weight_sum_select[i][1]) for item in pair])
    z = np.array(z)
    norm_z = (z - z.min())
    plt.figure(figsize=(16, 12.8))
    cmap = plt.cm.OrRd
    new_cmap = truncate_colormap(cmap, 0.3, 1.0)
    plt.scatter(x.reshape(-1), y.reshape(-1), c=norm_z.reshape(-1), s = 3500* np.ones_like(z.reshape(-1)),cmap=new_cmap, vmin=0.0, vmax=7.0, alpha=0.5) 
    z = np.around(z, 2)
    z = np.maximum(z, 0.0)
    for i in range(x.reshape(-1).shape[0]):
        plt.text(x.reshape(-1)[i], y.reshape(-1)[i], z.reshape(-1)[i], fontsize=17, verticalalignment='center', horizontalalignment='center')
    char = plt.colorbar()
    char.ax.tick_params(labelsize=18)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def plot_err_diff_importance(fold, cls=2):
    weight_result = np.load('/summer/ProtoPSleep/exp2/weight_result_{}.npy'.format(fold), allow_pickle=True).item()
    idx = [0, 1, 2, 3, 4, 5, 6, 9]
    ave_weight_true = weight_result['ave'][cls][idx]
    min_weight_true = weight_result['min'][cls][idx]
    ave_weight_err = - weight_result['ave_err'][cls][idx]
    min_weight_err = - weight_result['min_err'][cls][idx]
    ave_weight_true[ave_weight_true < 0.01] = 0
    min_weight_true[min_weight_true < 0.01] = 0
    ave_weight_err[ave_weight_err > -0.01] = 0
    min_weight_err[min_weight_err > -0.01] = 0
    fig, ax = plt.subplots(figsize=(6, 5))
    index = np.arange(0, len(ave_weight_err))
    ax.barh(index, min_weight_true, height=.8, color = '#82B0D2', label = 'WE Correct', zorder=1)
    ax.barh(index, ave_weight_true, height=.8, color = '#8ECFC9', label = 'PE Correct', zorder=2)
    ax.barh(index, min_weight_err, height=.8, color = '#FA7F6F', label = 'WE Error', zorder=3)
    ax.barh(index, ave_weight_err, height=.8, color = '#FFBE7A', label = 'PE Error', zorder=4)
    ##添加竖线
    ax.axvline(x=0, color='#bfbbbb', lw=.8)
    y_label = ('W0', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7')
    y_pos = np.arange(len(y_label))
    ax.legend(frameon=False, loc='upper center', ncol=8, prop={'size':14}, bbox_to_anchor=(0.5, 1.10), borderaxespad=0.)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_label)
    ax.set_xticks((-20, -10, 0, 10, 20), ('20', '10', '0', '10', '20'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()


def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    warnings.filterwarnings("ignore", category=UserWarning) 

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')
    parser.add_argument('--config', type=str, help='config file path', default='/summer/ProtoPSleep/configs/SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2018_wavesensing.json')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open(args.config) as config_file:
        config = json.load(config_file)
    config['name'] = os.path.basename(args.config).replace('.json', '')
    
    config['mode'] = 'normal'

    ## Choose the model trained in which one fold to explain
    fold = 3
    evaluator = OneFoldEvaluator(args, fold, config)
    evaluator.prototype_perturbation(fold)
    

if __name__ == "__main__":
    main()