import os, sys
sys.path.append('/summer/WaveSleepNet/')
import json
import argparse
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import *
from loader import EEGDataLoader
from models.protop import ProtoPNet      


class OneFoldTrainer:
    def __init__(self, args, fold, config):
        self.args = args
        self.fold = fold
        
        self.cfg = config
        self.ds_cfg = config['dataset']
        self.fp_cfg = config['feature_pyramid']
        self.tp_cfg = config['training_params']
        self.es_cfg = self.tp_cfg['early_stopping']
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('[INFO] Config name: {}'.format(config['name']))

        self.train_iter = 0
        self.model = self.build_model()
        self.loader_dict = self.build_dataloader()
        
        self.criterion = nn.CrossEntropyLoss()
        self.activate_train_mode()
        self.optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=self.tp_cfg['lr'], weight_decay=self.tp_cfg['weight_decay'])
        
        self.ckpt_path = os.path.join('checkpoints', config['name']+'_'+str(args.seed))
        self.ckpt_name = 'ckpt_fold-{0:02d}.pth'.format(self.fold)
        self.early_stopping = EarlyStopping(patience=self.es_cfg['patience'], verbose=True, ckpt_path=self.ckpt_path, ckpt_name=self.ckpt_name, mode=self.es_cfg['mode'])
        

    def build_model(self):
        model = ProtoPNet(self.cfg) 
        print('[INFO] Number of params of model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
        model = torch.nn.DataParallel(model, device_ids=list(range(len(self.args.gpu.split(",")))))
        if self.tp_cfg['mode'] != 'scratch':
            print('[INFO] Model loaded for finetune')
            load_name = self.cfg['name'].replace('SL-{:02d}'.format(self.ds_cfg['seq_len']), 'SL-01')
            load_name = load_name.replace('numScales-{}'.format(self.fp_cfg['num_scales']), 'numScales-1')
            load_name = load_name.replace(self.tp_cfg['mode'], 'pretrain')
            load_path = os.path.join('checkpoints', load_name, 'ckpt_fold-{0:02d}.pth'.format(self.fold))
        model.to(self.device)
        print('[INFO] Model prepared, Device used: {} GPU:{}'.format(self.device, self.args.gpu))

        return model
    
    def build_dataloader(self):
        train_dataset = EEGDataLoader(self.cfg, self.fold, set='train')
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.tp_cfg['batch_size'], shuffle=True, num_workers=4*len(self.args.gpu.split(",")), pin_memory=True)
        val_dataset = EEGDataLoader(self.cfg, self.fold, set='val')
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.tp_cfg['batch_size'], shuffle=False, num_workers=4*len(self.args.gpu.split(",")), pin_memory=True)
        test_dataset = EEGDataLoader(self.cfg, self.fold, set='test')
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.tp_cfg['batch_size'], shuffle=False, num_workers=4*len(self.args.gpu.split(",")), pin_memory=True)
        print('[INFO] Dataloader prepared')

        return {'train': train_loader, 'val': val_loader, 'test': test_loader}
    
    def activate_train_mode(self):
        self.model.train()

    def protop_loss(self, outputs, labels):
        self.loss_ensemble = {}
        loss_cfg = self.cfg['classifier']
        ## cross_entropy loss
        cross_entropy = self.criterion(outputs, labels)
        self.loss_ensemble['cross_entropy'] = loss_cfg['class_lambda']* cross_entropy
        ## distance loss
        min_dist = self.model.module.min_distance   ##[?, P]
        dist_loss = torch.mean(torch.min(min_dist, 1).values)
        self.loss_ensemble['dist_loss'] = loss_cfg['dist_lambda']* dist_loss
        ## pd loss
        prototype = self.model.module.prototype_vectors
        diversity = self._diversity_cal(prototype)
        pd_loss = 1 / (torch.log(diversity) + 1e-4)
        self.loss_ensemble['pd_loss'] = loss_cfg['pd_lambda']* pd_loss
        ## identity loss
        identity_loss = torch.mean(torch.min(min_dist, 0).values)
        self.loss_ensemble['identity_loss'] = loss_cfg['identity_lambda']* identity_loss
        ## weight loss
        fc_weight = self.model.module.fc.weight.view(-1)
        weight_loss = torch.sum(torch.abs(fc_weight)) 
        self.loss_ensemble['weight_loss'] = loss_cfg['weight_lambda']* weight_loss
        return (self.loss_ensemble['cross_entropy'] + self.loss_ensemble['dist_loss'] + 
                self.loss_ensemble['pd_loss'] + self.loss_ensemble['identity_loss'] + self.loss_ensemble['weight_loss']) 


    def _diversity_cal(self, X):
        def list_of_distance(x, y):
            '''
            Given a list of vectors, X = [x1, ..., xn], Y = [y1, ..., ym],
            Return a list of vectors, 
                [[d(x1, y1), d(x1, y2), ..., d(x1, ym)],
                ...
                [d(xn, y1), d(xn, y2), ..., d(xn, ym)]]
            '''
            XX = list_of_norms(x).view(-1, 1)
            YY = list_of_norms(y).view(1, -1)
            output = XX + YY - 2 * torch.matmul(x, y.transpose(0, 1))
            return output

        def list_of_norms(x):
            return torch.sum(torch.pow(x, 2), 1)
            
        ## x shape [P, C, 1]
        # diversity = []
        diversity = 0
        weight = X
        num = self.cfg['classifier']['prototype_num']
        for i in range(num):
            if i >= num - 1:
                pass
            else:
                signal1 = weight[i].view(-1, 1).transpose(0, 1) 
                signal2 = weight[i+1:]
                signal2 = torch.squeeze(signal2, dim=2)
                ## shape [1, M-i]
                distance = list_of_distance(signal1, signal2)
                diversity += torch.min(distance, 1).values
        return diversity / (num -1)
                
    def train_one_epoch(self, epoch):
        correct, total, train_loss = 0, 0, 0

        for i, (inputs, labels) in enumerate(self.loader_dict['train']):
            loss = 0
            total += labels.size(0)
            inputs = inputs.to(self.device)
            labels = labels.view(-1).to(self.device)
            outputs = self.model(inputs)
            outputs_sum = torch.zeros_like(outputs[0])

            if self.cfg['backbone']['name'] != 'ProtoP':
                for j in range(len(outputs)):
                    loss += self.protop_loss(outputs[j], labels)
                    outputs_sum += outputs[j]
            else:
                loss = self.protop_loss(outputs, labels) 
                outputs_sum = outputs

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            predicted = torch.argmax(outputs_sum, 1)
            correct += predicted.eq(labels).sum().item()
            self.train_iter += 1

            progress_bar(i, len(self.loader_dict['train']), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | cls_loss: %.3f |dist_loss: %.3f |pd_loss: %.3f |identity_loss: %.3f |weight_loss: %.3f '
                    % (train_loss / (i + 1), 100. * correct / total, correct, total, self.loss_ensemble['cross_entropy'], 
                        self.loss_ensemble['dist_loss'], self.loss_ensemble['pd_loss'], self.loss_ensemble['identity_loss'], 
                        self.loss_ensemble['weight_loss']))
            
            if self.train_iter % self.tp_cfg['val_period'] == 0:
                print('')
                val_acc, val_loss, val_mf1 = self.evaluate(mode='val')
                self.early_stopping(val_mf1, val_loss, self.model)
                if self.cfg['backbone']['name'] != 'ProtoP':
                    self.activate_train_mode()
                if self.early_stopping.early_stop:
                    break
            
    @torch.no_grad()
    def evaluate(self, mode):
        self.model.eval()
        correct, total, eval_loss = 0, 0, 0
        y_true = np.zeros(0)
        y_pred = np.zeros((0, self.cfg['classifier']['num_classes']))

        for i, (inputs, labels) in enumerate(self.loader_dict[mode]):
            loss = 0
            total += labels.size(0)
            inputs = inputs.to(self.device)
            labels = labels.view(-1).to(self.device)

            outputs = self.model(inputs)
            outputs_sum = torch.zeros_like(outputs[0])
            if self.cfg['backbone']['name'] != 'ProtoP':
                for j in range(len(outputs)):
                    loss += self.protop_loss(outputs[j], labels)
                    outputs_sum += outputs[j]
            else:
                loss = self.protop_loss(outputs, labels) 
                outputs_sum = outputs

            eval_loss += loss.item()
            predicted = torch.argmax(outputs_sum, 1)
            correct += predicted.eq(labels).sum().item()
            y_true = np.concatenate([y_true, labels.cpu().numpy()])
            y_pred = np.concatenate([y_pred, outputs_sum.cpu().numpy()])

            y_pred_argmax = np.argmax(y_pred, 1)
            result_dict = skmet.classification_report(y_true, y_pred_argmax, digits=3, output_dict=True)
            mf1 = round(result_dict['macro avg']['f1-score']*100, 1)

            progress_bar(i, len(self.loader_dict[mode]), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | MF1: %.3f'
                    % (eval_loss / (i + 1), 100. * correct / total, correct, total, mf1))

        if mode == 'val':
            return 100. * correct / total, eval_loss, mf1
        elif mode == 'test':
            return y_true, y_pred, mf1
        else:
            raise NotImplementedError
    
    def run(self):
        for epoch in range(self.tp_cfg['max_epochs']):
            print('\n[INFO] Fold: {}, Epoch: {}'.format(self.fold, epoch))
            self.train_one_epoch(epoch)
            if self.early_stopping.early_stop:
                break
        
        self.model.load_state_dict(torch.load(os.path.join(self.ckpt_path, self.ckpt_name)))
        y_true, y_pred, mf1 = self.evaluate(mode='test')
        print('')

        return y_true, y_pred

def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    warnings.filterwarnings("ignore", category=UserWarning) 

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=49, help='random seed')
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')
    parser.add_argument('--config', type=str, help='config file path', default='/summer/WaveSleepNet/configs/SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2018_wavesensing.json')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # For reproducibility
    set_random_seed(args.seed, use_cuda=True)

    with open(args.config) as config_file:
        config = json.load(config_file)
    config['name'] = os.path.basename(args.config).replace('.json', '')
    config['mode'] = 'normal'
    
    Y_true = np.zeros(0)
    Y_pred = np.zeros((0, config['classifier']['num_classes']))

    for fold in range(1, config['dataset']['num_splits'] + 1):
        trainer = OneFoldTrainer(args, fold, config)
        y_true, y_pred = trainer.run()
        Y_true = np.concatenate([Y_true, y_true])
        Y_pred = np.concatenate([Y_pred, y_pred])
    
        summarize_result(config, fold, Y_true, Y_pred)

    

if __name__ == "__main__":
    main()
