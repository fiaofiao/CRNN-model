import imp
from operator import mod
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
from torch import nn
from torch.optim import lr_scheduler
# ignored_params = list(map(id, model.fc.parameters())) # 返回的是parameters的 内存地址
# base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
# optimizer = torch.optim.AdamW([{'params': base_params, 'lr': 1e-4}, {'params':model.fc.parameters(), 'lr': 1e-3}])
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, verbose=True)
# optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
class Options:
    def __init__(self, message) -> None:
        self.save_path = 'results'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.exp_num = len(os.listdir(self.save_path))
        self.ckpts = os.path.join(self.save_path, 'exp' + str(self.exp_num), 'ckpts')
        if not os.path.exists(self.ckpts):
            os.makedirs(self.ckpts)
        self.ckpts = os.path.join(self.ckpts, 'scale_model.pth')
        self.performance_path = os.path.join(self.save_path, 'exp' + str(self.exp_num), 'performance')
        if not os.path.exists(self.performance_path):
            os.makedirs(self.performance_path)
        
        with open(os.path.join(self.save_path, 'exp' + str(self.exp_num), 'message.txt'), 'w') as f:
            f.write(message)


class Trainer:
    def __init__(self, model, criterion, optimizer, lr, epochs, train_loader, val_loader, test_loader=None, scheduel=None, opts=None) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduel = scheduel
        self.lr = lr
        self.epochs = epochs
        self.opts = opts

    def train(self):
        bestauc = 0 
        bestloss = 100  
        for epoch in range(self.epochs):
            self.model.train()
            for tensors in self.train_loader:
                tensors = [t.to('cuda') for t in tensors]
                y = tensors[-1]
                logits = self.model(*tensors[:-1])
                loss = self.criterion(logits, y)
                # print(loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            
            if epoch % 1 == 0:
                valauc, labels, preds, loss = self.evaluate(self.model, self.val_loader)
                print('epoch:', epoch)
                if valauc > bestauc:
                    bestauc = valauc
                    print('val loss is', loss, ', update params, present auc is', valauc)
                    torch.save(self.model.state_dict(), self.opts.ckpts)
                    print()
                    data = zip(labels, preds)
                    dataframe = pd.DataFrame(data, columns=['labels', 'preds'])
                    dataframe.to_csv(os.path.join(self.opts.performance_path, 'val.csv'))

                if valauc > bestauc or epoch % 10 == 0:
                    print('val loss is', loss, ', val auc is', valauc)
                    auc, labels, preds, loss = self.evaluate(self.model, self.train_loader)
                    print('train loss is', loss, ', train auc is', auc)
                    data = zip(labels, preds)
                    dataframe = pd.DataFrame(data, columns=['labels', 'preds'])
                    dataframe.to_csv(os.path.join(self.opts.performance_path, 'train.csv'))

                    if self.test_loader:
                        auc, labels, preds, loss = self.evaluate(self.model, self.test_loader)
                        print('test loss is', loss, ', test auc is', auc)
                        if auc > 0.8:
                            data = zip(labels, preds)
                            dataframe = pd.DataFrame(data, columns=['labels', 'preds'])
                            dataframe.to_csv(os.path.join(self.opts.performance_path, 'test.csv'))
                    
            
            if self.scheduel:
                self.scheduel.step()
        
    def evaluate(self, model, loader):
        model.eval()
        preds = []
        labels = []
        losses = []
        for tensors in loader:
            tensors = [t.to('cuda') for t in tensors]
            y = tensors[-1]
            
            with torch.no_grad():
                logits = self.model(*tensors[:-1])
                loss = self.criterion(logits, y)
                losses.append(loss.item())
                output = torch.nn.functional.softmax(logits, dim=1)
                pred = output[:, -1]
                preds = np.concatenate((preds, pred.cpu().numpy()), axis=0)
                labels = np.concatenate((labels, y.cpu().numpy()), axis=0)

            
        return roc_auc_score(labels, preds), labels, preds, np.mean(losses)
    
    @staticmethod
    def static_evaluate(model, loader):
        model.eval()
        preds = []
        labels = []
        for tensors in loader:
            tensors = [t.to('cuda') for t in tensors]
            
            with torch.no_grad():
                output = torch.nn.functional.softmax(model(*tensors[:-1]), dim=1)
                pred = output[:, -1]
                preds = np.concatenate((preds, pred.cpu().numpy()), axis=0)
                labels = np.concatenate((labels, tensors[-1].cpu().numpy()), axis=0)

            
        return roc_auc_score(labels, preds), labels, preds



