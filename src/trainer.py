import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import editdistance
from network.losses import Loss

class HTRtrainer(object):
    def __init__(self, model, optimizer, device, tokenizer, loss_name, self_supervised):
        super(HTRtrainer, self).__init__()
        self.htr_model = model.to(device)
        self.optimizer = optimizer

        self.loss = Loss(loss_name, self_supervised, tokenizer)
        
        self.device = device
        self.max_text_length = tokenizer.maxlen
        self.tokenizer = tokenizer

        


    def evaluate(self, y_pred, gt_labels):
        gt_labels = [self.tokenizer.decode(label) for label in gt_labels]
        y_pred = [self.tokenizer.decode(label) for label in y_pred]
        print(y_pred)
        cer = 0
        wer = 0

        for (pd, gt) in zip(y_pred, gt_labels):
            pd_cer, gt_cer = list(pd), list(gt)
            dist = editdistance.eval(pd_cer, gt_cer)
            cer += dist / (max(len(pd_cer), len(gt_cer)))

            pd_wer, gt_wer = pd.split(), gt.split()
            dist = editdistance.eval(pd_wer, gt_wer)
            wer += dist / (max(len(pd_wer), len(gt_wer)))
        return cer/ len(gt_labels), wer/ len(gt_labels)
    
    def validate(self, batch):
        self.htr_model.eval()
        imgs, gt_labels = batch

        imgs = imgs.to(self.device)
        gt_labels = gt_labels.to(self.device)

        y_pred = self.htr_model(imgs)
        loss = self.loss.loss_func(y_pred, gt_labels)

        y_pred_soft = torch.nn.functional.softmax(y_pred, dim=2).detach()
        y_pred_max = torch.max(y_pred_soft, dim=2).indices.cpu().numpy()
        gt_labels = gt_labels.detach().cpu().numpy()
        cer, wer = self.evaluate(y_pred_max, gt_labels)

        return loss, cer, wer


    
    def train_batch(self, batch):
        self.htr_model.train()
        imgs, gt_labels = batch

        imgs = imgs.to(self.device)
        gt_labels = gt_labels.to(self.device)

        self.optimizer.zero_grad()

        y_pred = self.htr_model(imgs)
        loss = self.loss.loss_func(y_pred, gt_labels)

        

        loss.backward()
         # Print gradients
        total_norm = 0
        parameters = [p for p in self.htr_model.parameters() if p.grad is not None and p.requires_grad]
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        print(total_norm)
        torch.nn.utils.clip_grad_norm_(self.htr_model.parameters(), 1.0)
        self.optimizer.step()

        y_pred_soft = torch.nn.functional.softmax(y_pred, dim=2).detach()
        y_pred_max = torch.max(y_pred_soft, dim=2).indices.cpu().numpy()
        gt_labels = gt_labels.detach().cpu().numpy()
        cer, wer = self.evaluate(y_pred_max, gt_labels)

        return loss, cer, wer, total_norm



    def train_model(self, train_loader, valid_loader, epochs):
        s_epoch, end_epoch = epochs
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(s_epoch, end_epoch):
            print(epoch)
            avg_loss = 0
            avg_cer = 0
            avg_wer = 0

            for batch in tqdm(train_loader):
                loss, cer, wer, norm = self.train_batch(batch)
                # print(loss)
                avg_loss += loss
                avg_cer += cer
                avg_wer += wer

            print(f"mean train loss epoch {epoch}: {avg_loss/len(train_loader)}, cer: {avg_cer/len(train_loader)}, wer: {avg_wer/len(train_loader)}, {norm}")
            avg_loss = 0
            avg_cer = 0
            avg_wer = 0

            for batch in tqdm(valid_loader):
                loss, cer, wer = self.train_batch(batch)
                # print(loss)
                avg_loss += loss
                avg_cer += cer
                avg_wer += wer

            print(f"mean validation loss epoch {epoch}: {avg_loss/len(valid_loader)}, cer: {avg_cer/len(valid_loader)}, wer: {avg_wer/len(valid_loader)}")

            





        