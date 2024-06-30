import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from loss_saver import LossSaver
import torch.nn.functional as F


class Trainer(object):
    def __init__(self, model, device, folder_name, model_name, editdist):
        super(Trainer, self).__init__()
        self.model = model.to(device)

        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.0001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=2, verbose=True)

        self.exp_folder = folder_name
        self.device = device
        self.model_name = model_name

        self.mode = editdist

# https://github.com/HarshSulakhe/siamesenetworks-pytorch/blob/master/loss.py
    def contrastive_loss(self, dists, labels):
        m = 1.0
        positive_loss = (labels) * torch.pow(dists, 2)
        negative_loss = (1 - labels) * torch.pow(torch.clamp(m - dists, min=0.0), 2)
        return torch.mean(positive_loss + negative_loss)


    def validate(self, batch):
        self.model.eval()
        
        imgs1, imgs2, labels, dist = batch

        imgs1 = imgs1.to(self.device)
        imgs2 = imgs2.to(self.device)
        labels = labels.to(self.device)
        dist = dist.to(self.device)

        imgs1 = torch.stack([imgs1, imgs1, imgs1], dim=1).squeeze(2)
        imgs2 = torch.stack([imgs2, imgs2, imgs2], dim=1)

        if not self.mode:
            feats1, feats2 = self.model(imgs1, imgs2)
            l2_dist = nn.functional.pairwise_distance(feats1, feats2)

            loss = self.contrastive_loss(l2_dist, labels)
        else:
            pred_edit = self.model(imgs1, imgs2)
            l2_dist = nn.functional.pairwise_distance(pred_edit, dist)
            loss = torch.mean(nn.functional.pairwise_distance(pred_edit, dist))

        mean_pos_dists = torch.mean(labels * l2_dist)
        mean_neg_dists = torch.mean((1. - labels) * l2_dist)

        return loss.detach().cpu(), imgs1.cpu(), imgs2.cpu(), labels.cpu(), mean_pos_dists.detach().cpu().numpy(), mean_neg_dists.detach().cpu().numpy()
    
    def train_batch(self, batch):
        self.model.train()
        imgs1, imgs2, labels, dist = batch

        imgs1 = imgs1.to(self.device)
        imgs2 = imgs2.to(self.device)
        labels = labels.to(self.device)
        dist = dist.to(self.device)

        imgs1 = torch.stack([imgs1, imgs1, imgs1], dim=1).squeeze(2)
        imgs2 = torch.stack([imgs2, imgs2, imgs2], dim=1)

        if not self.mode:
            feats1, feats2 = self.model(imgs1, imgs2)
            l2_dist = nn.functional.pairwise_distance(feats1, feats2)

            loss = self.contrastive_loss(l2_dist, labels)
        else:
            pred_edit = self.model(imgs1, imgs2)
            l2_dist = nn.functional.pairwise_distance(pred_edit, dist)
            loss = torch.mean(nn.functional.pairwise_distance(pred_edit, dist))

        loss.backward()

        total_norm = 0
        parameters = [p for p in self.model.parameters() if p.grad is not None and p.requires_grad]
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        mean_pos_dists = torch.mean(labels * l2_dist)
        mean_neg_dists = torch.mean((1. - labels) * l2_dist)

        return loss.detach().cpu(), total_norm, mean_pos_dists.detach().cpu(), mean_neg_dists.detach().cpu()


    def train_model(self, train_loader, valid_loader, epochs):
        if self.mode:
            train_saver = LossSaver(f"{self.exp_folder}-{self.model_name}-edit", "train", 16)
            valid_saver = LossSaver(f"{self.exp_folder}-{self.model_name}-edit", "valid", 16)
        else:
            train_saver = LossSaver(f"{self.exp_folder}-{self.model_name}", "train", 16)
            valid_saver = LossSaver(f"{self.exp_folder}-{self.model_name}", "valid", 16)
        s_epoch, end_epoch = epochs

        patience = 5
        pat_count = 0
        min_val_loss = 100000
        
        for epoch in range(s_epoch, end_epoch):
            torch.backends.cudnn.benchmark = True
            print(epoch)

            avg_loss = 0
            pos_dists = 0
            neg_dists = 0
            for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
                loss, norm, pos_dist, neg_dist = self.train_batch(batch)
                avg_loss += loss
                pos_dists += pos_dist
                neg_dists += neg_dist


            train_saver.save_to_csv(epoch, avg_loss/len(train_loader), pos_dists/len(train_loader), neg_dists/len(train_loader))
            dir = f"./models/{self.exp_folder}/"
            os.makedirs(dir, exist_ok=True)
            torch.save(self.model.state_dict(), f"{dir}{self.model_name}-{epoch}.model")
            print(f"mean train loss epoch {epoch}: {avg_loss/len(train_loader)} mean l2-distance pos: {pos_dists/len(train_loader)} mean l2-distance neg: {neg_dists/len(train_loader)} norm: {norm}")

            avg_loss = 0
            pos_dists_ = []
            neg_dists_ = []
            for idx, batch in tqdm(enumerate(valid_loader)):
                loss, img1, img2, labels, pos_dist, neg_dist = self.validate(batch)
                avg_loss += loss
                pos_dists_.append(pos_dist)
                neg_dists_.append(neg_dist)

            valid_saver.save_to_csv(epoch, avg_loss/len(valid_loader), np.mean(pos_dists_), np.mean(neg_dists_))
            print(f"mean validation loss epoch {epoch}: {avg_loss/len(valid_loader)} mean l2-distance pos: {np.mean(pos_dists_)} mean l2-distance neg: {np.mean(neg_dists_)}")
            if self.scheduler is not None:
                self.scheduler.step(avg_loss/len(valid_loader))
            
            if (avg_loss / len(valid_loader)) < min_val_loss:
                min_val_loss = avg_loss / len(valid_loader)
                pat_count = 0
            else:
                pat_count += 1
            
            if pat_count == patience:
                break