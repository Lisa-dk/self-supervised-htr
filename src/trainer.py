import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import editdistance
from network.losses import Loss
import matplotlib.pyplot as plt
import os


class HTRtrainer(object):
    def __init__(self, model, optimizer, lr_scheduler, device, tokenizer, loss_name, self_supervised):
        super(HTRtrainer, self).__init__()
        self.htr_model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = lr_scheduler

        self.loss = Loss(loss_name, self_supervised, tokenizer, device)
        
        self.device = device
        self.max_text_length = tokenizer.maxlen
        self.tokenizer = tokenizer

    def save_images(self, epoch, pred_imgs, true_imgs, pred_labels, labels, plot=False):
        dir_path = './results/imgs/'
        os.makedirs(dir_path, exist_ok=True)

        for idx in range(len(true_imgs)):
            f, axarr = plt.subplots(1,2)
            f.suptitle(str(labels[idx]))
            print(pred_imgs[idx].shape)
            print(pred_imgs[idx][0].shape)
            axarr[0].imshow(true_imgs[idx], cmap='gray')
            axarr[1].imshow(pred_imgs[idx][0], cmap='gray')
            axarr[0].set_title("synthetic ", pred_labels[idx])
            axarr[1].set_title("real")
            plt.savefig(dir_path +  str(labels[idx]) + 'images_at_epoch_{:04d}.png'.format(epoch))
            plt.show()
            plt.close()


        
    def evaluate(self, y_pred, gt_labels, show=False):
        gt_labels = [self.tokenizer.decode(label) for label in gt_labels]
        y_pred = [self.tokenizer.decode(label) for label in y_pred]
        
        cer = 0
        wer = 0

        for (pd, gt) in zip(y_pred, gt_labels):
            pd_cer, gt_cer = list(pd), list(gt)
            dist = editdistance.eval(pd_cer, gt_cer)
            cer += dist / (max(len(pd_cer), len(gt_cer)))

            pd_wer, gt_wer = pd.split(), gt.split()
            dist = editdistance.eval(pd_wer, gt_wer)
            wer += dist / (max(len(pd_wer), len(gt_wer)))
            if show: 
                return cer/ len(gt_labels), wer/ len(gt_labels), y_pred, gt_labels
        return cer/ len(gt_labels), wer/ len(gt_labels)
    
    def validate(self, batch):
        self.htr_model.eval()
        with torch.no_grad():
            imgs, gt_labels = batch

            imgs = imgs.to(self.device)
            gt_labels = gt_labels.to(self.device)

            y_pred = self.htr_model(imgs[:,0,:,:].unsqueeze(1))
            loss = self.loss.loss_func(y_pred, imgs)

            y_pred_soft = torch.nn.functional.softmax(y_pred, dim=2).detach()
            y_pred_max = torch.max(y_pred_soft, dim=2).indices
            gt_labels = gt_labels.detach()
            cer, wer, y_pred, y_true = self.evaluate(y_pred_max, gt_labels, show=True)

        return loss, cer, wer, y_pred, y_true, #imgs[:,0,:,:], synth_imgs
    
    def train_batch(self, batch):
        self.htr_model.train()
        imgs, gt_labels = batch

        imgs = imgs.to(self.device)
        gt_labels = gt_labels.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)

        y_pred = self.htr_model(imgs[:,0,:,:].unsqueeze(1))
        loss = self.loss.loss_func(y_pred, imgs)

        loss.backward()
         # Print gradients
        # total_norm = 0
        # parameters = [p for p in self.htr_model.parameters() if p.grad is not None and p.requires_grad]
        # for p in parameters:
        #     param_norm = p.grad.detach().data.norm(2)
        #     total_norm += param_norm.item() ** 2
        # total_norm = total_norm ** 0.5

        # print(total_norm)

        torch.nn.utils.clip_grad_norm_(self.htr_model.parameters(), 1.0)
        self.optimizer.step()

        y_pred_soft = torch.nn.functional.softmax(y_pred, dim=2).detach()
        y_pred_max = torch.max(y_pred_soft, dim=2).indices
        gt_labels = gt_labels.detach()
        cer, wer = self.evaluate(y_pred_max, gt_labels)

        return loss, cer, wer, #total_norm



    def train_model(self, train_loader, valid_loader, epochs):
        s_epoch, end_epoch = epochs
        # torch.autograd.set_detect_anomaly(True)
        for epoch in range(s_epoch, end_epoch):
            torch.backends.cudnn.benchmark = True
            print(epoch)
            avg_loss = 0
            avg_cer = 0
            avg_wer = 0

            for batch in tqdm(train_loader):
                loss, cer, wer = self.train_batch(batch)
                # print(loss)
                avg_loss += loss
                avg_cer += cer
                avg_wer += wer
            self.scheduler.step()

            torch.save(self.htr_model.state_dict(), "./htr_model.model")
            print(f"mean train loss epoch {epoch}: {avg_loss/len(train_loader)}, cer: {avg_cer/len(train_loader)}, wer: {avg_wer/len(train_loader)}")
            avg_loss = 0
            avg_cer = 0
            avg_wer = 0

            for idx, batch in tqdm(enumerate(valid_loader)):
                loss, cer, wer, y_pred, y_true = self.validate(batch)
                # print(loss)
                avg_loss += loss
                avg_cer += cer
                avg_wer += wer

            print(f"mean validation loss epoch {epoch}: {avg_loss/len(valid_loader)}, cer: {avg_cer/len(valid_loader)}, wer: {avg_wer/len(valid_loader)}")
            # self.save_images(4, syn_imgs.cpu().numpy(), imgs.cpu().numpy(), y_pred, y_true, plot=True)
            print("predictions last batch: ")
            for idx in range(len(y_pred)):
                print(f"gt: {y_true[idx]}, pred: {y_pred[idx]}")

            





        