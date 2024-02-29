import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import editdistance
from network.losses import Loss
import matplotlib.pyplot as plt
import os
from loss_saver import LossSaver
from network.gen_model.gen_model import GenModel_FC
from torchaudio.models.decoder import ctc_decoder


class HTRtrainer(object):
    def __init__(self, model, optimizer, lr_scheduler, device, tokenizer, loss_name, self_supervised, folder_name):
        super(HTRtrainer, self).__init__()
        self.htr_model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = lr_scheduler

        self.loss_name = loss_name
        self.loss = Loss(loss_name, tokenizer, device)
        self.exp_folder = folder_name
        
        self.device = device
        self.max_text_length = tokenizer.maxlen
        self.tokenizer = tokenizer
        self.beam_search_decoder = ctc_decoder(lexicon=None,
            tokens=[char for char in tokenizer.chars + '-' + '|'],
            nbest=3,
            beam_size=1500
        )

        if self_supervised:
            self.gen_model = GenModel_FC(tokenizer.maxlen, tokenizer.vocab_size, tokenizer.PAD)
            self.gen_model.load_state_dict(torch.load('./network/gen_model/gen_model.model')) #load
            self.gen_model.eval()
            self.gen_model.to(self.device)
            self.mode = "self_supervised"
        else:
            self.mode = "supervised"
        
        self.train_batch = getattr(self, f"train_batch_{self.mode}")
        self.validate = getattr(self, f"validate_{self.mode}")

    def save_images(self, epoch, pred_imgs, true_imgs, pred_labels, labels, plot=False):
        dir_path = './results/imgs/'
        os.makedirs(dir_path, exist_ok=True)

        for idx in range(len(true_imgs)):
            f, axarr = plt.subplots(1,2)
            f.suptitle(str(labels[idx]))
            print(pred_imgs[idx].shape)
            print(pred_imgs[idx][0].shape)
            axarr[0].imshow(true_imgs[idx][0], cmap='gray')
            axarr[1].imshow(pred_imgs[idx][0], cmap='gray')
            print(pred_labels[idx])
            axarr[0].set_title("synthetic " + pred_labels[idx])
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
    
    def validate_self_supervised(self, batch):
        self.htr_model.eval()
        with torch.no_grad():
            imgs, gen_imgs, gt_labels = batch

            imgs = imgs.to(self.device)
            gen_imgs = gen_imgs.to(self.device).squeeze(2)
            gt_labels = gt_labels.to(self.device)

            y_pred = self.htr_model(imgs)
            y_pred = nn.functional.softmax(y_pred, 2)
            synth_imgs = self.gen_model(gen_imgs, y_pred)
            loss = self.loss.loss_func(synth_imgs, gen_imgs[:,0,:,:])

            y_pred_max = torch.max(y_pred, dim=2).indices.detach()
            gt_labels = gt_labels.detach()
            cer, wer, y_pred, y_true = self.evaluate(y_pred_max, gt_labels, show=True)

        return loss.detach().cpu(), cer, wer, y_pred, y_true, imgs, synth_imgs

    def validate_supervised(self, batch):
        self.htr_model.eval()
        with torch.no_grad():
            imgs, _, gt_labels = batch

            imgs = imgs.to(self.device)
            gt_labels = gt_labels.to(self.device)

            y_pred = self.htr_model(imgs)
            loss = self.loss.loss_func(y_pred, gt_labels)

            y_pred_soft = torch.nn.functional.softmax(y_pred, dim=2).detach()
            y_pred_max = torch.max(y_pred_soft, dim=2).indices
            gt_labels = gt_labels.detach()
            cer, wer, y_pred, y_true = self.evaluate(y_pred_max, gt_labels, show=True)

        return loss.detach().cpu(), cer, wer, y_pred, y_true, #imgs[:,0,:,:], synth_imgs
    
    def train_batch_self_supervised(self, batch):
        self.htr_model.train()
        imgs, gen_imgs, gt_labels = batch

        imgs = imgs.to(self.device)
        gen_imgs = gen_imgs.to(self.device).squeeze(2)
        gt_labels = gt_labels.to(self.device)

        self.optimizer.zero_grad()

        y_pred = self.htr_model(imgs)
        y_pred = nn.functional.softmax(y_pred, 2)
        synth_imgs = self.gen_model(gen_imgs, y_pred)
        loss = self.loss.loss_func(synth_imgs, gen_imgs[:,0,:,:])

        loss.backward()

        total_norm = 0
        parameters = [p for p in self.htr_model.parameters() if p.grad is not None and p.requires_grad]
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        torch.nn.utils.clip_grad_norm_(self.htr_model.parameters(), 1.0)
        self.optimizer.step()

        # beam_search_transcript = self.beam_search_decoder(y_pred.detach().cpu())[:][0]
        # print(beam_search_transcript)
        y_pred_max = torch.max(y_pred, dim=2).indices
        gt_labels = gt_labels.detach()
        cer, wer = self.evaluate(y_pred_max, gt_labels)

        return loss.detach().cpu(), cer, wer, total_norm
    
    def train_batch_supervised(self, batch):
        self.htr_model.train()
        imgs, _, gt_labels = batch

        imgs = imgs.to(self.device)
        gt_labels = gt_labels.to(self.device)

        self.optimizer.zero_grad()

        y_pred = self.htr_model(imgs)
        loss = self.loss.loss_func(y_pred, gt_labels)

        loss.backward()

        total_norm = 0
        parameters = [p for p in self.htr_model.parameters() if p.grad is not None and p.requires_grad]
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        torch.nn.utils.clip_grad_norm_(self.htr_model.parameters(), 1.0)
        self.optimizer.step()

        y_pred_soft = torch.nn.functional.softmax(y_pred, dim=2).detach()
        y_pred_max = torch.max(y_pred_soft, dim=2).indices
        gt_labels = gt_labels.detach()
        cer, wer = self.evaluate(y_pred_max, gt_labels)

        return loss.detach().cpu(), cer, wer, total_norm


    def train_model(self, train_loader, valid_loader, epochs):
        train_saver = LossSaver(f"{self.exp_folder}", "train", 16)
        valid_saver = LossSaver(f"{self.exp_folder}", "valid", 16)
        s_epoch, end_epoch = epochs
        torch.autograd.set_detect_anomaly(True)
        print(self.loss_name)
        
        for epoch in range(s_epoch, end_epoch):
            torch.backends.cudnn.benchmark = True
            print(epoch)
            avg_loss = 0
            avg_cer = 0
            avg_wer = 0
            if self.scheduler is not None:
                print("learning rate: ", self.scheduler.get_last_lr())

            # for idx, batch in tqdm(enumerate(train_loader)):
            #     loss, cer, wer, norm = self.train_batch(batch)
            #     # print(loss)
            #     avg_loss += loss
            #     avg_cer += cer
            #     avg_wer += wer

            # if self.scheduler is not None:
            #     self.scheduler.step()

            # n_train_batches = len(train_loader)
            # train_saver.save_to_csv(epoch, avg_loss/n_train_batches, avg_cer/n_train_batches, avg_wer/n_train_batches)
            # dir = f"./htr_models/{self.exp_folder}/"
            # os.makedirs(dir, exist_ok=True)
            # torch.save(self.htr_model.state_dict(), f"{dir}htr_model_{self.mode}-{epoch}.model")
            # print(f"mean train loss epoch {epoch}: {avg_loss/len(train_loader)}, cer: {avg_cer/len(train_loader)}, wer: {avg_wer/len(train_loader)} last norm: {norm}")
            # avg_loss = 0
            # avg_cer = 0
            # avg_wer = 0

            for idx, batch in tqdm(enumerate(valid_loader)):
                loss, cer, wer, y_pred, y_true = self.validate(batch)
                # print(loss)
                avg_loss += loss
                avg_cer += cer
                avg_wer += wer

            n_valid_batches = len(valid_loader)
            valid_saver.save_to_csv(epoch, avg_loss/n_valid_batches, avg_cer/n_valid_batches, avg_wer/n_valid_batches)
            print(f"mean validation loss epoch {epoch}: {avg_loss/len(valid_loader)}, cer: {avg_cer/len(valid_loader)}, wer: {avg_wer/len(valid_loader)}")
            # self.save_images(8, syn_imgs.cpu().numpy(), imgs.cpu().numpy(), y_pred, y_true, plot=True)
            print("predictions last batch: ")
            for idx in range(len(y_pred)):
                print(f"gt: {y_true[idx]}, pred: {y_pred[idx]}")
            exit()
