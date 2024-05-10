import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import editdistance, random
from network.losses import Loss
import matplotlib.pyplot as plt
import os
from loss_saver import LossSaver
from network.gen_model.gen_model import GenModel_FC
import kornia, string
import matplotlib.pyplot as plt
from data.tokenizer import Tokenizer


class HTRtrainer(object):
    def __init__(self, model, optimizer, lr_scheduler, device, tokenizer, loss_name, self_supervised, folder_name, vgg_layer):
        super(HTRtrainer, self).__init__()
        self.htr_model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = lr_scheduler

        self.loss_name = loss_name
        self.loss = Loss(loss_name, tokenizer, device, vgg_layer)
        self.exp_folder = folder_name
        
        self.device = device
        self.max_text_length = tokenizer.maxlen
        self.tokenizer = tokenizer

        if self_supervised:
            self.gen_model = GenModel_FC(tokenizer.maxlen, tokenizer.vocab_size, tokenizer.PAD)
            # self.gen_model.load_state_dict(torch.load('./network/gen_model/gen_model-15all.model')) #load
            self.gen_model.load_state_dict(torch.load('./network/gen_model/gen_model-25half.model'))
            self.gen_model.eval()

            for param in self.gen_model.parameters():
                param.requires_grad = False

            self.gen_model.to(self.device)
            self.mode = "self_supervised"
        else:
            self.mode = "supervised"
        
        self.train_batch = getattr(self, f"train_batch_{self.mode}")
        self.validate = getattr(self, f"validate_{self.mode}")

    def save_images(self, epoch, pred_imgs, true_imgs, pred_labels, labels, plot=False):
        dir_path = './results/imgs/' + self.exp_folder + '/'
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
            imgs, gen_imgs, gt_labels, _ = batch

            imgs = imgs.to(self.device)
            gen_imgs = gen_imgs.to(self.device).squeeze(2)
            gt_labels = gt_labels.to(self.device)

            y_pred = self.htr_model(imgs)
            y_pred = nn.functional.softmax(y_pred, 2)
            # y_pred = y_pred[:,:,:-1]
            # y_pred_max, _ = torch.max(y_pred, dim=2, keepdim=True)
            # y_pred_max = y_pred / y_pred_max

            # mask = (y_pred_max >= 1.0).float()
            # y_pred_max =y_pred_max * mask

            gt_labels_1hot = gt_labels.long()
            gt_labels_1hot = torch.nn.functional.one_hot(gt_labels_1hot, 56).float()

            synth_imgs = self.gen_model(gen_imgs, y_pred)

            if self.loss_name == "htr":
                loss, gt_htr, synth_htr = self.loss.loss_func(synth_imgs, gen_imgs[:,0,:,:])
            else:
                loss = self.loss.loss_func(synth_imgs, gen_imgs[:,0,:,:])

            y_pred_max = torch.max(y_pred, dim=2).indices.detach()
            gt_labels = gt_labels.detach()
            cer, wer, y_pred, y_true = self.evaluate(y_pred_max, gt_labels, show=True)

        if self.loss_name == "htr":
            return loss.detach().cpu(), cer, wer, y_pred, y_true, imgs, synth_imgs, \
            torch.max(gt_htr.detach(), dim=2, keepdim=True).indices, torch.max(synth_htr.detach(), dim=2, keepdim=True).indices
        else:
            return loss.detach().cpu(), cer, wer, y_pred, y_true, imgs, synth_imgs

    def validate_supervised(self, batch):
        self.htr_model.eval()
        with torch.no_grad():
            imgs, _, gt_labels, _, _ = batch

            imgs = imgs.to(self.device)
            gt_labels = gt_labels.to(self.device)

            y_pred = self.htr_model(imgs)
            loss = self.loss.loss_func(y_pred, gt_labels)

            y_pred_soft = torch.nn.functional.softmax(y_pred[:,:,:], dim=2).detach()
            y_pred_max = torch.max(y_pred_soft, dim=2).indices
            gt_labels = gt_labels.detach()
            cer, wer, y_pred, y_true = self.evaluate(y_pred_max, gt_labels, show=True)

        return loss.detach().cpu(), cer, wer, y_pred, y_true #imgs[:,0,:,:], synth_imgs
    
    def train_batch_self_supervised(self, batch, epoch):
        self.htr_model.train()
        imgs, gen_imgs, gt_labels, _ = batch
        gt_labels_1hot = gt_labels.long()
        gt_labels_1hot = torch.nn.functional.one_hot(gt_labels_1hot, 56).float()

        imgs = imgs.to(self.device)
        gen_imgs = gen_imgs.to(self.device).squeeze(2)
        gt_labels = gt_labels.to(self.device)
        gt_labels_1hot = gt_labels_1hot.to(self.device)

        self.optimizer.zero_grad()

        y_pred = self.htr_model(imgs)
        y_pred_soft = nn.functional.softmax(y_pred, 2)

        # y_htr_pred = self.loss.iam_model(imgs)
        # y_htr_pred = nn.functional.softmax(y_htr_pred, 2)
        # y_pred_max = torch.max(y_htr_pred, dim=2).indices
        # print(y_pred_max[0])
        # print(y_pred_max[1])

        # charset_base = string.ascii_lowercase + string.ascii_uppercase
        # tokenizer = Tokenizer(chars=charset_base, max_text_length=25, self_supervised=False)

        # y_pred_labels = np.asarray([tokenizer.decode(label) for label in y_pred_max])
        # y_pred_labels_enc = np.asarray([tokenizer.encode(lab) for lab in y_pred_labels])
        # y_pred_labels_1hot = torch.from_numpy(y_pred_labels_enc).long()
        # y_pred_labels_1hot = torch.nn.functional.one_hot(y_pred_labels_1hot, 56).float()
        
        # y_pred_max, _ = torch.max(y_pred, dim=2, keepdim=True)
        # y_pred_max = y_pred / y_pred_max

        # mask = (y_pred_max >= 1.0).float()
        # y_pred_max =y_pred_max * mask
        # y_pred = y_pred[:,:,:-1]

        synth_imgs = self.gen_model(gen_imgs, y_pred_soft)

        # y_htr_pred_synth = self.loss.iam_model(synth_imgs)

        # y_htr_pred_synth = nn.functional.softmax(y_htr_pred_synth, 2)
        # y_htr_pred_synth_max = torch.max(y_htr_pred_synth, dim=2).indices

        # y_htr_pred_synth_max = y_htr_pred_synth_max.cpu().numpy()
        # y_pred_max = y_pred_max.cpu().numpy()
        # y_htr_pred_synth_labels = [tokenizer.decode(label) for label in y_htr_pred_synth_max]

        # synth_imgs = synth_imgs.detach().cpu()

        # for idx in range(len(y_htr_pred_synth_max)):
        #     if not np.array_equal(y_htr_pred_synth_max[idx],y_pred_max[idx]):
        #         print("not same")
        #         print(y_pred_max[idx])
        #         print(y_htr_pred_synth_max[idx])
        #         print(y_pred_labels[idx], y_htr_pred_synth_labels[idx])
                
        #         plt.imshow(synth_imgs[idx][0])
        #         plt.show()
        #     else:
        #         print("same")

        # synth_imgs = synth_imgs.detach().cpu()

        # y_htr_pred = y_htr_pred.detach().cpu()

    
        # y_pred = [tokenizer.decode(label) for label in y_pred_max]
        # gt_labels = gt_labels.detach().cpu().numpy()
        # gt_labels = [tokenizer.decode(label) for label in gt_labels]
        # print(y_pred[0], gt_labels[0])
        if random.random() <= 1./(epoch + 1):
            gt_1hot_flat = gt_labels_1hot.view(-1, self.tokenizer.vocab_size)
            y_pred_flat = y_pred.view(-1, self.tokenizer.vocab_size)
            loss = nn.functional.cross_entropy(y_pred_flat, gt_1hot_flat)
        elif self.loss_name == "htr":
                loss, _, _ = self.loss.loss_func(synth_imgs, gen_imgs[:,0,:,:])
        else:
            loss = self.loss.loss_func(synth_imgs, gen_imgs[:,0,:,:])

        
        # print(loss)

        loss.backward()

        total_norm = 0
        parameters = [p for p in self.htr_model.parameters() if p.grad is not None and p.requires_grad]
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        # print(total_norm)
        # print(loss)

        torch.nn.utils.clip_grad_norm_(self.htr_model.parameters(), 1.0)
        self.optimizer.step()
        np.save('./preds.npy', y_pred.detach().cpu().numpy())

        y_pred_max = torch.max(y_pred, dim=2).indices
        gt_labels = gt_labels.detach()
        cer, wer = self.evaluate(y_pred_max, gt_labels)

        return loss.detach().cpu(), cer, wer, total_norm
    
    def train_batch_supervised(self, batch, epoch):
        self.htr_model.train()
        imgs, _, gt_labels, _, _ = batch

        imgs = imgs.to(self.device)
        gt_labels = gt_labels.to(self.device)

        self.optimizer.zero_grad()

        print(imgs.shape)

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


    def train_model(self, train_loader, valid_loader, oov_valid_loader, epochs):
        train_saver = LossSaver(f"{self.exp_folder}", "train", 16)
        valid_saver = LossSaver(f"{self.exp_folder}", "valid", 16)
        oov_valid_saver = LossSaver(f"{self.exp_folder}", "oov_valid", 16)

        s_epoch, end_epoch = epochs
        # torch.autograd.set_detect_anomaly(True)
        print(self.loss_name)
        # charset_base = string.ascii_lowercase + string.ascii_uppercase
        # tokenizer = Tokenizer(chars=charset_base, max_text_length=25, self_supervised=False)

        min_val_loss = 100000
        patience = 15
        pat_count = 0
        
        for epoch in range(s_epoch, end_epoch):
            torch.backends.cudnn.benchmark = True
            print(epoch)
            avg_loss = 0
            avg_cer = 0
            avg_wer = 0

            for idx, batch in tqdm(enumerate(train_loader)):
                loss, cer, wer, norm = self.train_batch(batch, epoch)
                # print(loss)
                avg_loss += loss
                avg_cer += cer
                avg_wer += wer
                # if idx == 200:
                #     break

            n_train_batches = len(train_loader)

            train_saver.save_to_csv(epoch, avg_loss/n_train_batches, avg_cer/n_train_batches, avg_wer/n_train_batches)

            dir = f"./htr_models/{self.exp_folder}/"
            os.makedirs(dir, exist_ok=True)
            # TODO: save and load optimizer state
            torch.save(self.htr_model.state_dict(), f"{dir}htr_model_{self.mode}-{epoch}.model")

            print(f"mean train loss epoch {epoch}: {avg_loss/len(train_loader)}, cer: {avg_cer/len(train_loader)}, wer: {avg_wer/len(train_loader)} last norm: {norm}")

            avg_loss = 0
            avg_cer = 0
            avg_wer = 0

            for idx, batch in tqdm(enumerate(valid_loader)):
                if self.loss_name == "htr":
                    loss, cer, wer, y_pred, y_true, syn_imgs, imgs, gt_htr_indc, synth_htr_indc = self.validate(batch)
                else:
                    loss, cer, wer, y_pred, y_true  = self.validate(batch)
                # print(loss)
                avg_loss += loss
                avg_cer += cer
                avg_wer += wer
                # if idx == 200:
                #     print(loss)
                #     break

            n_valid_batches = len(valid_loader)
            
            if self.scheduler is not None:
                # self.scheduler.step()
                self.scheduler.step(avg_loss/idx)

            if self.scheduler is not None:
                # print("learning rate: ", self.scheduler.get_last_lr())
                print("learning rate: ", self.scheduler._last_lr)
            
            valid_saver.save_to_csv(epoch, avg_loss/n_valid_batches, avg_cer/n_valid_batches, avg_wer/n_valid_batches)
            print(f"mean validation loss epoch {epoch}: {avg_loss/len(valid_loader)}, cer: {avg_cer/len(valid_loader)}, wer: {avg_wer/len(valid_loader)}")

            # self.save_images(epoch, syn_imgs.cpu().numpy(), imgs.cpu().numpy(), y_pred, y_true, plot=True) 
            print("predictions last batch: ")
            for idx in range(len(y_pred)):
                if self.loss_name == "htr":
                    gt_htr_label = self.tokenizer.decode(gt_htr_indc[idx])
                    synth_htr_label = self.tokenizer.decode(synth_htr_indc[idx])
                    print(f"gt: {y_true[idx]}, self htr pred: {y_pred[idx]}, super-htr gt: {gt_htr_label}, super-htr synth: {synth_htr_label}")
                else:
                    print(f"gt: {y_true[idx]}, self htr pred: {y_pred[idx]}")

            if avg_loss / n_valid_batches <= min_val_loss:
                min_val_loss = avg_loss / n_valid_batches
                pat_count = 0
            else:
                pat_count += 1

            if oov_valid_loader is not None:
                avg_loss = 0
                avg_cer = 0
                avg_wer = 0

                for idx, batch in tqdm(enumerate(oov_valid_loader)):
                    if self.loss_name == "htr":
                        loss, cer, wer, y_pred, y_true, syn_imgs, imgs, gt_htr_indc, synth_htr_indc = self.validate(batch)
                    else:
                        loss, cer, wer, y_pred, y_true  = self.validate(batch)
                    # print(loss)
                    avg_loss += loss
                    avg_cer += cer
                    avg_wer += wer
                    # if idx == 200:
                    #     print(loss)
                    #     break

                n_valid_batches = len(oov_valid_loader)
                
                if self.scheduler is not None:
                    # self.scheduler.step()
                    self.scheduler.step(avg_loss/idx)

                if self.scheduler is not None:
                    # print("learning rate: ", self.scheduler.get_last_lr())
                    print("learning rate: ", self.scheduler._last_lr)
                
                oov_valid_saver.save_to_csv(epoch, avg_loss/n_valid_batches, avg_cer/n_valid_batches, avg_wer/n_valid_batches)
                print(f"mean oov validation loss epoch {epoch}: {avg_loss/len(valid_loader)}, cer: {avg_cer/len(valid_loader)}, wer: {avg_wer/len(valid_loader)}")

                # self.save_images(epoch, syn_imgs.cpu().numpy(), imgs.cpu().numpy(), y_pred, y_true, plot=True) 
                print("predictions last batch: ")
                for idx in range(len(y_pred)):
                    if self.loss_name == "htr":
                        gt_htr_label = self.tokenizer.decode(gt_htr_indc[idx])
                        synth_htr_label = self.tokenizer.decode(synth_htr_indc[idx])
                        print(f"gt: {y_true[idx]}, self htr pred: {y_pred[idx]}, super-htr gt: {gt_htr_label}, super-htr synth: {synth_htr_label}")
                    else:
                        print(f"gt: {y_true[idx]}, htr pred: {y_pred[idx]}")

            if pat_count == patience:
                print("Validation loss did not increase for more than 15 epochs")
                break
