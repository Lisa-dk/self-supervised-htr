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

        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001)
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.0001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=1, verbose=True)

        # self.loss = 
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
            # out = self.model(imgs1, imgs2.unsqueeze(1))

            l2_dist = nn.functional.pairwise_distance(feats1, feats2)

            loss = self.contrastive_loss(l2_dist, labels)
        else:
            pred_edit = self.model(imgs1, imgs2)
            loss = torch.mean(nn.functional.pairwise_distance(pred_edit, dist))

        # bce = nn.BCELoss()
        # loss = bce(out, labels)
        # print(labels, torch.sum(labels))

        return loss.detach().cpu(), imgs1.cpu(), imgs2.cpu(), labels.cpu()
    
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
            # out = self.model(imgs1, imgs2.unsqueeze(1))

            l2_dist = nn.functional.pairwise_distance(feats1, feats2)

            loss = self.contrastive_loss(l2_dist, labels)
        else:
            pred_edit = self.model(imgs1, imgs2)
            loss = torch.mean(nn.functional.pairwise_distance(pred_edit, dist))

        # bce = nn.BCELoss()
        # loss = bce(out, labels)

        loss.backward()

        total_norm = 0
        parameters = [p for p in self.model.parameters() if p.grad is not None and p.requires_grad]
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.detach().cpu(), total_norm


    def train_model(self, train_loader, valid_loader, epochs):
        if self.mode:
            train_saver = LossSaver(f"{self.exp_folder}-{self.model_name}-edit", "train", 16)
            valid_saver = LossSaver(f"{self.exp_folder}-{self.model_name}-edit", "valid", 16)
        else:
            train_saver = LossSaver(f"{self.exp_folder}-{self.model_name}", "train", 16)
            valid_saver = LossSaver(f"{self.exp_folder}-{self.model_name}", "valid", 16)
        s_epoch, end_epoch = epochs
        # torch.autograd.set_detect_anomaly(True)
        valid_loader, valid_oov_loader = valid_loader
        
        for epoch in range(s_epoch, end_epoch):
            torch.backends.cudnn.benchmark = True
            print(epoch)

            avg_loss = 0
            for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
                loss, norm = self.train_batch(batch)
                # print(loss)
                avg_loss += loss


            train_saver.save_to_csv(epoch, avg_loss/len(train_loader))
            dir = f"./models/{self.exp_folder}/"
            os.makedirs(dir, exist_ok=True)
            # TODO: save and load optimizer state
            torch.save(self.model.state_dict(), f"{dir}{self.model_name}-{epoch}.model")
            print(f"mean train loss epoch {epoch}: {avg_loss/len(train_loader)} norm: {norm}")

            avg_loss = 0
            for idx, batch in tqdm(enumerate(valid_loader)):
                loss, img1, img2, labels = self.validate(batch)
                # print(loss)
                avg_loss += loss

            valid_saver.save_to_csv(epoch, avg_loss/len(valid_loader))
            print(f"mean validation loss epoch {epoch}: {avg_loss/len(valid_loader)}")
            if self.scheduler is not None:
                self.scheduler.step(avg_loss/len(valid_loader))

            avg_loss = 0
            for idx, batch in tqdm(enumerate(valid_oov_loader)):
                loss, img1, img2, labels = self.validate(batch)
                # print(loss)
                avg_loss += loss

            print(f"mean oov validation loss epoch {epoch}: {avg_loss/len(valid_oov_loader)}")

            # for idx in range(len(img1)):
            #     f, axarr = plt.subplots(1,2)
            #     axarr[0].imshow(img1[idx][0], cmap='gray')
            #     axarr[1].imshow(img2[idx][0], cmap='gray')
            #     plt.show()

# 0
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2363/2363 [11:47<00:00,  3.34it/s]
# mean train loss epoch 0: 9.131165504455566 norm: 14.723899852457762
# 562it [00:50, 11.16it/s]
# mean validation loss epoch 0: 0.17019449174404144
# 1
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2363/2363 [11:37<00:00,  3.39it/s]
# mean train loss epoch 1: 0.1515507847070694 norm: 5.738639142242102
# 562it [00:41, 13.55it/s]
# mean validation loss epoch 1: 0.1633196473121643
# 2
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2363/2363 [11:34<00:00,  3.40it/s]
# mean train loss epoch 2: 0.11683313548564911 norm: 12.456202477066661
# 562it [00:49, 11.35it/s]
# mean validation loss epoch 2: 0.12903067469596863
# 3
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2363/2363 [11:37<00:00,  3.39it/s]
# mean train loss epoch 3: 0.08936439454555511 norm: 6.406335378572982
# 562it [00:46, 11.98it/s]
# mean validation loss epoch 3: 0.1250876486301422
# 4
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2363/2363 [11:38<00:00,  3.38it/s]
# mean train loss epoch 4: 0.06762882322072983 norm: 5.684662835854551
# 562it [00:48, 11.55it/s]
# mean validation loss epoch 4: 0.12579762935638428
# 5
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2363/2363 [11:36<00:00,  3.39it/s]
# mean train loss epoch 5: 0.049406688660383224 norm: 6.408255573782807
# 562it [00:49, 11.33it/s]
# mean validation loss epoch 5: 0.14729316532611847
# 6
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2363/2363 [11:35<00:00,  3.40it/s]
# mean train loss epoch 6: 0.04081016406416893 norm: 18.120152228691587
# 562it [00:49, 11.32it/s]
# mean validation loss epoch 6: 0.2335125207901001
# 7
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2363/2363 [1:10:39<00:00,  1.79s/it]
# mean train loss epoch 7: 0.03377377241849899 norm: 5.619276169269042
# 562it [00:49, 11.26it/s]
# mean validation loss epoch 7: 0.264232873916626
# 8
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2363/2363 [11:38<00:00,  3.38it/s]
# mean train loss epoch 8: 0.0314311645925045 norm: 5.117054665656228
# 562it [00:49, 11.33it/s]
# mean validation loss epoch 8: 0.24039557576179504
# 9
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2363/2363 [11:35<00:00,  3.40it/s]
# mean train loss epoch 9: 0.030371839180588722 norm: 1.0023062389928872
# 562it [00:49, 11.29it/s]
            









# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4726/4726 [21:13<00:00,  3.71it/s]
# mean train loss epoch 0: 4.9458818435668945 norm: 8.614369653359002
# 1123it [01:18, 14.24it/s]
# mean validation loss epoch 0: 0.11686596274375916
# 1
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4726/4726 [53:33<00:00,  1.47it/s]
# mean train loss epoch 1: 0.10012788325548172 norm: 6.409809375249177
# 1123it [01:41, 11.11it/s]
# mean validation loss epoch 1: 0.09872733801603317
# 2
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4726/4726 [23:26<00:00,  3.36it/s]
# mean train loss epoch 2: 0.07148121297359467 norm: 3.698145516175559
# 1123it [01:19, 14.18it/s]
# mean validation loss epoch 2: 0.08241306990385056
# 3
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4726/4726 [23:25<00:00,  3.36it/s]
# mean train loss epoch 3: 0.054169196635484695 norm: 5.559542189978506
# 1123it [01:30, 12.38it/s]
# mean validation loss epoch 3: 0.09397230297327042
# 4
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4726/4726 [23:22<00:00,  3.37it/s]
# mean train loss epoch 4: 0.04215517267584801 norm: 9.399924296802226
# 1123it [01:41, 11.10it/s]
# mean validation loss epoch 4: 0.10214189440011978
# 5
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4726/4726 [23:24<00:00,  3.37it/s]
# mean train loss epoch 5: 0.03561805561184883 norm: 4.803744260100148
# 1123it [01:41, 11.11it/s]
# mean validation loss epoch 5: 0.1057288721203804
# 6
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4726/4726 [23:23<00:00,  3.37it/s]
# mean train loss epoch 6: 0.03228502348065376 norm: 6.726673315237875
# 1123it [01:25, 13.10it/s]
# mean validation loss epoch 6: 0.15693551301956177
# 7
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4726/4726 [23:22<00:00,  3.37it/s]
# mean train loss epoch 7: 0.030383506789803505 norm: 9.095550107984728
# 1123it [01:22, 13.61it/s]
# mean validation loss epoch 7: 0.22618472576141357
# 8
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4726/4726 [23:23<00:00,  3.37it/s]
# mean train loss epoch 8: 0.027786381542682648 norm: 3.430838040048656
# 1123it [01:32, 12.17it/s]
# mean validation loss epoch 8: 0.15515603125095367
# 9
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4726/4726 [23:25<00:00,  3.36it/s]
# mean train loss epoch 9: 0.026962382718920708 norm: 1.7828732123352142
# 1123it [01:41, 11.09it/s]
# mean validation loss epoch 9: 0.17362195253372192