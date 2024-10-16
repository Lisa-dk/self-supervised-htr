from torch import nn
import torch
from torchvision import models

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchsummary import summary
from network.models import Puigcerver_supervised
import kornia, string
from network.sia_model.model import SiameseNetwork
import matplotlib.pyplot as plt
from data.tokenizer import Tokenizer

class Loss:
    def __init__(self, loss_name, tokenizer, device, vgg_layer):
        self.tokenizer = tokenizer
        self.device = device
        self.loss_func = self.get_loss_func(loss_name, vgg_layer)
        

    def get_loss_func(self, loss_input, vgg_layer):
        if loss_input.lower() == "ctc":
            self.ctc_loss = nn.CTCLoss(blank=self.tokenizer.BLANK, zero_infinity=True)
            return self.ctc_loss_func

        if loss_input.lower() == "ce":
            return self.ce_loss
        
        elif loss_input.lower() == "ssim":
            self.ssim = StructuralSimilarityIndexMeasure(data_range=(-1.0, 1.0)).to(self.device)
            return self.ssim_loss

        elif loss_input.lower() == "ms_ssim":
            return self.ms_ssim_loss
        
        elif loss_input.lower() == "vgg":
            self.vgg_model = models.vgg16(pretrained=True).features[:vgg_layer]
            self.vgg_model.to(self.device)
            print(self.vgg_model)
            self.vgg_model.eval()

            for param in self.vgg_model.parameters():
                param.requires_grad = False

            return self.perceptual_loss_func
        
        elif loss_input.lower() == "vgg_ssim":
            self.ssim = StructuralSimilarityIndexMeasure(data_range=10.0).to(self.device)
            self.vgg_model = models.vgg16(pretrained=True).features[:vgg_layer]
            self.vgg_model.to(self.device)
            print(self.vgg_model)
            self.vgg_model.eval()

            for param in self.vgg_model.parameters():
                param.requires_grad = False

            return self.vgg_ssim_loss
        
        elif loss_input.lower() == "pp":
            return self.prof_loss
        
        elif loss_input.lower() == "htr":
            self.iam_model = Puigcerver_supervised((64, 216, 1), self.tokenizer.vocab_size)
            model_name = f"./htr_models/iam_gan/ce-10char-maxpool/htr_model_supervised-73.model"
            self.iam_model.load_state_dict(torch.load(model_name))
            self.iam_model = self.iam_model.to(self.device)
            self.iam_model.cnn.eval()
            self.iam_model.dropout2.eval()

            for param in self.iam_model.parameters():
                param.requires_grad = False
            # print(self.iam_model)
            return self.htr_loss
        
        elif loss_input.lower() == "vgg_pp":
            self.vgg_model = models.vgg16(pretrained=True).features[:vgg_layer]
            self.vgg_model.to(self.device)
            print(self.vgg_model)
            self.vgg_model.eval()

            for param in self.vgg_model.parameters():
                param.requires_grad = False

            return self.vgg_prof
        
        elif loss_input.lower() == "siamese":
            self.feat_model = SiameseNetwork("resnet34", False)
            self.feat_model.load_state_dict(torch.load('../src/network/sia_model/resnet34-RMS-17.model'))
            self.feat_model.to(self.device)
            self.feat_model.eval()

            for param in self.feat_model.parameters():
                param.requires_grad = False

            return self.siamese_loss
        else:
            print("Loss not implemented. Choose one of: ctc, ssim, perceptual")
            exit()
    
    def ce_loss(self, y_pred, gt_labels):
        gt_labels_1hot = gt_labels.long()
        gt_labels_1hot = torch.nn.functional.one_hot(gt_labels_1hot, self.tokenizer.vocab_size).float()
        
        y_pred = y_pred.view(-1, self.tokenizer.vocab_size)
        gt_labels_1hot = gt_labels_1hot.view(-1, self.tokenizer.vocab_size)

        return nn.functional.cross_entropy(y_pred, gt_labels_1hot)

    def ctc_loss_func(self, y_pred, gt_labels):

        y_pred_log = nn.functional.log_softmax(y_pred, dim=2)

        input_lengths = torch.full(size=(y_pred_log.shape[0], ), fill_value=y_pred_log.shape[1], dtype=torch.int)
        target_lengths = torch.Tensor([self.tokenizer.maxlen - torch.bincount(i)[self.tokenizer.PAD] for i in gt_labels]).to(torch.int)

        y_pred_log = torch.permute(y_pred_log, dims=(1,0,2))

        return self.ctc_loss(y_pred_log, gt_labels, input_lengths, target_lengths)

    def ssim_loss(self, synth_imgs, gt_img):
        return 1. - self.ssim(preds=synth_imgs, target=gt_img.unsqueeze(1))
    
    def ms_ssim_loss(self, synth_imgs, gt_imgs):
        criterion = kornia.losses.MS_SSIMLoss().to(self.device)
        gt_imgs = (gt_imgs - (-1.)) / 2.    # 0-1
        synth_imgs = (synth_imgs - (-1.)) / 2.

        loss = criterion(synth_imgs, gt_imgs.unsqueeze(1))
        return loss

    def vgg_ssim_loss(self, synth_imgs, gt_img):
        gt_vgg_input = torch.stack([gt_img, gt_img, gt_img], dim=1)
        synth_vgg_input = torch.stack([synth_imgs.squeeze(1), synth_imgs.squeeze(1), synth_imgs.squeeze(1)], dim=1)

        gt_feats = self.vgg_model(gt_vgg_input)
        synth_feats = self.vgg_model(synth_vgg_input)

        vgg_loss = torch.mean((gt_feats - synth_feats) ** 2)

        ssim_loss = 1. - self.ssim(preds=synth_imgs, target=gt_img.unsqueeze(1))

        return ssim_loss + vgg_loss

    def perceptual_loss_func(self, synth_imgs, gt_img):
        gt_vgg_input = torch.stack([gt_img, gt_img, gt_img], dim=1)
        synth_vgg_input = torch.stack([synth_imgs.squeeze(1), synth_imgs.squeeze(1), synth_imgs.squeeze(1)], dim=1)

        gt_feats = self.vgg_model(gt_vgg_input) 
        synth_feats = self.vgg_model(synth_vgg_input)

        return torch.mean((gt_feats - synth_feats) ** 2)
    
    def prof_loss(self, synth_imgs, gt_imgs):
        gt_imgs = (gt_imgs - (-1.)) / 2.    # 0-1
        synth_imgs = (synth_imgs - (-1.)) / 2.

        vertical_profile = gt_imgs.sum(dim=1) / synth_imgs.shape[-2]
        synth_vertical_profile = synth_imgs.squeeze(1).sum(dim=1) / synth_imgs.shape[-2]

        return torch.mean((vertical_profile - synth_vertical_profile) ** 2.)      
    
    def htr_loss(self, synth_imgs, gt_imgs):
        gt_feats = self.iam_model(gt_imgs.squeeze(0).unsqueeze(1))
        gt_feats = nn.functional.softmax(gt_feats, 2)

        y_pred_max_flat = gt_feats.view(-1, self.tokenizer.vocab_size)

        synth_feats =  self.iam_model(synth_imgs)
        synth_feats_flat = synth_feats.view(-1, self.tokenizer.vocab_size)
        
        return nn.functional.cross_entropy(synth_feats_flat, y_pred_max_flat), gt_feats, nn.functional.softmax(synth_feats, 2)
    
    def vgg_prof(self, synth_imgs, gt_imgs):
        synth_imgs = synth_imgs.squeeze(1)

        gt_vgg_input = torch.stack([gt_imgs, gt_imgs, gt_imgs], dim=1)
        synth_vgg_input = torch.stack([synth_imgs, synth_imgs, synth_imgs], dim=1)

        gt_feats = self.vgg_model(gt_vgg_input) 
        synth_feats = self.vgg_model(synth_vgg_input)

        vgg_loss =  torch.mean((gt_feats - synth_feats) ** 2)

        gt_imgs = (gt_imgs - (-1.)) / 2.    # 0-1
        synth_imgs = (synth_imgs - (-1.)) / 2.
        
        vertical_profile = gt_imgs.sum(dim=1) / synth_imgs.shape[-2]
        synth_vertical_profile = synth_imgs.sum(dim=1) / synth_imgs.shape[-2]

        proj_prof_loss = torch.mean((vertical_profile - synth_vertical_profile) ** 2.) * 100
        return vgg_loss + proj_prof_loss

    def siamese_loss(self, synth_imgs, gt_imgs):

        gt_input = torch.stack([gt_imgs, gt_imgs, gt_imgs], dim=1).squeeze(2)
        synth_input = torch.stack([synth_imgs, synth_imgs, synth_imgs], dim=1).squeeze(2)

        feats1, feats2 = self.feat_model(gt_input, synth_input)
        return torch.mean(nn.functional.pairwise_distance(feats1, feats2))