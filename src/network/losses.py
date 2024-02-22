from torch import nn
import torch

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchsummary import summary

class Loss:
    def __init__(self, loss_name, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device
        self.loss_func = self.get_loss_func(loss_name, device)
        

    def get_loss_func(self, loss_input, device):
        if loss_input.lower() == "ctc":
            self.ctc_loss = nn.CTCLoss(blank=self.tokenizer.BLANK, zero_infinity=True)
            return self.ctc_loss_func
        elif loss_input.lower() == "ssim":
            self.ssim = StructuralSimilarityIndexMeasure(data_range=(-1.0, 1.0)).to(self.device)
            return self.ssim_loss
        elif loss_input.lower() == "perceptual":
            self.vgg_model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
            summary(self.vgg_model(3, 224, 224))
            self.vgg_model.eval()
            return self.perceptual_loss_func
        else:
            print("Loss not implemented. Choose one of: ctc, ssim")
    
    def ctc_loss_func(self, y_pred, gt_labels):

        y_pred_log = nn.functional.log_softmax(y_pred, dim=2)

        input_lengths = torch.full(size=(y_pred_log.shape[0], ), fill_value=y_pred_log.shape[1], dtype=torch.int)
        target_lengths = torch.Tensor([self.tokenizer.maxlen - torch.bincount(i)[self.tokenizer.PAD] for i in gt_labels]).to(torch.int)

        y_pred_log = torch.permute(y_pred_log, dims=(1,0,2))

        print(y_pred_log.shape, gt_labels.shape)
        ctc = self.ctc_loss(y_pred_log, gt_labels, input_lengths, target_lengths)
        return ctc
    

    def ssim_loss(self, synth_imgs, gt_imgs):
        return 1. - self.ssim(preds=synth_imgs, target=gt_imgs[:,0:1,:])

    def perceptual_loss_func(self, synth_imgs, gt_imgs):
        # TODO: copy twice for 3 channel vgg input

        gt_feats = self.vgg_model(gt_imgs[0]) # htr input is first image, others are to capture style
        synth_feats = self.vgg_model(synth_imgs)
        return torch.mean((gt_feats - synth_feats ** 2))


