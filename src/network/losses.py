from torch import nn
import torch
from gen_model.gen_model import GenModel_FC

class Loss:
    def __init__(self, loss_name, self_supervised, tokenizer):
        self.loss_func = self.get_loss_func(loss_name)
        self.tokenizer = tokenizer

        if self_supervised:
            self.vgg_model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True).features[:11]
            self.vgg_model.eval()

            self.gen_model = GenModel_FC(tokenizer.maxlen, tokenizer.vocab_size, tokenizer.PAD)
            self.gen_model.load_state_dict(torch.load('./contran-gen.model')) #load
            self.gen_model.eval()


    def get_loss_func(self, loss_input):
        if loss_input.lower() == "ctc":
            return self.ctc_loss_func
        elif loss_input.lower() == "ssim":
            print("ssim loss")
            return self.ssim_loss
        else:
            print("Loss not implemented. Choose one of: ctc, ssim")
    
    def ctc_loss_func(self, y_pred, gt_labels):
        ctc_loss = nn.CTCLoss(blank=self.tokenizer.BLANK, zero_infinity=True)

        y_pred_log = nn.functional.log_softmax(y_pred, dim=2)

        input_lengths = torch.full(size=(y_pred_log.shape[0], ), fill_value=y_pred_log.shape[1], dtype=torch.int)
        target_lengths = torch.Tensor([self.tokenizer.maxlen - torch.bincount(i)[self.tokenizer.PAD] for i in gt_labels]).to(torch.int)

        y_pred_log = torch.permute(y_pred_log, dims=(1,0,2))

        print(y_pred_log.shape, gt_labels.shape)
        ctc = ctc_loss(y_pred_log, gt_labels, input_lengths, target_lengths)
        return ctc

    def perceptual_loss_func(self, y_pred, gt_imgs):
        synth_imgs = self.gen_model(gt_imgs, y_pred)

        # TODO: copy twice for 3 channel vgg input

        gt_feats = self.vgg_model(gt_imgs[0]) # htr input is first image, others are to capture style
        synth_feats = self.vgg_model(synth_imgs)
        return torch.mean((gt_feats - synth_feats ** 2))


