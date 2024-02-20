import numpy as np
import os
import torch
from vgg_tro_channel3_modi import vgg19_bn
from blocks import LinearBlock, Conv2dBlock, ResBlocks, ActFirstResBlock
from torch import nn
import torch.nn.functional as Fun

gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GenModel_FC(nn.Module):
    def __init__(self, text_max_len, vocab_size, pad_token):
        super(GenModel_FC, self).__init__()
        self.enc_image = ImageEncoder().to(gpu)
        self.enc_text = TextEncoder_FC(text_max_len, vocab_size, pad_token).to(gpu)
        self.dec = Decoder().to(gpu)
        self.linear_mix = nn.Linear(1024, 512)

    def decode(self, content, adain_params):
        # decode content and style codes to an image
        def assign_adain_params(adain_params, model):
            # assign the adain_params to the AdaIN layers in model
            for m in model.modules():
                if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                    mean = adain_params[:, :m.num_features]
                    std = adain_params[:, m.num_features:2*m.num_features]
                    m.bias = mean.contiguous().view(-1)
                    m.weight = std.contiguous().view(-1)
                    if adain_params.size(1) > 2*m.num_features:
                        adain_params = adain_params[:, 2*m.num_features:]

        assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images

    # feat_mix: b,1024,8,27
    def mix(self, feat_xs, feat_embed):
        feat_mix = torch.cat([feat_xs, feat_embed], dim=1) # b,1024,8,27
        f = feat_mix.permute(0, 2, 3, 1)
        ff = self.linear_mix(f) # b,8,27,1024->b,8,27,512
        return ff.permute(0, 3, 1, 2)
    
    def forward(self, image_input, text_input):
        print("entered forward generator")
        f_xs = self.enc_image(image_input) # b,512,8,27
        f_xt, f_embed = self.enc_text(text_input, f_xs.shape) # b,4096  b,512,8,27
        f_mix = self.mix(f_xs, f_embed)

        xg = self.decode(f_mix, f_xt)  # translation b,1,64,128
        return xg

class TextEncoder_FC(nn.Module):
    def __init__(self, text_max_len, vocab_size, pad_token):
        super(TextEncoder_FC, self).__init__()
        # embed_size = 64
        self.embed_size = vocab_size
        self.max_text_len = text_max_len
        self.pad_token = pad_token

        self.fc = nn.Sequential(
                nn.Linear(text_max_len*self.embed_size, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=False),
                nn.Linear(1024, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=False),
                nn.Linear(2048, 4096)
                )
        '''embed content force'''
        self.linear = nn.Linear(self.embed_size, 512)

    def forward(self, x, f_xs_shape):
        # xx = self.embed(x) # b,t,embed        
        xx = x

        batch_size = xx.shape[0]
        xxx = xx.reshape(batch_size, -1) # b,t*embed
        out = self.fc(xxx)

        '''embed content force'''
        xx_new = self.linear(xx) # b, text_max_len, 512
        ts = xx_new.shape[1]
        height_reps = f_xs_shape[-2]
        width_reps = f_xs_shape[-1] // ts
        tensor_list = list()
        for i in range(ts):
            text = [xx_new[:, i:i + 1]] # b, text_max_len, 512
            tmp = torch.cat(text * width_reps, dim=1)
            tensor_list.append(tmp)

        padding_reps = f_xs_shape[-1] % ts
        if padding_reps:
            padding = torch.full((1, 1), self.pad_token, dtype=torch.long)
            # print(Fun.one_hot(padding, num_classes = vocab_size).float())
            # embedded_padding_char = self.embed(torch.full((1, 1), tokens['PAD_TOKEN'], dtype=torch.long).cuda())
            embedded_padding_char = Fun.one_hot(padding, num_classes = self.embed_size).float().cuda()
            embedded_padding_char = self.linear(embedded_padding_char)
            padding = embedded_padding_char.repeat(batch_size, padding_reps, 1)
            tensor_list.append(padding)

        res = torch.cat(tensor_list, dim=1) # b, text_max_len * width_reps + padding_reps, 512
        res = res.permute(0, 2, 1).unsqueeze(2) # b, 512, 1, text_max_len * width_reps + padding_reps
        final_res = torch.cat([res] * height_reps, dim=2)

        return out, final_res


'''VGG19_IN tro'''
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.model = vgg19_bn(False)
        self.output_dim = 512

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, ups=3, n_res=2, dim=512, out_dim=1, res_norm='adain', activ='relu', pad_type='reflect'):
        super(Decoder, self).__init__()

        self.model = []
        self.model += [ResBlocks(n_res, dim, res_norm,
                                 activ, pad_type=pad_type)]
        for i in range(ups):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2,
                                       norm='in',
                                       activation=activ,
                                       pad_type=pad_type)]
            dim //= 2
        self.model += [Conv2dBlock(dim, out_dim, 7, 1, 3,
                                   norm='none',
                                   activation='tanh',
                                   pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)