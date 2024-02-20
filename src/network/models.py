# from models.encoder_vgg import Encoder as rec_encoder
# from models.decoder import Decoder as rec_decoder
# from models.seq2seq import Seq2Seq as rec_seq2seq
# from models.attention import locationAttention as rec_attention
import torch
from torch import nn

# class GanRecModel(nn.Module):
#     def __init__(self, pretrain=False, input_size=(216, 64, 1), max_output_length=10, vocab_size=26):
#         super(GanRecModel, self).__init__()
#         hidden_size_enc = hidden_size_dec = 512
#         embed_size = 60
#         h, w, _ = input_size
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         self.enc = rec_encoder(hidden_size_enc, h, w, True, None, False).to(device)
#         self.dec = rec_decoder(hidden_size_dec, embed_size, vocab_size, rec_attention, None).to(device)
#         self.seq2seq = rec_seq2seq(self.enc, self.dec, max_output_length, vocab_size).to(device)
#         if pretrain:
#             model_file = 'recognizer/save_weights/seq2seq-72.model_5.79.bak'
#             print('Loading RecModel', model_file)
#             self.seq2seq.load_state_dict(torch.load(model_file))

#     def forward(self, img, label, img_width):
#         self.seq2seq.train()
#         img = torch.cat([img,img,img], dim=1) # b,1,64,128->b,3,64,128
#         output, attn_weights = self.seq2seq(img, label, img_width, teacher_rate=False, train=False)
#         return output.permute(1, 0, 2) # t,b,83->b,t,83


class Puigcerver(nn.Module):
    def __init__(self, input_size, d_model):
        super(Puigcerver, self).__init__()
        print(input_size, d_model)

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=input_size[2], out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding="same"),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding="same"),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(3, 3), stride=(1, 1), padding="same"),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding="same"),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=64, out_channels=80, kernel_size=(3, 3), stride=(1, 1), padding="same"),
            nn.BatchNorm2d(80),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.cnn.apply(self.weights_init)

        self.blstm = nn.LSTM(input_size=640, hidden_size=256, num_layers=5, bidirectional=True, batch_first=True, dropout=0.5)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, d_model)
        # self.softmax = nn.Softmax(dim=2)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(m.weight, a=0.01)

    def forward(self, x):
        # print(x.shape)
        x = self.cnn(x)
        # print(x.shape)
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, height, width * channels)
        # print(x.shape)
        x, _ = self.blstm(x)
        x = self.fc(self.dropout(x))
        # print(x.shape)
        # x = self.softmax(x)
        return x