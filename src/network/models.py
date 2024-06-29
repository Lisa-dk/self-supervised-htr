import torch
from torch import nn

class Puigcerver(nn.Module):
    def __init__(self, input_size, d_model):
        super(Puigcerver, self).__init__()
        print(input_size, d_model)

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=input_size[2], out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding="same", bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding="same", bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(3, 3), stride=(1, 1), padding="same", bias=False),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding="same", bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=64, out_channels=80, kernel_size=(3, 3), stride=(1, 1), padding="same", bias=False),
            nn.BatchNorm2d(80),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.cnn.apply(self.weights_init)

        self.blstm = nn.LSTM(input_size=80, hidden_size=256, num_layers=5, bidirectional=True, batch_first=True, dropout=0.5)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, d_model)
    
    def replace_head(self, new_d_model):
        self.fc = nn.Linear(512, new_d_model)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(m.weight, a=0.01)

    def forward(self, x):
        x = self.cnn(x)

        batch_size, channels, width, height = x.size()
        x = nn.functional.max_pool2d(x, [x.size(2), 1], stride=[x.size(2), 1], padding=[0, 1//2])
        x = x.permute(2, 0, 3, 1)[0]

        x = self.blstm(x)[0]

        x = self.fc(x)

        return x
    
class Puigcerver_supervised(nn.Module):
    def __init__(self, input_size, d_model):
        super(Puigcerver_supervised, self).__init__()
        print(input_size, d_model)

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=input_size[2], out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding="same", bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding="same", bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(3, 3), stride=(1, 1), padding="same", bias=False),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding="same", bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=64, out_channels=80, kernel_size=(3, 3), stride=(1, 1), padding="same", bias=False),
            nn.BatchNorm2d(80),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.cnn.apply(self.weights_init)

        self.blstm = nn.LSTM(input_size=80, hidden_size=256, num_layers=5, bidirectional=True, batch_first=True, dropout=0.0)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, d_model)

    
    def replace_head(self, new_d_model):
        self.fc = nn.Linear(512, new_d_model)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(m.weight, a=0.01)

    def forward(self, x):
        x = self.cnn(x)

        batch_size, channels, width, height = x.size()
        x = nn.functional.max_pool2d(x, [x.size(2), 1], stride=[x.size(2), 1], padding=[0, 1//2])
        x = x.permute(2, 0, 3, 1)[0]

        x = self.blstm(x)[0]

        x = self.fc2(self.dropout2(x))

        return x
    
class Puigcerver_Dropout(nn.Module):
    def __init__(self, input_size, d_model):
        super(Puigcerver_Dropout, self).__init__()
        print(input_size, d_model)

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=input_size[2], out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding="same", bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding="same", bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(3, 3), stride=(1, 1), padding="same", bias=False),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding="same", bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=64, out_channels=80, kernel_size=(3, 3), stride=(1, 1), padding="same", bias=False),
            nn.BatchNorm2d(80),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.cnn.apply(self.weights_init)

        self.blstm = nn.LSTM(input_size=640, hidden_size=256, num_layers=5, bidirectional=True, batch_first=True, dropout=0.5)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, d_model)
    
    def replace_head(self, new_d_model):
        self.fc = nn.Linear(512, new_d_model)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(m.weight, a=0.01)

    def forward(self, x):
        x = self.cnn(x)

        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, height, width * channels)
        print(x.shape)

        x, _ = self.blstm(x)
        x = self.fc(self.dropout(x))

        return x
    