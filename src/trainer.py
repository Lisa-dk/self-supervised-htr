import torch
from torch import nn

class HTRtrainer(object):
    def __init__(self, model, optimizer):
        super(HTRtrainer, self).__init__()
        self.htr_model = model
        self.optimizer = optimizer
        self.loss = nn.CTCLoss()

    
    def train_batch(self, batch):
        imgs, gt_label = batch


        self.optimizer.zero_grad()
        print(imgs)

        y_pred = self.htr_model(imgs)
        y_log_pred = torch.log(y_pred)
        print(y_log_pred.shape)
        print(gt_label.shape)

        ctc = self.loss(y_log_pred, gt_label)
        print(ctc)
        return ctc





    def train_model(self, train_loader, valid_loader, epochs):
        s_epoch, end_epoch = epochs
        for epoch in range(s_epoch, end_epoch):
            print(epoch)

            for i, batch in enumerate(train_loader):
                loss = self.train_batch(batch)





        