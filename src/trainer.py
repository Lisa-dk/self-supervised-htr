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
        print(imgs.shape)

        y_pred = self.htr_model(imgs)
        y_log_pred = torch.log(y_pred)
        print(y_log_pred.shape)
        print(gt_label.shape)

        input_lengths = torch.full(size=(y_log_pred.shape[0],), fill_value=y_log_pred.shape[1], dtype=torch.long)
        target_lengths = torch.full(size=(gt_label.shape[0], ), fill_value=torch.count_nonzero(gt_label, dim=1), dtype=torch.long)

        ctc = self.loss(y_log_pred, gt_label, input_lengths, target_lengths)
        print(ctc)
        return ctc





    def train_model(self, train_loader, valid_loader, epochs):
        s_epoch, end_epoch = epochs
        for epoch in range(s_epoch, end_epoch):
            print(epoch)

            for i, batch in enumerate(train_loader):
                loss = self.train_batch(batch)





        