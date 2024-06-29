import os, sys
import torch
import glob, cv2
from torch import optim
import numpy as np
import time
import argparse
import string
from data.reader import read_data
import data.preproc
from data.data_loader import IAM_data
from network.model import SiameseNetwork
from tqdm import tqdm
from trainer import Trainer
from scipy.stats import ttest_ind


if __name__ == "__main__":
    max_text_length = 25
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--valid", action="store_true", default=False)
    parser.add_argument("--editdistance", action="store_true", default=False)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument('--start_epoch', type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--max_word_len", type=int, default=max_text_length)
    parser.add_argument("--model", type=str, default="resnet34")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = (64, 216, 1)
    
    # Get data paths and labels (path, label, wid)
    dataset_path = os.path.join("..", "data", "iam_gan", "words")
    data_train, data_valid, data_test = read_data(dataset_path)

    print(data_train[0])
    print(data_valid[0])
    print(data_test[0])

    data_train = IAM_data(data_train)
    data_valid = IAM_data(data_valid)
    data_test = IAM_data(data_test)

    num_workers = 4
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(data_valid, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    
    model = SiameseNetwork(model_name=args.model, edisdist=args.editdistance)
    print(model.model)
    
    epochs = (args.start_epoch, args.epochs)

    if args.train:
        if args.editdistance:
            trainer = Trainer(model, device, "./results/", args.model + "-RMS-edit", args.editdistance)
        else:
            trainer = Trainer(model, device, "./results/", args.model + "-RMS", args.editdistance)
        trainer.train_model(train_loader, valid_loader, epochs)

    elif args.test or args.valid:
        model.load_state_dict(torch.load(f'./models/results/{args.model}-RMS-17.model'))
        model.eval()

        for param in model.parameters():
            param.requires_grad = False
            
        trainer =  Trainer(model, device, "./results/", args.model + "-RMS", args.editdistance)

        if args.test:
            loader = test_loader
        elif args.valid:
            loader = valid_loader

        avg_loss = 0
        pos_dists = []
        neg_dists = []
        for idx, batch in tqdm(enumerate(loader)):
            loss, img1, img2, labels, pos, neg = trainer.validate(batch)
            # print(loss)
            avg_loss += loss
            pos_dists.append(pos)
            neg_dists.append(neg)
        

        diff = [neg_dists[i] - pos_dists[i] for i in range(len(neg_dists))]
        print(f"mean validation loss: {avg_loss/len(loader)} mean pos dist: {np.mean(pos_dists)} pm {np.std(pos_dists)} mean neg dist: {np.mean(neg_dists)} pm {np.std(neg_dists)} mean diff: {np.mean(diff)} pm {np.std(diff)}")
        
        # Two-tailed t-test
        t, p = ttest_ind(neg_dists, pos_dists) 
        print("df: ", ttest_ind(neg_dists, pos_dists).df )
        print("T statistic:", t)  
        print("p-value", p)

