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

    dataset_path = os.path.join("..", "data", "iam", "words")

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_size = (64, 216, 1)
    
    # get data paths and labels (path, label, wid)
    data_train, data_valid, data_test, valid_oov, test_oov = read_data(dataset_path)

    print(data_train[0])
    print(data_valid[0])
    print(data_test[0])


    data_train = IAM_data(data_train)
    data_valid = IAM_data(data_valid)
    data_test = IAM_data(data_test)
    valid_oov = IAM_data(valid_oov)
    test_oov = IAM_data(test_oov)

    num_workers = 4
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(data_valid, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    valid_oov_loader = torch.utils.data.DataLoader(valid_oov, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    test_oov_loader = torch.utils.data.DataLoader(test_oov, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    
    model = SiameseNetwork(model_name=args.model, edisdist=args.editdistance)
    # model.load_state_dict(torch.load(f'./models/results/{args.model}-19.model'))
    print(model.model)
    epochs = (0, 30)

    if args.train:
        if args.editdistance:
            trainer = Trainer(model, device, "./results/semisup/", args.model + "-RMS-edit", args.editdistance)
        else:
            trainer = Trainer(model, device, "./results/semisup/", args.model + "-RMS", args.editdistance)
        trainer.train_model(train_loader, (valid_loader, valid_oov_loader), epochs)

    elif args.test or args.valid:
        model.load_state_dict(torch.load(f'./models/results/{args.model}-2.model'))
        model.eval()
        trainer = Trainer(model, device, "./results/")

        if args.test:
            loader = test_loader
            oov_loader = test_oov_loader
        elif args.valid:
            loader = valid_loader
            oov_loader = valid_oov_loader
        
        avg_loss = 0
        for idx, batch in tqdm(enumerate(loader)):
            loss, img1, img2, labels = trainer.validate(batch)
            # print(loss)
            avg_loss += loss

        print(f"mean validation loss: {avg_loss/len(loader)}")

        avg_loss = 0
        for idx, batch in tqdm(enumerate(oov_loader)):
            loss, img1, img2, labels = trainer.validate(batch)
            # print(loss)
            avg_loss += loss

        print(f"mean oov validation loss epoch: {avg_loss/len(oov_loader)}")

