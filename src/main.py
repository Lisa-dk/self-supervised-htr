import os, sys
import torch
import glob
from torch import optim
import numpy as np
import time
import argparse
from data.tokenizer import Tokenizer
import string
from data.reader import read_rimes
from data.preproc import preproc_rimes
from network.models import Puigcerver
from trainer import HTRtrainer
from data.data_loader import RIMES_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--self_supervised", action="store_true", default=False)

    parser.add_argument("--preproc", action="store_true", default=False)

    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument('--start_epoch', type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--loss", type=str, default="")
    args = parser.parse_args()

    dataset_path = os.path.join("..", "data", args.dataset, "words")

    if args.preproc:
        path_from = os.path.join("..", "raw", args.dataset, "words")
        os.makedirs(dataset_path, exist_ok=True)
        preproc_rimes(path_from, dataset_path)


    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_size = (64, 216, 1)
    num_style_imgs = 15

    charset_base = string.ascii_lowercase + string.ascii_uppercase
    max_text_length = 25
    tokenizer = Tokenizer(chars=charset_base, max_text_length=max_text_length)

    
    data_train, data_valid, data_test = read_rimes(dataset_path)

    data_train = RIMES_data(data_train, input_size=input_size, tokenizer=tokenizer, num_images=num_style_imgs)
    data_valid = RIMES_data(data_valid, input_size=input_size, tokenizer=tokenizer, num_images=num_style_imgs)
    data_test = RIMES_data(data_test, input_size=input_size, tokenizer=tokenizer, num_images=num_style_imgs)

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(data_valid, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=False)

    if args.train:
        htr_model = Puigcerver(input_size=input_size, d_model=tokenizer.vocab_size)
        optimizer = torch.optim.RMSprop(htr_model.parameters(), lr=0.0003, momentum=0.9)

        trainer = HTRtrainer(htr_model, optimizer=optimizer)
        trainer.train_model(train_loader=train_loader, valid_loader=valid_loader, epochs=(0, 5))











