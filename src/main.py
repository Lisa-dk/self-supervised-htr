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
import data.preproc
# from data.preproc import preproc_iam, preproc_rimes
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

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument('--start_epoch', type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--loss", type=str, default="ctc")
    parser.add_argument("--pretrained", action="store_true", default=False)
    args = parser.parse_args()

    dataset_path = os.path.join("..", "data", args.dataset, "words")

    if args.preproc:
        print("Preparing data...")
        path_from = os.path.join("..", "raw", args.dataset)
        os.makedirs(dataset_path, exist_ok=True)
        getattr(data.preproc, f"preproc_{args.dataset}")(path_from, dataset_path)


    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_size = (64, 216, 1)
    num_style_imgs = 15

    charset_base = string.ascii_lowercase + string.ascii_uppercase
    max_text_length = 25
    tokenizer = Tokenizer(chars=charset_base, max_text_length=max_text_length, self_supervised=args.self_supervised)

    num_classes = len(charset_base)
    tokens = {'GO_TOKEN': tokenizer.GO, 'PAD_TOKEN': tokenizer.PAD, 'UNK_TOKEN': tokenizer.UNK, "END_TOKEN":tokenizer.END}
    num_tokens = len(tokens.keys())
    vocab_size = num_classes + num_tokens
    
    data_train, data_valid, data_test = read_rimes(dataset_path)

    data_train = RIMES_data(data_train, input_size=input_size, tokenizer=tokenizer, num_images=num_style_imgs)
    data_valid = RIMES_data(data_valid, input_size=input_size, tokenizer=tokenizer, num_images=num_style_imgs)
    data_test = RIMES_data(data_test, input_size=input_size, tokenizer=tokenizer, num_images=num_style_imgs)

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(data_valid, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    if args.train:
        
        if not args.pretrained:
            htr_model = Puigcerver(input_size=input_size, d_model=tokenizer.vocab_size)
            if args.self_supervised:
                model_name = f"./htr_models/{args.loss}/htr_model_self_supervised-{args.start_epoch}.model"
                
            else:
                model_name = f"./htr_models/{args.loss}/htr_model_self_supervised-{args.start_epoch}.model"

            if os.path.exists(model_name):
                print("loading model: ", model_name)
                htr_model.load_state_dict(torch.load(model_name)) #load
        else:
            htr_model = Puigcerver(input_size=input_size, d_model=tokenizer.vocab_size + 1)
            model_name = f"./htr_models/iam/htr_model_supervised.model"
            if os.path.exists(model_name):
                print("loading pretrained model model: ", model_name)
                htr_model.load_state_dict(torch.load(model_name)) #load
                htr_model.replace_head(tokenizer.vocab_size)

        
        if args.self_supervised:
            optimizer = torch.optim.Adam(htr_model.parameters(), lr=0.005)
            #optimizer = torch.optim.RMSprop(htr_model.parameters(), lr=0.01, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, 50, 0.5)
        else:
            optimizer = torch.optim.Adam(htr_model.parameters(), lr=0.0001)
            #optimizer = torch.optim.RMSprop(htr_model.parameters(), lr=0.0001, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1.0, 0.1, 100)

        trainer = HTRtrainer(htr_model, optimizer=optimizer, lr_scheduler=scheduler, device=device, tokenizer=tokenizer, loss_name=args.loss, self_supervised=args.self_supervised)
        trainer.train_model(train_loader=train_loader, valid_loader=valid_loader, epochs=(args.start_epoch, args.epochs))

    if args.test:
        pass











