import os, sys
import torch
import glob
from torch import optim
import numpy as np
import time
import argparse
from data.tokenizer import Tokenizer
import string
from data.reader import read_rimes, read_iam_subset
import data.preproc
# from data.preproc import preproc_iam, preproc_rimes
from network.models import Puigcerver, Puigcerver_Dropout, Puigcerver_supervised
from trainer import HTRtrainer
from data.data_loader import RIMES_data
import editdistance
from torchaudio.models.decoder import ctc_decoder
from tqdm import tqdm
from network.gen_model.gen_model import GenModel_FC
from sklearn.model_selection import StratifiedKFold



if __name__ == "__main__":
    max_text_length = 25
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--self_supervised", action="store_true", default=False)
    parser.add_argument("--subset" , action="store_true", default=False)
    parser.add_argument("--fold" , type=int, default=0)

    parser.add_argument("--preproc", action="store_true", default=False)

    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--valid", action="store_true", default=False)
    parser.add_argument("--test_supervised", action="store_true", default=False)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument('--start_epoch', type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--loss", type=str, default="ctc")
    parser.add_argument("--pretrained", action="store_true", default=False)

    parser.add_argument("--vgg_layer", type=int, default=9)
    parser.add_argument("--max_word_len", type=int, default=max_text_length)

    args = parser.parse_args()

    if args.subset:
        dataset_path = os.path.join("..", "subset", args.dataset, "words")
    else:
        dataset_path = os.path.join("..", "data", args.dataset, "words")


        # preprocess data: resizing
        if args.preproc:
            print("Preparing data...")
            path_from = os.path.join("..", "raw", args.dataset)
            os.makedirs(dataset_path, exist_ok=True)
            getattr(data.preproc, f"preproc_{args.dataset}")(path_from, dataset_path)


    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_size = (64, 216, 1)
    num_style_imgs = 25 # num imgs for generator to extract style from

    charset_base = string.ascii_lowercase + string.ascii_uppercase
    
    tokenizer = Tokenizer(chars=charset_base, max_text_length=max_text_length, self_supervised=args.self_supervised)

    # definitions for generator
    num_classes = len(charset_base)
    tokens = {'GO_TOKEN': tokenizer.GO, 'PAD_TOKEN': tokenizer.PAD, 'UNK_TOKEN': tokenizer.UNK, "END_TOKEN":tokenizer.END}
    num_tokens = len(tokens.keys())
    vocab_size = num_classes + num_tokens
    
    # get data paths and labels (path, label)
    # TODO: change function name to sth general

    if args.subset:
        print("Fold", args.fold)
        data_train, data_valid, data_test, wid_train, wid_valid, wid_test = read_iam_subset(dataset_path, args.max_word_len, n_fold=args.fold)
    else:
        data_train, data_valid, data_test, wid_train, wid_valid, wid_test = read_rimes(dataset_path, args.max_word_len)

    data_train = RIMES_data(data_train, input_size=input_size, tokenizer=tokenizer, num_images=num_style_imgs, wids=wid_train)
    data_valid = RIMES_data(data_valid, input_size=input_size, tokenizer=tokenizer, num_images=num_style_imgs, wids=wid_valid)
    data_test = RIMES_data(data_test, input_size=input_size, tokenizer=tokenizer, num_images=num_style_imgs, wids=wid_test)

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(data_valid, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    if args.train:
        
        if not args.pretrained:
            
            if args.self_supervised:
                htr_model = Puigcerver(input_size=input_size, d_model=tokenizer.vocab_size)
                if args.subset:
                    folder_name = f"{args.dataset}/{args.loss}-{args.vgg_layer}-{args.max_word_len}char-fold{args.fold}" if "vgg" in args.loss else f"{args.dataset}/{args.loss}-{args.max_word_len}char-fold{args.fold}"
                else:
                    folder_name =f"{args.dataset}/{args.loss}-{args.vgg_layer}-{args.max_word_len}char-adam-lr001-lin" if "vgg" in args.loss else f"{args.dataset}/{args.loss}-{args.max_word_len}char-adam-lr001"
                model_name = f"./htr_models/{folder_name}/htr_model_self_supervised-{args.start_epoch - 1}.model"
                print(model_name)
                
            else:
                htr_model = Puigcerver_Dropout(input_size=input_size, d_model=tokenizer.vocab_size)
                folder_name = f"{args.dataset}/{args.loss}-{args.max_word_len}-chars-rms-lr0001-dense"
                model_name = f"./htr_models/{folder_name}/htr_model_supervised-{args.start_epoch}.model"
                print(model_name)

            if os.path.exists(model_name):
                print("loading model: ", model_name)
                htr_model.load_state_dict(torch.load(model_name)) #load
        else:
            htr_model = Puigcerver_Dropout(input_size=input_size, d_model=tokenizer.vocab_size + 1)
            model_name = f"./htr_models/iam/ctc/htr_model_supervised-40.model"
            folder_name = f"{args.dataset}/{args.loss}-{args.vgg_layer}-{args.max_word_len}chars-adam-lr001" if "vgg" in args.loss else f"{args.loss}-{args.max_word_len}chars-adam-lr001"
            if os.path.exists(model_name):
                print("loading pretrained model model: ", model_name)
                htr_model.load_state_dict(torch.load(model_name)) #load
                # htr_model.replace_head(tokenizer.vocab_size)

        
        if args.self_supervised:
            optimizer = torch.optim.Adam(htr_model.parameters(), lr=0.001)
            #optimizer = torch.optim.RMSprop(htr_model.parameters(), lr=0.001, momentum=0.9)
            # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1.0, 0.25, 50)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)
        else:
            optimizer = torch.optim.Adam(htr_model.parameters(), lr=0.0001)
            #optimizer = torch.optim.RMSprop(htr_model.parameters(), lr=0.0001, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5, verbose=True)

        trainer = HTRtrainer(htr_model, optimizer=optimizer, lr_scheduler=scheduler, device=device, tokenizer=tokenizer, loss_name=args.loss, self_supervised=args.self_supervised, folder_name=folder_name, vgg_layer=args.vgg_layer)
        trainer.train_model(train_loader=train_loader, valid_loader=valid_loader, epochs=(args.start_epoch, args.epochs))

    elif args.test or args.valid:
        if args.test:
            data_loader = test_loader
        elif args.valid:
            data_loader = valid_loader

       
        folder_name = f"{args.dataset}/{args.loss}-{args.max_word_len}-chars-rms-lr0001"
        if args.self_supervised:
            htr_model = Puigcerver(input_size=input_size, d_model=tokenizer.vocab_size).cuda()
            model_name = f"./htr_models/{folder_name}/htr_model_self_supervised-{args.start_epoch}.model"
            # from torchaudio, so uses a silent token
            beam_search_decoder = ctc_decoder(lexicon=None,
                tokens=[char for char in tokenizer.chars + "|" + '#'],
                nbest=1,
                beam_size=50,
                blank_token = "#",
                sil_token = "|"
            )
        
        else:
            #model_name = f"./htr_models/{folder_name}/htr_model_supervised-{args.start_epoch}.model"
            htr_model = Puigcerver_Dropout(input_size=input_size, d_model=tokenizer.vocab_size).cuda()
            model_name = f"./htr_models/iam/ctc/htr_model_supervised-40.model"
            # from torchaudio, so uses a silent token
            beam_search_decoder = ctc_decoder(lexicon=None,
                tokens=[char for char in tokenizer.chars + "|"],
                nbest=1,
                beam_size=50,
                blank_token = "#",
                sil_token = "|"
            )
        
        if os.path.exists(model_name):
                print("Loading model: ", model_name)
                htr_model.load_state_dict(torch.load(model_name)) #load
                htr_model.eval()
        else:
            print("Model not found")
            exit()


        cer = 0
        wer = 0
        total = 0
        for batch in tqdm(data_loader):
            imgs, _, gt_labels = batch
            imgs = imgs.to(device)
            gt_labels = gt_labels.to(device)

            y_pred = htr_model(imgs)

            y_pred_soft = torch.nn.functional.softmax(y_pred, dim=2).detach().cpu()
            y_pred_max = torch.max(y_pred_soft, dim=2).indices
            gt_labels = gt_labels.detach().cpu().numpy()
            gt_labels = [tokenizer.decode(label) for label in gt_labels]
            y_pred = [tokenizer.decode(label) for label in y_pred_max]
            
            beam_search = beam_search_decoder(y_pred_soft)
            y_pred_bs = [tokenizer.decode(label[0].tokens * (label[0].tokens < 57)) for label in beam_search]

            for (pd, gt) in zip(y_pred_bs, gt_labels):
                pd_cer, gt_cer = list(pd), list(gt)
                dist = editdistance.eval(pd_cer, gt_cer)
                
                cer += dist / (max(len(pd_cer), len(gt_cer)))

                pd_wer, gt_wer = pd.split(), gt.split()
                dist = editdistance.eval(pd_wer, gt_wer)
                wer += dist / (max(len(pd_wer), len(gt_wer)))
            total += len(gt_labels)

        print(f"CER: {cer / total}, WER: {wer / total}")   
    elif args.test_supervised:
        charset_base = string.ascii_lowercase + string.ascii_uppercase
    
        tokenizer = Tokenizer(chars=charset_base, max_text_length=max_text_length, self_supervised=True)

        htr_model = Puigcerver_supervised(input_size=input_size, d_model=tokenizer.vocab_size).cuda()
        model_name = f"./network/htr_model_supervised-ce-40.model"
        
        if os.path.exists(model_name):
                print("Loading model: ", model_name)
                htr_model.load_state_dict(torch.load(model_name)) #load
                htr_model.eval()
        else:
            print("Model not found")
            exit()

        beam_search_decoder = ctc_decoder(lexicon=None,
                tokens=[char for char in tokenizer.chars + "|" + '#'],
                nbest=1,
                beam_size=50,
                blank_token = "#",
                sil_token = "|"
            )

        # gen_model = GenModel_FC(tokenizer.maxlen, tokenizer.vocab_size - 1, tokenizer.PAD)
        # # self.gen_model.load_state_dict(torch.load('./network/gen_model/gen_model-15all.model')) #load
        # gen_model.load_state_dict(torch.load('./network/gen_model/gen_model-25half.model'))
        # gen_model.eval()
        # gen_model.to(device)

        cer = 0
        wer = 0
        total = 0
        for batch in tqdm(train_loader):
            imgs, gen_imgs, gt_labels, _ = batch
            imgs = imgs.to(device)
            gt_labels = gt_labels.to(device)

            imgs = imgs.to(device)
            gen_imgs = gen_imgs.to(device).squeeze(2)
            gt_labels = gt_labels.to(device)

            # gt_labels_1hot = gt_labels.long()
            # gt_labels_1hot = torch.nn.functional.one_hot(gt_labels_1hot, 56).float()

            # synth_imgs = gen_model(gen_imgs, gt_labels_1hot)

            y_pred = htr_model(imgs)

            y_pred_soft = torch.nn.functional.softmax(y_pred, dim=2).detach().cpu()
            y_pred_max = torch.max(y_pred_soft, dim=2).indices
            gt_labels = gt_labels.detach().cpu().numpy()
            gt_labels = [tokenizer.decode(label) for label in gt_labels]
            y_pred = [tokenizer.decode(label) for label in y_pred_max]

            # beam_search = beam_search_decoder(y_pred_soft)
            # y_pred_bs = [tokenizer.decode(label[0].tokens * (label[0].tokens < 57)) for label in beam_search]
            
            for (pd, gt) in zip(y_pred, gt_labels):
                pd_cer, gt_cer = list(pd), list(gt)
                dist = editdistance.eval(pd_cer, gt_cer)
                
                cer += dist / (max(len(pd_cer), len(gt_cer)))

                pd_wer, gt_wer = pd.split(), gt.split()
                dist = editdistance.eval(pd_wer, gt_wer)
                wer += dist / (max(len(pd_wer), len(gt_wer)))
            total += len(gt_labels)

        print(f"CER: {cer / total}, WER: {wer / total}")   

