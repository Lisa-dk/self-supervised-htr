import os, sys
import torch
import glob
from torch import optim
import numpy as np
import time
import argparse
from data.tokenizer import Tokenizer
import string
from data.reader import read_iam, read_iam_subset
import data.preproc
# from data.preproc import preproc_iam, preproc_rimes
from network.models import Puigcerver, Puigcerver_supervised
from trainer import HTRtrainer
from data.data_loader import IAM_data
import editdistance
from torchaudio.models.decoder import ctc_decoder
from tqdm import tqdm
from network.gen_model.gen_model import GenModel_FC
from sklearn.model_selection import StratifiedKFold
sys.path.append('../../')
sys.path.append('../src')
sys.path.append('../../GANwriting')
sys.path.append('../GANwriting/corpora_english')



if __name__ == "__main__":
    max_text_length = 10
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--self_supervised", action="store_true", default=False)
    parser.add_argument("--subset" , action="store_true", default=False)
    parser.add_argument("--synth", action="store_true", default=False)
    parser.add_argument("--oov", action="store_true", default=False)
    parser.add_argument("--fold" , type=int, default=0)

    parser.add_argument("--preproc", action="store_true", default=False)

    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--valid", action="store_true", default=False)
    parser.add_argument("--beam_search", action="store_true", default=False)

    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument('--start_epoch', type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--loss", type=str, default="ctc")
    parser.add_argument("--pretrained", action="store_true", default=False)

    parser.add_argument("--vgg_layer", type=int, default=9)
    parser.add_argument("--max_word_len", type=int, default=max_text_length)
    parser.add_argument("--lr", type=float, default=0.001)

    args = parser.parse_args()

    if args.subset:
        dataset_path = os.path.join("..", "subset", args.dataset, "words")
    else:
        dataset_path = os.path.join("..", "data", args.dataset, "words")


        # preprocess data: resizing
        if args.preproc:
            print("Preparing data...")
            os.makedirs(dataset_path, exist_ok=True)

            path_from = os.path.join("..", "raw", args.dataset)
            if args.dataset == "iam_gan":
                path_from = os.path.join("..", "GANwriting_adapted") # Change path to location of GANwriting
                getattr(data.preproc, f"preproc_iam")(path_from, dataset_path)
            else:
                getattr(data.preproc, f"preproc_{args.dataset}")(path_from, dataset_path)
            
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_size = (64, 216, 1)
    num_style_imgs = 25 # num imgs for generator to extract style from

    charset_base = string.ascii_lowercase + string.ascii_uppercase
    max_padding = 25 # set to num timesteps (i.e. width final feature map of model) 

    if args.loss == "ctc":
        tokenizer = Tokenizer(chars=charset_base, max_text_length=max_padding, ctc=True) # ctc loss requires an extra blank symbol
    else:
        tokenizer = Tokenizer(chars=charset_base, max_text_length=max_padding, ctc=False)

    # definitions for generator
    num_classes = len(charset_base)
    tokens = {'GO_TOKEN': tokenizer.GO, 'PAD_TOKEN': tokenizer.PAD, 'UNK_TOKEN': tokenizer.UNK, "END_TOKEN":tokenizer.END}
    num_tokens = len(tokens.keys())
    vocab_size = num_classes + num_tokens
    
    # get data paths and labels (path, label)
    if args.subset:
        # Can be used for cross-validation, training/evaluation per predefined fold.
        print("Fold", args.fold)
        data_train, data_valid, data_test, wid_train, wid_valid, wid_test = read_iam_subset(dataset_path, args.max_word_len, n_fold=args.fold, synth=args.synth)
        data_test = IAM_data(data_test, input_size=input_size, tokenizer=tokenizer, num_images=num_style_imgs, wids=wid_test)
        test_loader = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
    elif args.dataset == "iam_gan":
        # To train the supervised HTR on IAM-GEN-SIA
        data_train, data_valid, data_test, wid_train, wid_valid, wid_test = read_iam(dataset_path, args.max_word_len, synth=args.synth)
        data_test = IAM_data(data_test, input_size=input_size, tokenizer=tokenizer, num_images=num_style_imgs, wids=wid_test)
        test_loader = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
    else:
        # (Self-)supervised HTR on IAM-HTR
        data_train, data_valid, data_test, oov_data_train, oov_data_valid, oov_data_test, wid_train, wid_valid, wid_test, oov_wid_train, oov_wid_valid, oov_wid_test = read_iam(dataset_path, args.max_word_len, synth=args.synth)
        
        oov_data_train = IAM_data(oov_data_train, input_size=input_size, tokenizer=tokenizer, num_images=num_style_imgs, wids=oov_wid_train)
        oov_train_loader = torch.utils.data.DataLoader(oov_data_train, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
        
        oov_data_valid = IAM_data(oov_data_valid, input_size=input_size, tokenizer=tokenizer, num_images=num_style_imgs, wids=oov_wid_valid)
        oov_valid_loader = torch.utils.data.DataLoader(oov_data_valid, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
        
        data_test = IAM_data(data_test, input_size=input_size, tokenizer=tokenizer, num_images=num_style_imgs, wids=wid_test)
        test_loader = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

        oov_data_test = IAM_data(oov_data_test, input_size=input_size, tokenizer=tokenizer, num_images=num_style_imgs, wids=oov_wid_test)
        oov_test_loader = torch.utils.data.DataLoader(oov_data_test, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    # Training on IAM-HTR-OOV or -IV
    if args.oov:
        train_loader = oov_train_loader
    else:
        data_train = IAM_data(data_train, input_size=input_size, tokenizer=tokenizer, num_images=num_style_imgs, wids=wid_train)
        train_loader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    
    data_valid = IAM_data(data_valid, input_size=input_size, tokenizer=tokenizer, num_images=num_style_imgs, wids=wid_valid)
    valid_loader = torch.utils.data.DataLoader(data_valid, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
    

    if args.train:
        
        if not args.pretrained:
            
            if args.self_supervised:
                htr_model = Puigcerver(input_size=input_size, d_model=tokenizer.vocab_size)
                if args.subset:
                    folder_name = f"{args.dataset}/{args.loss}-{args.vgg_layer}-{args.max_word_len}char-fold{args.fold}" if "vgg" in args.loss else f"{args.dataset}/{args.loss}-{args.max_word_len}char-fold{args.fold}"
                else:
                    folder_name =f"{args.dataset}/{args.loss}-{args.vgg_layer}-{args.max_word_len}char-gen3000-adam-lr0003-lrplat" if "vgg" in args.loss else f"{args.dataset}/{args.loss}-{args.max_word_len}char-gen3000-adam-lr0003-lrplat"
                
                if args.synth:
                    folder_name = folder_name + "-synth"
                
                if args.oov:
                    folder_name = folder_name + "-oov"
                
                model_name = f"./htr_models/{folder_name}/htr_model_self_supervised-{args.start_epoch}.model"
                print(model_name)
                
            else:
                htr_model = Puigcerver_supervised(input_size=input_size, d_model=tokenizer.vocab_size)
                folder_name = f"{args.dataset}/{args.loss}-{args.max_word_len}char-gen3000-maxpool-lrplat"
                model_name = f"./htr_models/{folder_name}/htr_model_supervised-{args.start_epoch - 1}.model"
                print(model_name)
            
            if args.synth:
                folder_name = folder_name + "-synth"
            
            if args.oov:
                folder_name = folder_name + "-oov"

            # Load existing model (for testing or continued training)
            # Change line 140 to Puigcerver_supervised if done for pretrained settings
            if os.path.exists(model_name):
                print("loading model: ", model_name)
                htr_model.load_state_dict(torch.load(model_name)) 
        else:
            # Training in the pretrained setting
            htr_model = Puigcerver_supervised(input_size=input_size, d_model=tokenizer.vocab_size)
            model_name = f"./htr_models/iam_gan/ce-10char-maxpool/htr_model_supervised-73.model"
            folder_name =f"{args.dataset}/{args.loss}-{args.vgg_layer}-{args.max_word_len}char-gen3000-adam-lr0001-lrplat-pretr" if "vgg" in args.loss else f"{args.dataset}/{args.loss}-{args.max_word_len}char-gen3000-adam-lr0001-lrplat-pretr"
            
            if args.synth:
                folder_name = folder_name + "-synth"
            
            if args.oov:
                folder_name = folder_name + "-oov"
            
            if os.path.exists(model_name):
                print("Loading pretrained model: ", model_name)
                print(folder_name)
                htr_model.load_state_dict(torch.load(model_name))
        
        # Optimizer and lr schedulers
        if args.self_supervised:
            optimizer = torch.optim.Adam(htr_model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)
        else:
            optimizer = torch.optim.Adam(htr_model.parameters(), lr=0.0001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5, verbose=True)

        trainer = HTRtrainer(htr_model, optimizer=optimizer, lr_scheduler=scheduler, device=device, tokenizer=tokenizer, loss_name=args.loss, self_supervised=args.self_supervised, folder_name=folder_name, vgg_layer=args.vgg_layer)
        
        # Model training
        if args.dataset == "iam_gan" or args.subset:
            trainer.train_model(train_loader=train_loader, valid_loader=valid_loader, oov_valid_loader=None, epochs=(args.start_epoch, args.epochs))
        else:
            trainer.train_model(train_loader=train_loader, valid_loader=valid_loader, oov_valid_loader=oov_valid_loader, epochs=(args.start_epoch, args.epochs))
    
    elif args.test or args.valid:
        if args.test:
            data_loader = test_loader # for oov results: change to oov_test_loader
        elif args.valid:
            data_loader = valid_loader # for oov results: change to oov_valid_loader
        beam_search = False
       
        folder_name = f"{args.dataset}/{args.loss}-{args.vgg_layer}-{args.max_word_len}char-gen3000-adam-lr0003-lrplat-pretr-synth-oov" if "vgg" in args.loss else f"{args.dataset}/{args.loss}-{args.max_word_len}char-gen3000-maxpool-synth"
        if args.self_supervised:
            htr_model = Puigcerver_supervised(input_size=input_size, d_model=tokenizer.vocab_size).cuda()
            model_name = f"./htr_models/{folder_name}/htr_model_self_supervised-{args.start_epoch}.model"
            
            # from torchaudio, so uses a silent token "|"
            beam_search_decoder = ctc_decoder(lexicon=None,
                tokens=[char for char in tokenizer.chars + "|" + '#'],
                nbest=1,
                beam_size=50,
                blank_token = "#",
                sil_token = "|"
            )
        
        else:
            folder_name = f"{args.dataset}/{args.loss}-{args.max_word_len}char-maxpool"
            htr_model = Puigcerver_supervised(input_size=input_size, d_model=tokenizer.vocab_size).cuda()
            model_name = f"./htr_models/{folder_name}/htr_model_supervised-{args.start_epoch}.model"
            # from torchaudio, so uses a silent token
            if args.loss == "ctc":
                beam_search_decoder = ctc_decoder(lexicon=None,
                    tokens=[char for char in tokenizer.chars + "|"],
                    nbest=1,
                    beam_size=50,
                    blank_token = "#",
                    sil_token = "|"
                )
            else:
                beam_search_decoder = ctc_decoder(lexicon=None,
                tokens=[char for char in tokenizer.chars + "|" + '#'],
                nbest=1,
                beam_size=50,
                blank_token = "#",
                sil_token = "|"
                )
        
        print(model_name)
        # Load model
        if os.path.exists(model_name):
            print("Loading model: ", model_name)
            htr_model.load_state_dict(torch.load(model_name)) 
            htr_model.eval()
        else:
            print("Model not found")
            exit()

        cer = 0
        wer = 0
        total = 0
        for batch in tqdm(data_loader):
            imgs, _, gt_labels, _, _ = batch
            imgs = imgs.to(device)
            gt_labels = gt_labels.to(device)

            y_pred = htr_model(imgs)

            y_pred_soft = torch.nn.functional.softmax(y_pred, dim=2).detach().cpu()
            y_pred_max = torch.max(y_pred_soft, dim=2).indices
            gt_labels = gt_labels.detach().cpu().numpy()
            gt_labels = [tokenizer.decode(label) for label in gt_labels]
            y_pred = [tokenizer.decode(label) for label in y_pred_max]

            if beam_search:
                beam_search = beam_search_decoder(y_pred_soft)
                y_pred = [tokenizer.decode(label[0].tokens * (label[0].tokens < 57)) for label in beam_search]

            for (pd, gt) in zip(y_pred, gt_labels):
                pd_cer, gt_cer = list(pd), list(gt)
                dist = editdistance.eval(pd_cer, gt_cer)
                
                cer += dist / (max(len(pd_cer), len(gt_cer)))

                pd_wer, gt_wer = pd.split(), gt.split()
                dist = editdistance.eval(pd_wer, gt_wer)
                wer += dist / (max(len(pd_wer), len(gt_wer)))
            total += len(gt_labels)

        print(f"CER: {cer / total}, WER: {wer / total}")   

