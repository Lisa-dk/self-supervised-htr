import os, sys
print(os.listdir('../../'))
sys.path.append("../")
sys.path.append("../../../src/network")
sys.path.append("../../../src")
print(os.getcwd())
import torch
import glob
import string
import os, sys
import cv2
import numpy as np
from network.gen_model.gen_model import GenModel_FC
from random import choices
import random
import matplotlib.pyplot as plt
from preproc import preproc_iam
from torchvision import transforms
from tokenizer import Tokenizer
import multiprocessing
import time

from read import *
from utils import *
from edits import *

NUM_PROCESSES = 2

charset_base = string.ascii_lowercase + string.ascii_uppercase
max_text_length = 25
tokenizer = Tokenizer(chars=charset_base, max_text_length=max_text_length)

gen_model = GenModel_FC(tokenizer.maxlen, tokenizer.vocab_size, tokenizer.PAD).cuda()
gen_model.load_state_dict(torch.load('../../network/gen_model/gen_model-25half.model')) #load
gen_model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = (64, 216, 1)
num_style_imgs = 25 # num imgs for generator to extract style from
batch_size = 32
dataset = "iam"
dataset_path = os.path.join("..", "..", "..", "data", dataset, "words")
max_word_len = 10

transforms_ = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

data, wids = read_data(dataset_path, max_word_len)
wid_ids_keys = [key for key in wids.keys()]

preproc_bool = False
if preproc_bool:
    path_from = os.path.join("..", "..", "..", "handwriting-recognition", "raw", dataset)
    os.makedirs(dataset_path, exist_ok=True)
    preproc_iam(path_from, dataset_path)

word_len_dict, all_words = get_unique_words(wids)
oov_test_words, oov_valid_words = get_oov_words(word_len_dict)

print(len(oov_test_words), len(oov_valid_words),len(set(all_words)))

train_data, valid_data, test_data, oov_valid_data, oov_test_data = get_splits(wids, oov_test_words, oov_valid_words)
print(len(train_data), len(valid_data), len(test_data), len(oov_valid_data), len(oov_test_data))
print(train_data[0],  valid_data[0], test_data[0])

char_weights = get_char_weights(charset_base, train_data)
print(char_weights)

def copy_style_imgs_wid(single_img, wid):
    """Obtain different same-style images based on the wid"""
    style_paths = wids[wid]
    copy_paths = random.choices(style_paths, k=num_style_imgs - 1)

    final_img = [single_img[:]]

    for path in copy_paths:
        new_img = 255 - cv2.imread("..\\"+ path[0], cv2.IMREAD_GRAYSCALE) 
        new_img = transforms_(new_img)
        final_img = final_img + [new_img]

    final_img = np.stack(final_img, axis=0)

    return np.asarray(final_img, dtype="float32")

def generate_data(dataset):
    i = 0
    pairs = []
    for tup in dataset:
        inputs = []
        labels = []
        img_path, gt_label, wid = tup[0], tup[1], tup[2]
        img_path = "..\\" + img_path

        img = 255 - cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = transforms_(img)
        inputs.append(copy_style_imgs_wid(img, wid))
        labels.append(tokenizer.encode(gt_label))


        wid_diff = wid
        while wid_diff == wid:
            wid_diff = random.choice(wid_ids_keys)

        inputs.append(copy_style_imgs_wid(img, wid_diff))
        labels.append(tokenizer.encode(gt_label))

        if random.random() <= 0.5:
            rnd_sample = random.choice(dataset)
            while rnd_sample[1] == gt_label:
                rnd_sample = random.choice(dataset)
            nonesense_label = rnd_sample[1]
        else:
            nonesense_label = get_nonesense_label(char_weights)

            if random.random() <= 0.25:
                nonesense_label = random_multi_insert(nonesense_label, char_weights)

        inputs.append(copy_style_imgs_wid(img, wid))
        labels.append(tokenizer.encode(nonesense_label))
    
        edit_label = random_edits(gt_label, char_weights)
        while edit_label == gt_label:
            edit_label = random_edits(gt_label, char_weights)
        
        if random.random() <= 0.25:
            edit_label = random_multi_insert(edit_label, char_weights)
        inputs.append(copy_style_imgs_wid(img, wid))
        labels.append(tokenizer.encode(edit_label))

        inputs = torch.from_numpy(np.array(inputs)).squeeze(2).cuda()

        labels = torch.from_numpy(np.array(labels)).cuda().long()
        labels = torch.nn.functional.one_hot(labels, 56).float()
        # print(gt_label, nonesense_label, edit_label)

        synth_imgs = gen_model(inputs, labels).detach()
        synth_imgs = ((synth_imgs * 0.5) + 0.5) * 255
        synth_imgs = synth_imgs.cpu().numpy()

        new_img_path = img_path.split('.png')[0] # remove '.png
        
        cv2.imwrite(new_img_path + '_sames.png', synth_imgs[0][0])
        pairs.append(f"{img_path} {new_img_path}_sames.png 1 {gt_label} {gt_label} \n")

        cv2.imwrite(new_img_path + '_samed.png', synth_imgs[1][0])
        pairs.append(f"{img_path} {new_img_path}_samed.png 1 {gt_label} {gt_label} \n")

        cv2.imwrite(new_img_path + '_nonesense.png', synth_imgs[2][0])
        pairs.append(f"{img_path} {new_img_path}_nonesense.png 0 {gt_label} {nonesense_label}\n")

        cv2.imwrite(new_img_path + '_edits.png', synth_imgs[3][0])
        pairs.append(f"{img_path} {new_img_path}_edits.png 0 {gt_label} {edit_label}\n")

    return pairs

def write_pairs(data, fn):
    start = time.time()
    pairs = generate_data(data)
    print(time.time() - start)

    start = time.time()
    with open(fn, 'w') as f:
        for pair in pairs:
            f.write(pair)
    print(time.time() - start)

if __name__ == "__main__":

    # multi_train_data = prep_data_multiproc(train_data, NUM_PROCESSES)
    # multi_valid_data = prep_data_multiproc(valid_data, NUM_PROCESSES)
    # multi_test_data = prep_data_multiproc(test_data, NUM_PROCESSES)
    # multi_oov_valid_data = prep_data_multiproc(oov_valid_data, NUM_PROCESSES)
    # multi_oov_test_data = prep_data_multiproc(oov_test_data, NUM_PROCESSES)

    random.seed(42)
    tr_data_fn = "../../../data/iam/words/ground_truth_train_filtered.txt"
    test_data_fn = "../../../data/iam/words/ground_truth_test_filtered.txt"
    valid_data_fn = "../../../data/iam/words/ground_truth_valid_filtered.txt"
    oov_valid_data_fn = "../../../data/iam/words/ground_truth_valid_oov_filtered.txt"
    oov_test_data_fn = "../../../data/iam/words/ground_truth_test_oov_filtered.txt"

    write_pairs(train_data, tr_data_fn)
    write_pairs(valid_data, valid_data_fn)
    write_pairs(test_data, test_data_fn)
    write_pairs(oov_valid_data, oov_valid_data_fn)
    write_pairs(oov_test_data, oov_test_data_fn)
    
    
    
    # with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
    #     results = pool.map(generate_data, multi_oov_test_data)

    # # print(results)
    




