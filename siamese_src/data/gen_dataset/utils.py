import random, cv2
import numpy as np
from sklearn.model_selection import train_test_split

def get_char_weights(charset_base, train_data):
    character_freq_dict = {charset_base[idx] : 0 for idx in range(len(charset_base))}

    for tup in train_data:
        mg_path, gt_label, wid = tup[0], tup[1], tup[2]
        for char in list(gt_label):
            character_freq_dict[char] += 1

    total = sum(character_freq_dict.values())

    for key in character_freq_dict.keys():
        character_freq_dict[key] /= total

    return [character_freq_dict[char] for char in charset_base]

def get_unique_words(wids):
    unique_word_counts = []
    all_words = []

    word_len_dict = {}

    for wid in sorted(wids.keys()):
        words = list(set([wids[wid][idx][1] for idx in range(len(wids[wid]))]))
        all_words += words
        unique_word_counts.append(len(words))
    unique_word_counts = sorted(unique_word_counts, reverse=True)
    len(set(all_words))

    for word in sorted(set(all_words)):
        if len(word) not in word_len_dict.keys():
            word_len_dict[len(word)] = []
        word_len_dict[len(word)].append(word)
    return word_len_dict, all_words

def get_oov_words(word_len_dict):
    random.seed(42)
    oov_test_words = []
    oov_valid_words = []

    for word_len in sorted(word_len_dict.keys()):
        words = word_len_dict[word_len]
        print(word_len, len(words), len(words)// 10)
        _, oov_test_valid_words = train_test_split(words, test_size=0.1, shuffle=True, random_state=42)
        # oov_test_valid_words = words[:len(words) // 10]
        oov_test_words += oov_test_valid_words[:len(oov_test_valid_words)//2]
        oov_valid_words += oov_test_valid_words[len(oov_test_valid_words)//2:]
        # print(oov_test_words)
    return oov_test_words, oov_valid_words

def get_splits(wids, oov_test_words, oov_valid_words):
    oov_test_data = []
    oov_valid_data = []

    train_data = []
    valid_data = []
    test_data = []

    for wid in wids.keys():
        wid_data = wids[wid]
        train_wid = []
        test_wid = []
        wid_valid = []

        for tup in wid_data:
            if tup[1] in oov_test_words:
                oov_test_data.append((tup[0], tup[1], wid))
            elif tup[1] in oov_valid_words:
                oov_valid_data.append((tup[0], tup[1], wid))
            else:
                train_wid.append((tup[0], tup[1], wid))
        
        split_test_idx = len(train_wid) // 5
        if split_test_idx > 0:
            wid_test = train_wid[:split_test_idx]
            train_wid = train_wid[split_test_idx:]
        
        split_valid_idx = len(train_wid) // 5
        if split_valid_idx > 1:
            wid_valid = train_wid[:split_valid_idx]
            train_wid = train_wid[split_valid_idx:]
        
        if len(wid_test) > 0:
            test_data += wid_test
        
        if len(wid_valid) > 0:
            valid_data += wid_valid
        
        train_data += train_wid

    
    return train_data, valid_data, test_data, oov_valid_data, oov_test_data

def prep_data_multiproc(data, num_processes):
    new_data = []
    size = len(data) // num_processes
    for i in range(num_processes ):
        start_idx = i*size
        if i == num_processes - 1:
            new_data.append(data[start_idx:])
        else:
            new_data.append(data[start_idx:start_idx+size])
    return new_data