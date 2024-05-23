import os
import string, random

def read_data(folder):
    """Get data paths and labels (with max_word_len) of images in folder."""
    partitions = ['train', 'valid', 'test']
    dataset = {}

    for partition in partitions:
        lens = {}
        print(partition)
        dataset[partition] = []
        text_file = os.path.join(folder, f"sia_ground_truth_{partition}_filtered.txt")

        with open(text_file, encoding='utf-8') as data_file:
            lines = data_file.read().splitlines()

        for line in lines:
            line_split = line.split(' ')
            # print(line_split)
            if len(line_split) >= 5:
                img_path1, img_path2, label, word1, word2 = line_split[0], line_split[1], line_split[2], line_split[3], line_split[4]
            else:
                img_path1, img_path2, label, word1 = line_split[0], line_split[1], line_split[2], line_split[3]
                word2 = ""

            # rnd = random.random()
            # typ_synth = img_path2.split("_")[-1]
            # if rnd >= 0.5:
            #     if typ_synth == "same.png":
            #         continue
            typ_synth = img_path2.split("_")[-1]
            # if typ_synth == "samed.png" or typ_synth == "nonesense.png":
                # continue
            img_path1 = img_path1.replace("/", "\\")
            img_path2 = img_path2.replace("//", "\\")
            img_path2 = img_path2.replace("/", "\\")

            # img_path = os.path.join(folder, partition, img_path)
            dataset[partition].append((img_path1[3:], img_path2[3:], label, word1, word2)) # skip first '..//'
        print(f"number of pairs in {partition}: {len(dataset[partition])}")
    
        lens = dict(sorted(lens.items()))

    return dataset['train'], dataset['valid'], dataset['test']
            

