import os
import string

def read_rimes(folder, max_word_len, synth):
    """Get data paths and labels (with max_word_len) of images in folder."""
    if "iam_gan" in folder:
        partitions = ['train', 'valid', 'test']
    else:
        partitions = ['train', 'valid', 'test', 'oov_train', 'oov_valid', 'oov_test']
    dataset = {}
    wid_dict = {}
    for partition in partitions:
        lens = {}
        print(partition)
        dataset[partition] = []
        wid_dict[partition] = {}

        text_file = os.path.join(folder, f"superHTR.ground_truth_{partition}_filtered.txt") if "iam_gan" in folder else os.path.join(folder, f"ground_truth_{partition}_filtered.txt")
        # text_file = os.path.join(folder, f"ground_truth_{partition}_filtered.txt")

        with open(text_file, encoding='utf-8') as data_file:
            lines = data_file.read().splitlines()

        for line in lines:
            line_split = line.split(' ')
            wid = None

            if not set(string.digits).isdisjoint(set(line_split[-1])):
                wid = line_split[-1]
                line_split = line_split[:-1]
                
            if len(line_split) > 2:
                img_path, gt_label = line_split[0], ''.join(line_split[1:])
            else:
                img_path, gt_label = line_split[0], line_split[1]
                
            if len(gt_label) > max_word_len:
                continue

            if len(gt_label) in lens.keys():
                lens[len(gt_label)] += 1
            else:
                lens[len(gt_label)] = 1
            
            img_path = img_path.replace("/", "\\")

            if wid is not None:
                if wid not in wid_dict[partition].keys():
                    wid_dict[partition][wid] = []
                
                wid_dict[partition][wid].append((img_path, gt_label))
            
            if synth:
                img_path = img_path.split('.png')[0] + "_synth.png"

            # img_path = os.path.join(folder, partition, img_path)
            dataset[partition].append((img_path, gt_label, wid))
        print(f"number of words in {partition}: {len(dataset[partition])}")
        print(f"number of wids in {partition}: {len(wid_dict[partition].keys())}")
    
        lens = dict(sorted(lens.items()))

        print(f"Number of words per word length")
        for key in lens.keys():
            print(key, lens[key])

    # Adding training samples to wid dictionaries for validation and testing sets
    for wid in wid_dict['train'].keys():
        if wid in wid_dict['valid'].keys():
            wid_dict['valid'][wid] += wid_dict['train'][wid]

        if wid in wid_dict['test'].keys():
            wid_dict['test'][wid] += wid_dict['train'][wid]

        if "iam_gan" in folder:
            continue

        if wid in wid_dict['oov_valid'].keys():
            wid_dict['oov_valid'][wid] += wid_dict['train'][wid]
        
        if wid in wid_dict['oov_train'].keys():
            wid_dict['oov_train'][wid] += wid_dict['train'][wid]
        
        if wid in wid_dict['oov_test'].keys():
            wid_dict['oov_test'][wid] += wid_dict['train'][wid]

    if "iam_gan" in folder:
        return dataset['train'], dataset['valid'],  dataset['test'], wid_dict['train'], wid_dict['valid'], wid_dict['test']
    else:
        return dataset['train'], dataset['valid'], dataset['test'], dataset['oov_train'], dataset['oov_valid'], dataset['oov_test'], \
            wid_dict['train'], wid_dict['valid'], wid_dict['test'], wid_dict['oov_train'], wid_dict['oov_valid'], wid_dict['oov_test']

def read_iam_subset(folder, max_word_len, n_fold, synth=False):
    """Get data paths and labels (with max_word_len) of images in folder."""
    partitions = [f"train-{n_fold}", f"valid-{n_fold}", 'test']
    dataset = {}
    wid_dict = {}
    for partition in partitions:
        lens = {}
        print(partition)
        dataset[partition] = []
        wid_dict[partition] = {}

        if synth:
            text_file = os.path.join(folder, f"ground_truth_{partition}_synth.txt")
        else:
            text_file = os.path.join(folder, f"ground_truth_{partition}.txt")

        with open(text_file, encoding='utf-8') as data_file:
            lines = data_file.read().splitlines()

        for line in lines:
            line_split = line.split(' ')
            wid = None

            if not set(string.digits).isdisjoint(set(line_split[-1])):
                wid = line_split[-1]
                line_split = line_split[:-1]
                
            if len(line_split) > 2:
                img_path, gt_label = line_split[0], ''.join(line_split[1:])
            else:
                img_path, gt_label = line_split[0], line_split[1]
            
            # if "aug" in img_path:
            #     continue
                
            if len(gt_label) > max_word_len:
                continue

            if len(gt_label) in lens.keys():
                lens[len(gt_label)] += 1
            else:
                lens[len(gt_label)] = 1
            
            img_path = img_path.replace("/", "\\")

            if wid is not None:
                if wid not in wid_dict[partition].keys():
                    wid_dict[partition][wid] = []
                
                if synth:
                    img_path_real = img_path.split('_synth')[0] + ".png"
                    print(img_path_real)
                    wid_dict[partition][wid].append((img_path_real, gt_label))
                else:
                    wid_dict[partition][wid].append((img_path, gt_label))

            # img_path = os.path.join(folder, partition, img_path)
            dataset[partition].append((img_path, gt_label, wid))
        print(f"number of words in {partition}: {len(dataset[partition])}")
        print(f"number of wids in {partition}: {len(wid_dict[partition].keys())}")
    
        lens = dict(sorted(lens.items()))

        print(f"Number of words per word length")
        for key in lens.keys():
            print(key, lens[key])

    return dataset[f"train-{n_fold}"], dataset[f"valid-{n_fold}"], dataset['test'], wid_dict[f"train-{n_fold}"], wid_dict[f"valid-{n_fold}"], wid_dict['test']
            

