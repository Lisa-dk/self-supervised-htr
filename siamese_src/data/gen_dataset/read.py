import os, string

def read_data(folder, max_word_len):
    """Get data paths and labels (with max_word_len) of images in folder."""
    dataset = {}
    wid_dict = {}
    lens = {}
    dataset = []
    wid_dict = {}

    text_file = os.path.join(folder, f"ground_truth_filtered.txt")

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
            if wid not in wid_dict.keys():
                wid_dict[wid] = []
            
            wid_dict[wid].append((img_path, gt_label))

        # img_path = os.path.join(folder, partition, img_path)
        dataset.append((img_path, gt_label, wid))
    print(f"number of words: {len(dataset)}")
    print(f"number of wids: {len(wid_dict.keys())}")

    lens = dict(sorted(lens.items()))

    print(f"Number of words per word length")
    for key in lens.keys():
        print(key, lens[key])

    return dataset, wid_dict