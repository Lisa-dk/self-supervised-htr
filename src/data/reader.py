import os

def read_rimes(folder, max_word_len):
    partitions = ['train', 'valid', 'test']
    dataset = {}
    lens = {}
    for partition in partitions:
        dataset[partition] = []

        text_file = os.path.join(folder, f"ground_truth_{partition}_filtered.txt")

        with open(text_file, encoding='utf-8') as data_file:
            lines = data_file.read().splitlines()

        for line in lines:
            line_split = line.split(' ')
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
            # print(img_path, gt_label)
            
            img_path = img_path.replace("/", "\\")
            # img_path = os.path.join(folder, partition, img_path)
            dataset[partition].append((img_path, gt_label))
        print(f"number of words in {partition}: {len(dataset[partition])}")
    
    lens = dict(sorted(lens.items()))

    
    for key in lens.keys():
        print(key, lens[key])


    return dataset['train'], dataset['valid'], dataset['test']
            

