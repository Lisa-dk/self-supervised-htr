import cv2, sys, os, string
import numpy as np
import matplotlib.pyplot as plt
INPUT_SIZE = (216, 64, 1)


def img_padding(img, input_height):
       
        if len(img)/input_height <= 0.5:
            desired_height = input_height
            delta_h = int(((desired_height - len(img))/2) * (1 - (len(img)/input_height))) # * 1 - ratio other small images are too zoomed int
        else:
            delta_h = 0
        new_im = np.pad(
                    img,
                    pad_width=((delta_h, delta_h), (0, 0)),
                    mode="constant",
                    constant_values=(255),
        )
        return new_im


def resize(img, input_size):
    """Resize and center img to given input_size by ratio to height or width"""

    u, i = np.unique(np.array(img).flatten(), return_inverse=True)
    background = int(u[np.argmax(np.bincount(i))]) # most frequent occuring pixel value

    wt, ht, _ = input_size
    h, w = np.asarray(img).shape

    f = max((w / wt), (h / ht)) # ratio for resizing

    new_size = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1))

    img = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)

    target = np.ones([ht, wt], dtype=np.uint8) * background

    # vertical centering
    start = int((ht / 2) - (new_size[1] / 2))
    end = start + new_size[1]
    target[start:end, :img.shape[-1]] = img 

    return target

def normalize(img):
    img = 1. - (img / 255.) # 0-255 -> 0-1

    m, s = 0.5, 0.5
    return (img - m) / s

def preproc_rimes(folder_from, folder_to) -> None:
    """Read RIMES data in given folder and save pre-processed images to new folder"""
    folder_from = os.path.join(folder_from, "words")
    partitions = ['train', 'valid', 'test']
    for partition in partitions:
        new_file_name = os.path.join(folder_to, f"ground_truth_{partition}_filtered.txt")

        text_file = os.path.join(folder_from, f"ground_truth_{partition}_icdar2011.txt")

        with open(text_file, encoding='utf-8') as data_file:
            lines = data_file.read().splitlines()
        
        for line in lines:
            img_path, gt_label = line.split(' ')            

            # skip 1-char words
            if len(gt_label) <= 1:
                continue
            
            # skip words with punctuation
            if not set(string.punctuation).isdisjoint(set(gt_label)):
                continue

            # skip words with digits
            if not set(string.digits).isdisjoint(set(gt_label)):
                continue

            img_path = img_path.replace("/", "\\")

            img_path_from = os.path.join(folder_from, partition, img_path)

            img = cv2.imread(img_path_from, cv2.IMREAD_GRAYSCALE)
            img = img_padding(img, INPUT_SIZE[1])
            img = resize(img, INPUT_SIZE)

            img_folder = '\\'.join(img_path.split('\\')[:-1])
            os.makedirs(os.path.join(folder_to, partition, img_folder), exist_ok=True)
            new_img_path = os.path.join(folder_to, partition, img_path)

            cv2.imwrite(new_img_path, img)

            with open(new_file_name, mode="a", encoding="utf-8") as new_data_file:
                new_data_file.write(f"{new_img_path} {gt_label}\n")

def preproc_iam_prev(folder_from, folder_to) -> None:
    """Read IAM words from given folder and save preprocessed images to new folder"""
    partitions = ['train', 'valid', 'test']
    gt_dict = {}

    pt_path = os.path.join(folder_from, "largeWriterIndependentWordRecognitionTask")
    
    paths = {"train": open(os.path.join(pt_path, "trainset.txt")).read().splitlines(),
                "valid": open(os.path.join(pt_path, "validationset1.txt")).read().splitlines() +
                open(os.path.join(pt_path, "validationset2.txt")).read().splitlines(),
                "test": open(os.path.join(pt_path, "testset.txt")).read().splitlines()}
    
    lines = open(os.path.join(folder_from, "ascii", "words.txt")).read().splitlines()

    for line in lines:
        if (not line or line[0] == "#"):
            continue

        split = line.split()
        gt_dict[split[0]] = " ".join(split[8::]).replace("|", " ")

    for i in partitions:
        new_file_name = os.path.join(folder_to, f"ground_truth_{i}_filtered.txt")
        for line in paths[i]:
            try:
                gt_label = gt_dict[line]

                # skip 1-char words
                if len(gt_label) <= 1:
                    continue

                # skip words with punctuation
                if not set(string.punctuation).isdisjoint(set(gt_label)):
                    continue

                # skip words with digits
                if not set(string.digits).isdisjoint(set(gt_label)):
                    continue

                split = line.split("-")
                folder = f"{split[0]}-{split[1]}"

                img_file = f"{split[0]}-{split[1]}-{split[2]}-{split[3]}.png"
                img_path_from = os.path.join(folder_from, "words", split[0], folder, img_file)

                img = cv2.imread(img_path_from, cv2.IMREAD_GRAYSCALE)
                img = img_padding(img, INPUT_SIZE[1])
                img = resize(img, INPUT_SIZE)

                os.makedirs(os.path.join(folder_to, i, split[0], folder), exist_ok=True)
                new_img_path = os.path.join(folder_to, i, split[0], folder, img_file)
                cv2.imwrite(new_img_path, img)

                with open(new_file_name, mode="a", encoding="utf-8") as new_data_file:
                    new_data_file.write(f"{new_img_path} {gt_label}\n")

            except KeyError:
                pass

def preproc_iam(folder_from, folder_to) -> None:
    """Read IAM words from given folder and save preprocessed images to new folder"""
    pt_path = os.path.join(folder_from, "Groundtruth")

    if "GANwriting" in folder_from:
        partitions = ['train', 'valid']
        paths = {"train": open(os.path.join(pt_path, "gan.iam.gan_tr_va.filter27")).read().splitlines(),
                    "valid": open(os.path.join(pt_path, "gan.iam.gan_test.filter27")).read().splitlines()}
    else:
        partitions = ['train', 'valid', 'test', 'oov_train', 'oov_valid', 'oov_test']
        paths = {"train": open(os.path.join(pt_path, "htr.iam.train.filter27")).read().splitlines(),
                    "valid": open(os.path.join(pt_path, "htr.iam.val.filter27")).read().splitlines(),
                    "test": open(os.path.join(pt_path, "htr.iam.test.filter27")).read().splitlines(),
                    "oov_train": open(os.path.join(pt_path, "htr.iam.oov_train.filter27")).read().splitlines(),
                    "oov_valid": open(os.path.join(pt_path, "htr.iam.oov_val.filter27")).read().splitlines(),
                    "oov_test": open(os.path.join(pt_path, "htr.iam.oov_test.filter27")).read().splitlines()}

    for i in partitions:
        print(i)
        new_file_name = os.path.join(folder_to, f"ground_truth_{i}_filtered.txt")
        new_data_file = open(new_file_name, 'w', encoding="utf-8")
        for line in paths[i]:
            try:
                wid_path, gt_label = line.split(' ')
                wid, img_path = wid_path.split(',')
                img_path = img_path.replace('/', '\\')

                # skip 1-char words
                if len(gt_label) <= 1:
                    continue

                # skip words with punctuation
                if not set(string.punctuation).isdisjoint(set(gt_label)):
                    continue

                # skip words with digits
                if not set(string.digits).isdisjoint(set(gt_label)):
                    continue
                
                if "GANwriting" in folder_from:
                    img_path_from = os.path.join(folder_from, "data", "iam", "words", img_path + ".png") 
                    f1, f2, _, _ =  img_path.split('-')
                    new_img_path = os.path.join(folder_to, i, img_path + ".png")
                else: 
                    img_path_from = os.path.join(folder_from, "words", img_path)
                    f1, f2, _ = img_path.split('\\')
                    new_img_path = os.path.join(folder_to, i, img_path)

                os.makedirs(os.path.join(folder_to, i, f1, f2), exist_ok=True)
                print(img_path_from)
                
                img = cv2.imread(img_path_from, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = img_padding(img, INPUT_SIZE[1])
                    img = resize(img, INPUT_SIZE)

                    cv2.imwrite(new_img_path, img)

                    new_data_file.write(f"{new_img_path} {gt_label} {wid}\n")

            except KeyError:
                pass
        new_data_file.close()
        

def main():
    dataset_name = sys.argv[1]
    format = sys.argv[2]
    folder_from = "raw"
    folder_to =  "data"
    path_from = os.path.join("..", "..", folder_from, dataset_name, format)
    path_to = os.path.join("..", "..", folder_to, dataset_name, format)
    print(path_to)
    os.makedirs(path_to, exist_ok=True)
    preproc_rimes(path_from, path_to)


if __name__ == "__main__":
    main()