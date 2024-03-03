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

# def normalize(img):
#     imgs = np.asarray(imgs).astype(np.float32)
#     _, h, w = img.shape

#     m, s = cv2.meanStdDev(img)
#     img = img - m[0][0]
#     img = img / s[0][0] if s[0][0] > 0 else img
#     return img

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

def preproc_iam(folder_from, folder_to) -> None:
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

    # try:
    #     preproc_rimes(path_from, path_to)
    #     getattr(f"preproc_{dataset_name}")(path_from, path_to)
    # except:
    #     print("dataset processing not implemented")



if __name__ == "__main__":
    main()