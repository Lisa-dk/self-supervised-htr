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


def preproc_iam(folder_from, folder_to) -> None:
    """Read IAM words from given folder and save preprocessed images to new folder"""
    partitions = ['train', 'test']

    pt_path = os.path.join("..", "..", "Groundtruth")
    
    paths = {"train": open(os.path.join(pt_path, "gan.iam.gan_tr_va.filter27")).read().splitlines(),
                "test": open(os.path.join(pt_path, "gan.iam.gan_test.filter27")).read().splitlines()}
    
    new_file_name = os.path.join(folder_to, f"ground_truth_filtered.txt")

    for i in partitions:
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
                
                f1, f2, _, _ = img_path.split('-')

                img_path_from = os.path.join(folder_from, "words", f1, f"{f1}-{f2}", img_path + ".png")
                print(img_path_from)

                img = cv2.imread(img_path_from, cv2.IMREAD_GRAYSCALE)

                if img is not None:
                    img = img_padding(img, INPUT_SIZE[1])
                    img = resize(img, INPUT_SIZE)

                    os.makedirs(os.path.join(folder_to, f1, f"{f1}-{f2}"), exist_ok=True)
                    new_img_path = os.path.join(folder_to, f1, f"{f1}-{f2}", img_path)
                    cv2.imwrite(new_img_path + '.png', img)

                    with open(new_file_name, mode="a", encoding="utf-8") as new_data_file:
                        new_data_file.write(f"{new_img_path + '.png'} {gt_label} {wid}\n")

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




if __name__ == "__main__":
    main()