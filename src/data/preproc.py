import cv2, sys, os, string
import numpy as np

INPUT_SIZE = (216, 64, 1)

def resize(img, input_size):

    u, i = np.unique(np.array(img).flatten(), return_inverse=True)
    background = int(u[np.argmax(np.bincount(i))])

    wt, ht, _ = input_size
    h, w = np.asarray(img).shape

    f = max((w / wt), (h / ht))

    new_size = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1))

    img = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)

    target = np.ones([ht, wt], dtype=np.uint8) * background
    target[0:new_size[1], 0:new_size[0]] = img

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
    partitions = ['train', 'valid', 'test']
    for partition in partitions:
        new_file_name = text_file = os.path.join(folder_to, f"ground_truth_{partition}_filtered.txt")

        text_file = os.path.join(folder_from, f"ground_truth_{partition}_icdar2011.txt")

        with open(text_file, encoding='utf-8') as data_file:
            lines = data_file.read().splitlines()
        
        for line in lines:
            img_path, gt_label = line.split(' ')            

            if len(gt_label) <= 1:
                continue

            if not set(string.punctuation).isdisjoint(set(gt_label)):
                continue

            if not set(string.digits).isdisjoint(set(gt_label)):
                continue

            img_path = img_path.replace("/", "\\")

            img_path_from = os.path.join(folder_from, partition, img_path)

            img = cv2.imread(img_path_from, cv2.IMREAD_GRAYSCALE)
            img = resize(img, INPUT_SIZE)
            # img = normalize(img)

            img_folder = '\\'.join(img_path.split('\\')[:-1])
            os.makedirs(os.path.join(folder_to, partition, img_folder), exist_ok=True)
            new_img_path = os.path.join(folder_to, partition, img_path)

            cv2.imwrite(new_img_path, img)

            with open(new_file_name, mode="a", encoding="utf-8") as new_data_file:
                new_data_file.write(f"{new_img_path} {gt_label}\n")
            

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