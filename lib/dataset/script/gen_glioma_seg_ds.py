import os
import random

import cv2
import shutil
from functools import partial

from pathlib import Path

import numpy as np
import albumentations as alb


def mix_img_lbl(img, lbl):
    lbl = cv2.applyColorMap(120 * lbl, cv2.COLORMAP_JET)
    ret = cv2.addWeighted(img, 0.6, lbl, 0.4, 1)
    return ret


def center_crop(img, w, h):
    img_h, img_w = img.shape[:2]
    x_min = img_w // 2 - w // 2
    y_min = img_h // 2 - h // 2
    return img[y_min:y_min+h, x_min:x_min+w]


def vis_seg(ds_path):
    for cls_type in ["tb", "ys"]:
        cls_path = os.path.join(ds_path, cls_type)
        for seq in os.listdir(cls_path):
            seq_path = os.path.join(cls_path, seq)
            print(seq_path)
            img_path = os.path.join(seq_path, "img")
            lbl_path = os.path.join(seq_path, "lbl")
            file_names = os.listdir(img_path)
            file_names = sorted(file_names, key=lambda x: int(x.split('.')[0]))
            for file_name in file_names:
                img_fp = os.path.join(img_path, file_name)
                lbl_fp = os.path.join(lbl_path, file_name)
                img = cv2.imread(img_fp)
                lbl = cv2.imread(lbl_fp, cv2.IMREAD_GRAYSCALE)
                to_vis = mix_img_lbl(img, lbl)
                # cv2.putText(to_vis, f"{cls_type} {seq} {file_name.split('.')[0]}",
                #             (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow("img", img)
                cv2.imshow("vis", to_vis)
                cv2.waitKey()

            to_vis = np.zeros((512, 512, 3), dtype=np.uint8)
            cv2.imshow("vis", to_vis)
            cv2.waitKey()


def augment_glioma_seg_ds(ds_path, save_path, img_func, lbl_func):
    for cls_type in ["tb", "ys"]:
        cls_path = os.path.join(ds_path, cls_type)
        for seq in os.listdir(cls_path):
            seq_path = os.path.join(cls_path, seq)
            print(seq_path)
            img_path = os.path.join(seq_path, "img")
            lbl_path = os.path.join(seq_path, "lbl")
            file_names = os.listdir(img_path)
            file_names = sorted(file_names, key=lambda x: int(x.split('.')[0]))

            seq_save_dir = os.path.join(save_path, cls_type, seq)
            img_save_dir = os.path.join(seq_save_dir, "img")
            lbl_save_dir = os.path.join(seq_save_dir, "lbl")
            if not os.path.exists(img_save_dir):
                os.makedirs(img_save_dir)
            if not os.path.exists(lbl_save_dir):
                os.makedirs(lbl_save_dir)

            for file_name in file_names:
                img_fp = os.path.join(img_path, file_name)
                lbl_fp = os.path.join(lbl_path, file_name)
                img = cv2.imread(img_fp)
                lbl = cv2.imread(lbl_fp, cv2.IMREAD_GRAYSCALE)

                img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
                lbl = cv2.resize(lbl, (256, 256), interpolation=cv2.INTER_NEAREST)

                img_parts = img_func(img)
                lbl_parts = lbl_func(lbl)

                assert len(img_parts) == len(lbl_parts), "invalid augmentation function!"

                for i, (img_p, lbl_p) in enumerate(zip(img_parts, lbl_parts)):
                    fn = f"{file_name.split('.')[0]}_{i+1}.png"
                    cv2.imwrite(os.path.join(img_save_dir, fn), img_p)
                    cv2.imwrite(os.path.join(lbl_save_dir, fn), lbl_p)

                vis = False
                if vis:
                    for img_part, lbl_part in zip(img_parts, lbl_parts):
                        cv2.imshow("p", mix_img_lbl(img_part, lbl_part))
                        cv2.waitKey()


def gen_txt_for_ds(ds_path, train_seq, val_seq):
    train_lines = []
    val_lines = []

    for seq in train_seq:
        seq_path = os.path.join(ds_path, seq)
        img_path = os.path.join(seq_path, "img")
        file_names = os.listdir(img_path)
        file_names = sorted(file_names)
        for file_name in file_names:
            img_path = os.path.join(seq, "img", file_name)
            lbl_path = os.path.join(seq, "lbl", file_name)
            train_lines.append(f"{img_path} {lbl_path} {1 if seq[:2] == 'tb' else 2}\n")

    for seq in val_seq:
        seq_path = os.path.join(ds_path, seq)
        img_path = os.path.join(seq_path, "img")
        file_names = os.listdir(img_path)
        file_names = sorted(file_names)
        for file_name in file_names:
            img_path = os.path.join(seq, "img", file_name)
            lbl_path = os.path.join(seq, "lbl", file_name)
            val_lines.append(f"{img_path} {lbl_path} {1 if seq[:2] == 'tb' else 2}\n")

    with open(os.path.join(ds_path, "seg_train.txt"), "w") as f:
        f.writelines(train_lines)

    with open(os.path.join(ds_path, "seg_val.txt"), "w") as f:
        f.writelines(val_lines)


def split_ds(ds_path):
    seq_list = []
    for cls_type in ["tb", "ys"]:
        cls_path = os.path.join(ds_path, cls_type)
        for seq in os.listdir(cls_path):
            seq_path = os.path.join(cls_type, seq)
            seq_list.append(seq_path)

    random.shuffle(seq_list)
    train = seq_list[:int(0.8 * len(seq_list))]
    val = seq_list[int(0.8 * len(seq_list)):]

    with open(os.path.join(ds_path, "train_seq.txt"), "w") as f:
        f.writelines([v+"\n" for v in train])

    with open(os.path.join(ds_path, "val_seq.txt"), "w") as f:
        f.writelines([v+"\n" for v in val])

    return train, val


def split_ds_k_fold(ds_path, k=10):
    seq_list = []
    for cls_type in ["tb", "ys"]:
        cls_path = os.path.join(ds_path, cls_type)
        for seq in os.listdir(cls_path):
            seq_path = os.path.join(cls_type, seq)
            seq_list.append(seq_path)

    random.shuffle(seq_list)
    val_num = len(seq_list) // k
    train_seqs, val_seqs = [], []
    for i in range(k):
        val = seq_list[val_num * i: val_num * (i+1)].copy()
        train = seq_list[:val_num * i].copy() + seq_list[val_num * (i+1):].copy()

        train_seqs.append(train)
        val_seqs.append(val)

        with open(os.path.join(ds_path, "train_seq_fold{}.txt".format(i + 1)), "w") as f:
            f.writelines([v+"\n" for v in train])

        with open(os.path.join(ds_path, "val_seq_fold{}.txt".format(i + 1)), "w") as f:
            f.writelines([v+"\n" for v in val])

    return train_seqs, val_seqs


def gen_txt_for_ds_k_fold(ds_path, train_seqs, val_seqs):
    for i, (train_seq, val_seq) in enumerate(zip(train_seqs, val_seqs)):
        train_lines = []
        val_lines = []

        for seq in train_seq:
            seq_path = os.path.join(ds_path, seq)
            img_path = os.path.join(seq_path, "img")
            file_names = os.listdir(img_path)
            file_names = sorted(file_names)
            for file_name in file_names:
                img_path = os.path.join(seq, "img", file_name)
                lbl_path = os.path.join(seq, "lbl", file_name)
                train_lines.append(f"{img_path} {lbl_path} {1 if seq[:2] == 'tb' else 2}\n")

        for seq in val_seq:
            seq_path = os.path.join(ds_path, seq)
            img_path = os.path.join(seq_path, "img")
            file_names = os.listdir(img_path)
            file_names = sorted(file_names)
            for file_name in file_names:
                img_path = os.path.join(seq, "img", file_name)
                lbl_path = os.path.join(seq, "lbl", file_name)
                val_lines.append(f"{img_path} {lbl_path} {1 if seq[:2] == 'tb' else 2}\n")

        with open(os.path.join(ds_path, "seg_train_fold{}.txt".format(i+1)), "w") as f:
            f.writelines(train_lines)

        with open(os.path.join(ds_path, "seg_val_fold{}.txt".format(i+1)), "w") as f:
            f.writelines(val_lines)


def get_mean_var(ds_path):
    mean_bs = []
    std_bs = []

    for cls_type in ["tb", "ys"]:
        cls_path = os.path.join(ds_path, cls_type)
        for seq in os.listdir(cls_path):
            seq_path = os.path.join(cls_path, seq)
            img_path = os.path.join(seq_path, "img")
            file_names = os.listdir(img_path)
            file_names = sorted(file_names)
            for fn in file_names:
                img_fp = os.path.join(img_path, fn)
                img = cv2.imread(img_fp)
                b, g, r = cv2.split(img)
                mean_b, var_b = np.mean(b), np.std(b)
                mean_bs.append(mean_b)
                std_bs.append(var_b)

    print(np.mean(mean_bs) / 255)
    print(np.mean(std_bs) / 255)


def get_cls_weight(ds_path):
    cls_cnt = [0, 0, 0]

    for cls_type in ["tb", "ys"]:
        cls_path = os.path.join(ds_path, cls_type)
        for seq in os.listdir(cls_path):
            seq_path = os.path.join(cls_path, seq)
            img_path = os.path.join(seq_path, "lbl")
            file_names = os.listdir(img_path)
            file_names = sorted(file_names)
            for fn in file_names:
                lbl_fp = os.path.join(img_path, fn)
                lbl = cv2.imread(lbl_fp, cv2.IMREAD_GRAYSCALE)
                for cls_id in range(3):
                    cls_cnt[cls_id] += np.sum(lbl == cls_id)

    print(cls_cnt)


def flip(img):
    flipped = img[:, ::-1].copy()
    parts = [img, flipped]
    return parts


if __name__ == '__main__':
    # vis_seg("/home/asus/Documents/outpart-t1_seg")

    # prefix = "glioma"
    prefix = "glioma-t1"

    # augment_glioma_seg_ds(f"/home/asus/Documents/{prefix}_seg",
    #                       f"/home/asus/Documents/{prefix}_seg_256",
    #                       flip, flip)
    #
    # t_seq, v_seq = split_ds_k_fold(f"/home/asus/Documents/{prefix}_seg", k=5)
    # gen_txt_for_ds_k_fold(f"/home/asus/Documents/{prefix}_seg_256", t_seq, v_seq)

    get_mean_var(f"/home/asus/Documents/{prefix}_seg_256")
    # get_cls_weight(f"/home/asus/Documents/{prefix}_seg")
