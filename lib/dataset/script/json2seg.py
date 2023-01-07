import os
import glob
import shutil
import cv2
import random
from pathlib import Path


def center_crop(img, w, h):
    img_h, img_w = img.shape[:2]
    x_min = img_w // 2 - w // 2
    y_min = img_h // 2 - h // 2
    return img[y_min:y_min+h, x_min:x_min+w]


if __name__ == '__main__':
    prefix = "outpart-t1"
    # prefix = "glioma"

    ori_root = f"/home/asus/Documents/{prefix}_ori"
    ds_root = f"/home/asus/Documents/{prefix}_ds"
    seg_root = f"/home/asus/Documents/{prefix}_seg"

    # for json in glob.glob(ori_root+"/*/*/*.json"):
    #     os.system(f"labelme_json_to_dataset {json}")

    patients = []
    for cls_dir in ["tb", "ys"]:
        cls_path = os.path.join(ori_root, cls_dir)
        for patient_id in os.listdir(cls_path):
            seq_path = os.path.join(cls_path, patient_id)
            frames = [p for p in os.listdir(seq_path) if os.path.isdir(os.path.join(seq_path, p))]
            frames = sorted(frames, key=lambda x: int(x.split('_')[1]))

            dest_dir = os.path.join(ds_root, cls_dir[:2], patient_id)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
                os.makedirs(os.path.join(dest_dir, "img"))
                os.makedirs(os.path.join(dest_dir, "lbl"))

            for frame in frames:
                frame_path = os.path.join(seq_path, frame)
                img_path = os.path.join(frame_path, "img.png")
                label_path = os.path.join(frame_path, "label.png")

                frame_ind = int(frame.split('_')[1])
                assert frame_ind < 1000, 'frame_ind > 1000'
                dest_img = os.path.join(dest_dir, "img", "{:3>0}.png".format(frame_ind))
                dest_lbl = os.path.join(dest_dir, "lbl", "{:3>0}.png".format(frame_ind))

                shutil.copyfile(img_path, dest_img)
                shutil.copyfile(label_path, dest_lbl)

                cmp_dir = os.path.join(cls_dir[:2], patient_id)

            patients.append([patient_id, cls_dir])

    for i, patient in enumerate(patients):
        print("{} / {}".format(i + 1, len(patients)))

        patient_id, cls_str = patient

        img_dir = os.path.join(ds_root, cls_str, patient_id, "img")
        lbl_dir = os.path.join(ds_root, cls_str, patient_id, "lbl")

        for fn in os.listdir(img_dir):
            img_path = os.path.join(img_dir, fn)
            lbl_path = os.path.join(lbl_dir, fn)

            img = cv2.imread(img_path)
            lbl = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)

            h, w, c = img.shape
            l_sqr = min(h, w)

            img = center_crop(img, l_sqr, l_sqr)
            lbl = center_crop(lbl, l_sqr, l_sqr)

            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
            lbl = cv2.resize(lbl, (512, 512), interpolation=cv2.INTER_NEAREST)

            cls_id = 1 if cls_str == "tb" else 2
            lbl[lbl == 38] = cls_id

            img2save = img.copy()
            lbl2save = lbl.copy()

            img_path = Path(img_path)
            lbl_path = Path(lbl_path)

            target_dir = str(img_path.parent).replace("_ds", "_seg")
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            target_dir = str(lbl_path.parent).replace("_ds", "_seg")
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            img_save_path = str(img_path).replace("_ds", "_seg")
            lbl_save_path = str(lbl_path).replace("_ds", "_seg")

            cv2.imwrite(img_save_path, img2save)
            cv2.imwrite(lbl_save_path, lbl2save)

        lines = [f"{cls_str}/{patient_id}\n" for patient_id, cls_str in patients]
        with open(f"/{seg_root}/seq.txt", "w") as f:
            f.writelines(lines)
