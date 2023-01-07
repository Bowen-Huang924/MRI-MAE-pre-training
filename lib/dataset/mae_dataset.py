import os
import numpy as np
import cv2
import torch.nn as nn
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from albumentations import Compose, RandomBrightnessContrast, HorizontalFlip, ShiftScaleRotate, VerticalFlip, Crop

__all__ = ["Mae_Dataset"]


class Mae_Dataset(data.Dataset):
    def __init__(self, data_root, k, cfg, is_train=True):
        print('use mae dataset')
        super().__init__()
        self.data_root = data_root
        self.is_train = is_train
        self.txt_path = '/home/lthpc/xx/glioma-mae/mae_file.txt'
        self.ann_list = self._create_ann_lst()
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(tuple(cfg.mean), tuple(cfg.var))  # image-net bgr
        ])

        self.transforms = None
        if is_train:
            self.transforms = Compose([
                RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5,
                                 border_mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_AREA)
            ])

    def __getitem__(self, index):
        img_path = self.ann_list[index].split('\n')[0]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

        if self.is_train:
            ori_img = self._transform(img)
        else:
            ori_img = img

        img = torch.from_numpy(ori_img).permute(2, 0, 1)
        img = torch.unsqueeze(img, 0)
        x = self.patchify(img)
        x, mask, ids_restore = self.random_masking(x, mask_ratio=0.45)
        mask_token = nn.Parameter(torch.zeros(1, 1, 768))
        mask_tokens = mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        # print("x_", x_.shape)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        img = self.unpatchify(x_)
        img = torch.squeeze(img)
        # print(img.shape)
        img = img.permute(1, 2, 0)
        img = img.detach().numpy()/255
        # cv2.imshow('a', img)
        # cv2.waitKey(0)
        # cv2.imshow('b', ori_img)
        # cv2.waitKey(0)
        img = self.to_tensor(img).float()
        ori_img = self.to_tensor(ori_img).float()
        # label = torch.from_numpy(label).long()
        # cls_id = torch.from_numpy(np.array(cls_id)).long()



        return img,mask,ori_img

    def patchify(self,imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        # p = self.patch_embed.patch_size[0]
        p = 16
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self,x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        # p = self.patch_embed.patch_size[0]
        p = 16
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def __len__(self):
        return len(self.ann_list)

    def _transform(self, img):
        data = self.transforms(image=img)
        img = data["image"]
        # label = data["mask"]
        return img

    def _create_ann_lst(self):
        with open(self.txt_path, 'r') as f:
            lines = f.readlines()
            # print(lines)

        # anns = []
        # for line in lines:
        #     img_path, label_path, cls_id = line[:-1].split(' ')
        #     image_path = os.path.join(self.data_root, img_path)
        #     label_path = os.path.join(self.data_root, label_path)
        #     cls_id = int(cls_id)
        #     anns.append([image_path, label_path, cls_id])
        # return anns
        return lines


if __name__ == '__main__':
    res = 256
    data_path = f'/home/asus/Documents/glioma-_seg_{res}'
    dataset = GliomaSegKFold(data_path, is_train=False, k=1)
    for i in range(len(dataset)):
        print("\r{:d}/{:d}".format(i + 1, len(dataset)))
        img, lbl, cls_id = dataset[i]
