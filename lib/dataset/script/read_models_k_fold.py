import os
import torch

if __name__ == '__main__':
    model_path = "/home/asus/Desktop/glioma-python/Checkpoints/checkpoints"
    model_dirs = os.listdir(model_path)

    acc_list = []
    for model_dir in model_dirs:
        mp = os.path.join(model_path, model_dir, "best.pt")
        pt_info = torch.load(mp)
        acc_list.append(pt_info['k_fold_acc_best'])

    sum = 0
    for acc in acc_list:
        sum += acc

    print(sum / len(acc_list))
    print(acc_list)
