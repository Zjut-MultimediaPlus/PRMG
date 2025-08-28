import os
from math import sqrt
from pathlib import Path
import xarray as xr
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


from utils.data_loading_npy import BasicDataset


import evaluation_index_gpu as evaluation_index
 

def compute_eva(obj, pred, t=0.1):
    obj = np.array(obj)
    pred = np.array(pred)
    mse_loss = torch.nn.MSELoss()
    rmse = sqrt(mse_loss(torch.tensor(pred), torch.tensor(obj)))

    if np.all(obj == 0) or np.all(pred == 0):
        cc = 0  # 设置相关系数为0或其他特定值
    else:
        cc = np.corrcoef(obj, pred)[0, 1]

    acc = evaluation_index.ACC(obj, pred, t)
    precision = evaluation_index.precision(obj, pred, t)
    f1 = evaluation_index.FSC(obj, pred, t)
    # prmd_recall = evaluation_index.recall(obj, pred)
    pod = evaluation_index.POD(obj, pred, t)
    far = evaluation_index.FAR(obj, pred, t)
    csi = evaluation_index.CSI(obj, pred, t)
    miou = compute_miou(pred, obj, t)

    return [rmse, cc, pod, far, csi, acc, precision, f1, miou]


def get_eva_list(total_mask, total_pred, info, t=0.1, title=True):

    prmd_acc_list = compute_eva_gpu(total_mask, total_pred, t)
    if title:
        acc_str = " total:\n  rmse    cc   pod  far  csi   acc  precision  f1   miou\n" \
            + info +' {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}\n'.format(prmd_acc_list[0],
              prmd_acc_list[1], prmd_acc_list[2], prmd_acc_list[3], prmd_acc_list[4],
              prmd_acc_list[5], prmd_acc_list[6], prmd_acc_list[7], prmd_acc_list[8])
    else:
        acc_str = info +' {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}\n'.format(prmd_acc_list[0],
              prmd_acc_list[1], prmd_acc_list[2], prmd_acc_list[3], prmd_acc_list[4],
              prmd_acc_list[5], prmd_acc_list[6], prmd_acc_list[7], prmd_acc_list[8])

    return prmd_acc_list, acc_str

def compute_eva_gpu(obj, pred, t=0.1):
    # obj = np.array(obj)
    # pred = np.array(pred)
    mse_loss = torch.nn.MSELoss()
    rmse = sqrt(mse_loss(pred, obj))

    if torch.all(obj == 0) or torch.all(pred == 0):
        cc = 0  # 设置相关系数为0或其他特定值
    else:
        # cc = np.corrcoef(obj, pred)[0, 1]
        stacked_data = torch.stack([obj, pred], dim=0)
        cc = torch.corrcoef(stacked_data)[0, 1].item()

    acc = evaluation_index.ACC(obj, pred, t)
    precision = evaluation_index.precision(obj, pred, t)
    f1 = evaluation_index.FSC(obj, pred, t)
    # prmd_recall = evaluation_index.recall(obj, pred)
    pod = evaluation_index.POD(obj, pred, t)
    far = evaluation_index.FAR(obj, pred, t)
    csi = evaluation_index.CSI(obj, pred, t)
    miou = 9999  # compute_miou(pred, obj, t)

    return [rmse, cc, pod, far, csi, acc, precision, f1, miou]

@torch.inference_mode()
def evaluate(net_cls, net_reg, dataloader, device, thre = 0.1):

    net_cls.eval()
    net_reg.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    total_mask = []
    total_pred_v1 = []
    total_pred_v2 = []
    total_gpm = []
    # iterate over the validation set

    with torch.no_grad():
        for ids_index, batch in tqdm(enumerate(dataloader), total=num_val_batches, desc='eva round', unit='batch', leave=False):

            image, geo, mask_true = batch['image'], batch['geo'], batch['mask']
            geo = geo.to(device=device, dtype=torch.float32)
            image = image.to(device=device, dtype=torch.float32)

            mask_true = mask_true.to(device=device, dtype=torch.float32)
            no_nan_index = ~torch.isnan(mask_true)
            # predict the mask

            # 进行模型反演
            mask_pred_cls = net_cls(image, geo)
            mask_pred_reg = net_reg(image, geo)

            mask_pred_cls = mask_pred_cls.argmax(dim=1)
            mask_pred_reg = torch.squeeze(mask_pred_reg, 1)

            mask_pred_reg_v1 = mask_pred_reg.clone()
            mask_pred_reg_v2 = mask_pred_reg.clone()

            # 两种合并方式
            mask_pred_reg_v1[torch.logical_and(mask_pred_cls == 1, mask_pred_reg <= 0.1)] = 0.1
            mask_pred_reg_v1[mask_pred_cls == 0] = 0

            mask_pred_reg_v2[mask_pred_cls == 0] = 0

            total_mask.append(mask_true[no_nan_index])
            total_pred_v1.append(mask_pred_reg_v1[no_nan_index])
            total_pred_v2.append(mask_pred_reg_v2[no_nan_index])


    total_mask = torch.cat(total_mask, dim=0)
    total_pred_v1 = torch.cat(total_pred_v1, dim=0)
    total_pred_v2 = torch.cat(total_pred_v2, dim=0)

    total_prmd_acc_list_v1, prmd_s_v1 = get_eva_list(total_mask, total_pred_v1, 'prmd_v1', thre)
    total_prmd_acc_list_v2, prmd_s_v2 = get_eva_list(total_mask, total_pred_v2, 'prmd_v2', thre, False)
    # if total_gpm:
    #     total_gpm_acc_list, gpm_s = get_eva_list(total_mask, total_gpm, 'gpm', thre, False)
    #     total_s = 'all ' + prmd_s + gpm_s
    # else:
    total_s = 'all ' + prmd_s_v1 + prmd_s_v2


    return dice_score / max(num_val_batches, 1), total_s, total_prmd_acc_list_v1, total_prmd_acc_list_v2


if __name__ == '__main__':


    # 输入模型文件
    from unet_middle_fusion_v4_cat_geo_v4_32_change2 import UNet
    model_cls = r'F:\pycharm_project\PRMD-train-rain-geo-vF\checkpoints_cls\2024-07-30_01-17\checkpoint_epoch80.pth'
    model_reg = r'F:\pycharm_project\PRMD-train-rain-geo-vF\checkpoints_reg\2024-07-30_02-12\checkpoint_epoch80.pth'

    test_dir_img = Path(r'G:\2021_2022_dataset\PRMD_DATA_21-22_npy_v4\test/')
    with open('geo_norm.txt', 'r') as f:
        geo_str = f.read()
    geo_norm = list(map(int, geo_str.split('_')))

    test_dataset = BasicDataset(test_dir_img, 1, 1, geo_norm, [0, 3])
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=True, batch_size=1, num_workers=4, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net_cls = UNet(n_channels=197, n_classes=2, geo_channels=3)
    net_cls.to(device=device)
    state_dict = torch.load(model_cls, map_location=device)
    net_cls.load_state_dict(state_dict)

    net_reg = UNet(n_channels=197, n_classes=1, geo_channels=3)
    net_reg.to(device=device)
    state_dict = torch.load(model_reg, map_location=device)
    net_reg.load_state_dict(state_dict)

    test_score, test_str, test_list_v1, test_list_v2 = evaluate(net_cls, net_reg, test_loader, device)

    print(test_str)

