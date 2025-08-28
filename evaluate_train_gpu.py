from math import sqrt

import numpy as np
import torch

from tqdm import tqdm


import evaluation_index_gpu as evaluation_index


def compute_eva_gpu(obj, pred, t=0.1):

    rmse = torch.sqrt(torch.mean((obj - pred) ** 2))
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
    miou = 9999

    return [rmse, cc, pod, far, csi, acc, precision, f1, miou]

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    total_mask = []
    total_pred = []
    # iterate over the validation set

    with torch.no_grad():
        for ids_index, batch in tqdm(enumerate(dataloader), total=num_val_batches, desc='eva round', unit='batch'):
        # for ids_index, batch in enumerate(dataloader):

            image, geo, mask_true = batch['image'], batch['geo'], batch['mask']
            image = image.to(device=device, dtype=torch.float32)
            geo = geo.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=device, dtype=torch.float32)

            # predict the mask

            mask_pred = net(image, geo)

            if net.n_classes == 1:
                no_nan_index = ~torch.isnan(mask_true)
                total_mask.append(mask_true[no_nan_index])
                total_pred.append(torch.squeeze(mask_pred, 1)[no_nan_index])

            else:
                mask_pred = mask_pred.argmax(dim=1)
                no_nan_index = ~torch.isnan(mask_true)
                total_mask.append(mask_true[no_nan_index])
                total_pred.append(mask_pred[no_nan_index])

    total_mask = torch.cat(total_mask, dim=0)
    total_pred = torch.cat(total_pred, dim=0)

    total_prmd_acc_list = compute_eva_gpu(total_mask, total_pred, 0.1)
    total_s = " total:\n  rmse    cc   pod  far  csi   acc  precision  f1   miou\n" \
        + 'pred {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}\n'.format(total_prmd_acc_list[0],
                                                                                        total_prmd_acc_list[1],
                                                                                        total_prmd_acc_list[2],
                                                                                        total_prmd_acc_list[3],
                                                                                        total_prmd_acc_list[4],
                                                                                        total_prmd_acc_list[5],
                                                                                        total_prmd_acc_list[6],
                                                                                        total_prmd_acc_list[7],
                                                                                        total_prmd_acc_list[8])

    net.train()

    return dice_score / max(num_val_batches, 1), total_s, total_prmd_acc_list
