import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate_train_gpu import evaluate
from process_recode import find_best
from unet_middle_fusion_v4_cat_geo_v4_32_change2 import UNet
from utils.data_loading_npy import BasicDataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 设定随机种子
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_cur_time():
    return datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 1,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        classes: float=1.0
):
    start_time = get_cur_time()
    savedir = './checkpoints_reg/' + start_time + '/'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    recode_txt = savedir + 'val_and_test_acc.txt'
    with open(recode_txt, mode='w', encoding='utf-8') as f:
        f.write(start_time + '\n')

    # Create dataset
    train_dataset = BasicDataset(train_dir_img, img_scale, classes, geo_norm, geo_channel)
    val_dataset = BasicDataset(val_dir_img, img_scale, classes, geo_norm, geo_channel)
    test_dataset = BasicDataset(test_dir_img, img_scale, classes, geo_norm, geo_channel)

    n_val = len(val_dataset)
    n_train = len(train_dataset)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, batch_size=1,  num_workers=8,pin_memory=True)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=True, batch_size=1,  num_workers=8,pin_memory=True)

    # experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    # experiment.config.update(
    #     dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    #          val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    # )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')


    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.MSELoss()

    global_step = 0

    # Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, geo, true_masks = batch['image'], batch['geo'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                geo = geo.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images, geo)

                    mask_index = torch.isnan(true_masks)
                    masks_pred_no_nan = masks_pred.squeeze(1)[~mask_index]
                    masks_true_no_nan = true_masks.float()[~mask_index]

                    loss = criterion(masks_pred_no_nan, masks_true_no_nan)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                # experiment.log({
                #     'train loss': loss.item(),
                #     'step': global_step,
                #     'epoch': epoch
                # })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

            # Evaluation round
            division_step = 20
            if epoch % division_step == 0 or epoch == epochs or epoch == 1:
                val_aver_rmse, val_str, val_list = evaluate(model, val_loader, device, amp)
                with open(recode_txt, 'a') as f:
                    f.write(str(epoch) + '_val_res:' + val_str)

                logging.info('Validation str score: {}'.format(val_str))

                # experiment.log({
                #     'learning rate': optimizer.param_groups[0]['lr'],
                #     'aver_rmse': val_aver_rmse,
                #     'total_rmse': val_list[0],
                #     'cc': val_list[1],
                #     'pod': val_list[2],
                #     'far': val_list[3],
                #     'csi': val_list[4],
                #     'f1': val_list[7],
                #     # 'images': wandb.Image(images[0].cpu()),
                #     'masks': {
                #         'true': wandb.Image(true_masks[0].float().cpu()),
                #         'pred': wandb.Image((masks_pred.argmax(dim=1)[0].float()).cpu()),
                #     },
                #     'step': global_step,
                #     'epoch': epoch,
                # })

        scheduler.step()

        if save_checkpoint and epoch % 20 == 0 or epoch == 1:

            test_aver_rmse, test_str, test_list = evaluate(model, test_loader, device, amp)

            with open(recode_txt, 'a') as f:
                f.write(str(epoch) + '_test_res:' + test_str + '\n')
            logging.info('test_res: {}'.format(test_str))
            _, val_best, test_best = find_best(recode_txt, True)
            # if val_best == epoch or test_best == epoch:
            state_dict = model.state_dict()
            torch.save(state_dict, savedir + 'checkpoint_epoch{}.pth'.format(epoch))
            logging.info(f'Checkpoint {epoch} saved!')

        if epoch == epochs:
            state_dict = model.state_dict()
            torch.save(state_dict, savedir + 'checkpoint_epoch{}.pth'.format(epoch))
            logging.info(f'Checkpoint {epoch} saved!')

    # test_aver_rmse, test_str, test_list = evaluate(model, test_loader, device, amp)
    # with open(recode_txt, 'a') as f:
    #     f.write(str(epochs) + '_test_res: ' + test_str + '\n')
    # logging.info('test_res: {}'.format(test_str))
    # print('test', test_str)
    find_best(recode_txt)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=120, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=64, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    # parser.add_argument('--bilinear',  default=True, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')

    return parser.parse_args()

    # train_dir_img = Path('/root/prmd/PRMD_DATA_21-22_npy_v2/train')
    # val_dir_img = Path('/root/prmd/PRMD_DATA_21-22_npy_v2/val')
    # test_dir_img = Path('/root/prmd/PRMD_DATA_21-22_npy_v2/test')



if __name__ == '__main__':

    # =======================输入参数===========================#
    # 输入封装好的npy格式的PRMD数据集路径
    # train_dir_img = Path('PRMD_DATA_21-22_npy_v2/train')
    # val_dir_img = Path('PRMD_DATA_21-22_npy_v2/val')
    # test_dir_img = Path('PRMD_DATA_21-22_npy_v2/test')
    # train_dir_img = Path('E:/2021_2022_dataset/PRMD_DATA_21-22_npy/train1')
    # val_dir_img = Path('E:/2021_2022_dataset/PRMD_DATA_21-22_npy/val1')
    # test_dir_img = Path('E:/2021_2022_dataset/PRMD_DATA_21-22_npy/test1')

    # train_dir_img = Path('/root/prmd/PRMD_DATA_21-22_npy_v3/train')
    # val_dir_img = Path('/root/prmd/PRMD_DATA_21-22_npy_v3/val')
    # test_dir_img = Path('/root/prmd/PRMD_DATA_21-22_npy_v3/test')
    #

    train_dir_img = Path('/root/prmd/PRMD_DATA_21-22_npy_v4/train')
    val_dir_img = Path('/root/prmd/PRMD_DATA_21-22_npy_v4/val')
    test_dir_img = Path('/root/prmd/PRMD_DATA_21-22_npy_v4/test')

    # 输入高程归一化最大最小值
    with open('geo_norm.txt', 'r') as f:
        geo_str = f.read()

    with open('geo_channel.txt', 'r') as f:
        channel_str = f.read()

    geo_norm = list(map(int, geo_str.split('_')))  # [6000, -300]
    geo_channel = list(map(int, channel_str.split('_')))  # [6000, -300]

    print(geo_norm, geo_channel)

    # =======================开始训练============================#
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = UNet(n_channels=197, n_classes=args.classes, geo_channels=geo_channel[1]-geo_channel[0], bilinear=args.bilinear)  # geo通道为1，输入海拔

    model = model.to(device)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)

    train_model(
        model=model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        img_scale=args.scale,
        val_percent=args.val / 100,
        amp=args.amp,
        classes=args.classes
    )
