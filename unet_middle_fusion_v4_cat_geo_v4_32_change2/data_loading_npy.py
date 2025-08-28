import logging
import numpy as np
import torch
from PIL import Image
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
import xarray as xr


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.hdf':
        return xr.open_dataset(filename)
    elif ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def load_image_xr(filename):


    ds = xr.open_dataset(filename)
    img = ds['hy_data'].values
    mask = ds['PRECI_cldas'].values

    lat = ds['Latitude'].values
    lon = ds['Longitude'].values
    dem = ds['DEM_elevation'].values

    geo = np.array([lat, lon, dem]).transpose(1,2,0)[:,:,2:3]
    # geo = np.array([lat, lon, dem]).transpose(1,2,0)
    real_len = len(ds['TIME_dayOfYear'].values)

    return img, geo, mask, real_len


class BasicDataset(Dataset):
    def __init__(self, images_dir, scale, classes, geo_norm, geo_channel):
        self.images_dir = Path(images_dir)
        self.classes = classes
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.geo_norm = geo_norm
        self.geo_channel = geo_channel

        # self.ids = sorted([splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')])
        # self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if 'train' in str(images_dir):
            with open('random_test_node3.txt', 'r') as f:
                self.ids = [line.strip() for line in f]

        else:
             self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        # self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(npy_file, classes, geo_norm, geo_channel):

        data = np.load(npy_file)
        ori_mask = data[198]
        land_mask = data[199].astype(bool)

        # 处理标签
        if classes == 2:  # 如果是分类
            mask = np.zeros(ori_mask.shape)
            mask[0.1 <= ori_mask] = 1
            mask[~land_mask] = np.nan  # 陆地置为

        else:
            mask = ori_mask
            mask[~land_mask] = np.nan

        # 处理卫星数据
        img = data[:197]

        # 处理经高程、纬度、坡度、坡向数据
        # img--[0:197] geo--[197] mask--[198] landmask--[199] lat--[200] lon--[201] slope--[202] aspect--[203]
        dem, lat, lon, slope, aspect = data[197], data[200], data[201], data[202], data[203]

        dem_max = geo_norm[0]
        dem_min = geo_norm[1]
        dem = (dem - dem_min) / (dem_max - dem_min)

        lat = (lat+90)/180
        lon = (lon+180)/360
        slope = slope/90
        aspect = aspect/360

        geo = np.array([dem, lat, lon, slope, aspect])[geo_channel[0]:geo_channel[1],:,:]

        return img, geo, mask

    def __getitem__(self, idx):
        name = self.ids[idx]
        # print(idx, name)
        img_file = list(self.images_dir.glob(name + '.*'))

        img, geo, mask = self.preprocess(img_file[0], self.classes, self.geo_norm, self.geo_channel)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'geo': torch.as_tensor(geo.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).float().contiguous()
        }

