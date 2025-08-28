import os

import numpy as np

import xarray as xr
from tqdm import tqdm


def load_image_xr(filename):

    ds = xr.open_dataset(filename)
    img = ds['hy_data'].values
    mask = ds['PRECI_cldas'].values

    lat = ds['Latitude'].values
    lon = ds['Longitude'].values
    geo = ds['DEM_elevation'].values
    slope = ds['DEM_slope'].values
    aspect = ds['DEM_aspect'].values
    land_mask = ds['china_land_mask'].values
    # geo = np.array([lat, lon, dem]).transpose(1,2,0)[:,:,2:3]
    # geo = np.array([lat, lon, dem]).transpose(1,2,0)
    real_len = len(ds['TIME_dayOfYear'].values)

    return img, geo, mask, real_len, land_mask, lat, lon, slope, aspect

def preprocess(ori_img, ori_geo, ori_mask, real_len):

    # 处理降水数据
    mask = ori_mask
    mask[real_len:] = np.nan  # 将滑窗采样的256长度数据中用来补充的长度设为无效值

    # 处理卫星数据
    img = ori_img.transpose((2, 0, 1))
    for i in range(img.shape[0]):

        # 获取无效值索引，查找数据中的nan、填补值、-9999.9等无效值，用exam_nan数据记录为True。
        exam_nan = np.isnan(img[i, :, :])  # 本身存在的无效值
        exam_nan[real_len:] = True  # 填补区域视为无效值
        full_index = img[i, :, :] == -9999.9  # 本身存在的填充值-9999.9
        exam_nan[full_index] = True

        # 归一化
        if i < 8:
            min_value, max_value = 0, 500
        elif 8 <= i <21:
            min_value, max_value = 50, 350
        else:
            min_value, max_value = 0, 100
        img[i, :, :] = (img[i, :, :] - min_value) / (max_value - min_value)

        # 为了训练，img中的无效值都置为0，而不是nan，不然无法参与卷积。
        img[i, :, :][exam_nan] = 0

    # 处理海拔数据
    geo = ori_geo
    exam_nan = np.isnan(geo)  # 本身存在的无效值
    exam_nan[real_len:] = True  # 填补区域视为无效值
    full_index = geo.astype(int) < -9999  # 本身存在的填充值-9999.9
    exam_nan[full_index] = True
    geo[exam_nan] = 0

    return img, geo, mask


if __name__ == '__main__':

    # file_dir_ori = r'PRMD_DATA_21-22_new_cldas/'
    # save_dir_ori = r'PRMD_DATA_21-22_new_cldas_npy/'
    file_dir_ori = r'E:\2021_2022_dataset\PRMD_DATA_21-22_v4/'
    save_dir_ori = r'E:\2021_2022_dataset\PRMD_DATA_21-22_npy_v4/'

    for mode in ['/train/', '/val/', '/test/']:
        file_dir = file_dir_ori + mode
        save_dir = save_dir_ori + mode

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        file_list = [file_dir + name for name in os.listdir(file_dir)]
        for i in tqdm(file_list):
            print(i)
            out_path = save_dir+os.path.basename(i).replace('hdf', 'npy')

            img, geo, mask, real_len, land_mask, lat, lon, slope, aspect = load_image_xr(i)
            img, geo, mask = preprocess(img, geo, mask, real_len)
            cat = np.array([geo, mask, land_mask, lat, lon, slope, aspect])
            # npy_data = np.concatenate([img, geo, mask.reshape((1, mask.shape[0], mask.shape[1])), land_mask.reshape((1,mask.shape[0], mask.shape[1]))],axis=0)
            npy_data = np.concatenate([img, cat], axis=0)
            # img--[0:197] geo--[197] mask--[198] landmask--[199] lat--[200] lon--[201] slope--[202] aspect--[203]

            np.save(out_path, npy_data)










