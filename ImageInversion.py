# coding: utf-8

import gdal
import Models
import torch
import os
import numpy as np
from Utils import ModelName
from tqdm.auto import tqdm


def save_to_hdr(npArray,param_num, dstFilePath, geoTransform, rpc):
    dshape = npArray.shape
    format = "ENVI"
    driver = gdal.GetDriverByName(format)
    dst_ds = driver.Create(dstFilePath, dshape[1], dshape[0], param_num, gdal.GDT_Float32)
    if (rpc is not None) and len(rpc) > 0:
        dst_ds.SetMetadata(rpc, 'RPC')
    else:
        if geoTransform is not None:
            dst_ds.SetGeoTransform(geoTransform)
    for i in range(param_num):
        dst_ds.GetRasterBand(i+1).WriteArray(npArray[:, :, i])
    dst_ds = None


class ImageInversion(object):
    def __init__(self, input_image, out_image, model_name, target_param, input_value_scale=10000, batch_size=128):
        self.input_image = input_image
        self.out_image = out_image
        self.model_name = model_name
        self.target_param = target_param
        self.batch_size = batch_size
        self.input_value_scale = input_value_scale

    def __get_serial_name(self):
        return self.model_name + "_" + self.target_param + ".pth"

    def invert(self):
        dataset = gdal.Open(self.input_image)
        x_size, y_size, num_bands = dataset.RasterXSize, dataset.RasterYSize, dataset.RasterCount
        rpc = dataset.GetMetadata('RPC')
        geo_transform = None
        if len(rpc) == 0:
            geo_transform = dataset.GetGeoTransform()
        band_data = []
        for i in range(0, num_bands):
            band = dataset.GetRasterBand(i + 1)
            band_data.append(band.ReadAsArray(0, 0, band.XSize, band.YSize))
        band_data = np.array(band_data)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        param_names = self.target_param.split("_")
        param_num = len(param_names)
        model = None
        if self.model_name == ModelName.ANN:
            model = Models.ANN()
        elif self.model_name == ModelName.AlexNet1D:
            model = Models.AlexNet1D(param_num)
        model.load_state_dict(torch.load(os.path.join("model", self.__get_serial_name())))
        model = model.to(device)

        param = np.zeros((y_size, x_size, param_num))
        for row in tqdm(range(y_size)):
            for col in range(0, x_size, self.batch_size):
                col_start = col
                col_end = (col + self.batch_size) if (col + self.batch_size) < x_size else x_size-1
                reflectance = torch.from_numpy(np.transpose(band_data[:, row, col_start:col_end]/self.input_value_scale).astype(np.float32))
                with torch.no_grad():
                    output = model(reflectance.to(device)).cpu().numpy()
                    param[row, col_start:col_end] = output
        save_to_hdr(param, param_num, self.out_image, geo_transform, rpc)
