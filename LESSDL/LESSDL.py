# coding: utf-8
from TrainingDataset import create_dataloaders
import os
import torch
from torchvision import transforms
import Models, Utils, Engine
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from scipy import stats
from Utils import ModelName
from ImageInversion import ImageInversion


class LESSDL(object):
    def __init__(self, training_data_dir):
        # Deep learning parameters
        self.NUM_EPOCHS = 1
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 0.001
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Training parameters
        # training_data_dir should contain two files: canopy_reflectance.txt, parameters.txt
        # Each line of canopy_reflectance.txt is canopy reflectance of different bands
        # Each line of parameters.txt is the corresponding parameters, e.g., LAI, Cab, etc.
        # These two files should have exactly the same lines
        self.training_data_dir = training_data_dir
        # param_names stores the name of each line of the parameters.txt
        self.param_names = ["LAI", "SZA", "ALIA", "SOIL", "N", "Cab", "Cw", "Cm", "Cbr", "Car"]
        self.training_percent = 0.8
        self.model_name = ModelName.AlexNet1D  # ann, alexnet1d
        # the target parameter to inverted, it should be in param_names,
        # LAI_FVC means inverting LAI and FVC simultaneously
        self.target_param = "LAI"  # Or LAI

        # computed parameters
        self.train_data_loader = None
        self.test_data_loader = None

    def __get_serial_name(self):
        return self.model_name + "_" + self.target_param + ".pth"

    def __data_prepare(self):
        print(" - Using device:", self.device)
        data_transform = transforms.Compose([
            # transforms.ToTensor()
        ])
        self.train_data_loader, self.test_data_loader = create_dataloaders(self.training_data_dir,
                                                                           transform=data_transform,
                                                                           batch_size=self.BATCH_SIZE,
                                                                           training_percent=self.training_percent,
                                                                           param_names=self.param_names,
                                                                           target_param=self.target_param)
        print(" - Total number of training samples: ", len(self.train_data_loader)*self.BATCH_SIZE)
        print(" - Total number of testing samples: ", len(self.test_data_loader) * self.BATCH_SIZE)

    def train(self):
        self.__data_prepare()
        print(" - Total number of epochs:", self.NUM_EPOCHS)
        print(" - Batch size:", self.BATCH_SIZE)
        print(" - Learning rate:", self.LEARNING_RATE)
        model = None
        target_param_num = len(self.target_param.split("_"))
        if self.model_name == ModelName.ANN:
            model = Models.ANN().to(self.device)
        elif self.model_name == ModelName.AlexNet1D:
            model = Models.AlexNet1D(target_param_num).to(self.device)
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.LEARNING_RATE)
        Engine.train(model, self.train_data_loader, optimizer, loss_fn, self.NUM_EPOCHS, self.device)
        Utils.save_model(model, "model", self.__get_serial_name())

    def test(self):
        self.__data_prepare()
        model = None
        target_param_names = self.target_param.split("_")
        target_param_num = len(target_param_names)
        if self.model_name == ModelName.ANN:
            model = Models.ANN()
        elif self.model_name == ModelName.AlexNet1D:
            model = Models.AlexNet1D(target_param_num)
        model.load_state_dict(torch.load(os.path.join("model", self.__get_serial_name())))
        model = model.to(self.device)
        pred = np.array([])
        reference = np.array([])
        for batch, (reflectance, value) in enumerate(self.test_data_loader):
            reflectance = reflectance.to(self.device)
            with torch.no_grad():
                output = model(reflectance).cpu().numpy()
                pred = np.vstack((pred, output)) if pred.size else output
                reference = np.vstack((reference, value.numpy())) if reference.size else value.numpy()
        for i in range(target_param_num):
            x, y = pred[:, i], reference[:, i]
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            rmse = np.sqrt(np.mean((np.array(x) - np.array(y)) ** 2))
            mean_abs_error = (np.abs(np.array(x) - np.array(y)) / np.array(y)).mean()
            rRMSE = rmse / x.mean() * 100
            print("rmse: ", rmse, "rRMSE", rRMSE, "%", " mean_abs_error: ", mean_abs_error * 100, "%")

            data, x_e, y_e = np.histogram2d(x, y, bins=30, density=True)
            z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([x, y]).T,
                        method="splinef2d",
                        bounds_error=False)
            # To be sure to plot all data
            z[np.where(np.isnan(z))] = 0.0
            if True:
                idx = z.argsort()
                x, y, z = x[idx], y[idx], z[idx]
            plt.plot([0, x.max()], [0, x.max()], '-k', linewidth=0.5)
            plt.scatter(x, y, marker='o', s=10, c=z, cmap='Spectral_r')
            plt.legend(["1:1 line", "$R^2$=%.3f" % (r_value * r_value) + ", RMSE=%.3f" % rmse + ", rRMSE=%.3f" % rRMSE + "%"],
                       loc="upper left")
            # plt.plot(x,y, ".")
            # plt.xlim([0, 6])
            # plt.ylim([0, 6])
            plt.ylabel("Reference " + target_param_names[i])
            plt.xlabel("Predicted " + target_param_names[i])
            plt.show()

    def inversion_image(self, input_image_path, out_dir):
        input_value_scale = 10000
        batch_size = 128
        out_image_path = os.path.join(out_dir, self.target_param)
        img_inversion = ImageInversion(input_image=input_image_path,
                                       out_image=out_image_path,
                                       model_name=self.model_name,
                                       target_param=self.target_param,
                                       input_value_scale = input_value_scale,
                                       batch_size=batch_size)
        img_inversion.invert()
        print(" - Finished, save image to: ", out_image_path)


if __name__ == "__main__":
    lessdl = LESSDL(r"F:\param_ref")
    lessdl.model_name = ModelName.AlexNet1D
    lessdl.target_param = "LAI"
    lessdl.NUM_EPOCHS = 3
    lessdl.train()
    lessdl.test()
    # input_image = r"G:\02-Dataset\实验数据\卫星高光谱数据\承德塞罕坝\ZY1E_AHSI_E117.33_N42.80_20210917_010559_L1A0000336326_Check\vn_reflectance_subregion"
    # out_dir = r"F:\TMP"
    # lessdl.inversion_image(input_image, out_dir)
