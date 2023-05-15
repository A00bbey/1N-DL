import prosail
import numpy as np
import random
import scipy.stats as stats
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl


def guass_distribution(region, mu, sigma, classes):
    lower, upper = region
    guass = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    para = guass.rvs(classes)
    return para


def uniform_distribution(region, classes):
    lower, upper = region
    para = [np.random.uniform(lower, upper) for i in range(classes)]
    return para


def read_zy1e_wavelength():
    wavelength = pd.read_csv(r"./utils/VNIR.csv")
    return wavelength["wavelength"].values


wavelength = read_zy1e_wavelength()
print(wavelength[1:])
wavelength = list(map(lambda x: int(x)-400, wavelength[1:]))

fref = open(r"F:\param_ref\reflectance.txt", "w")
fparam = open(r"F:\param_ref\parameters.txt", "w")

def simulation():
    phi = 0
    tto = 0
    Car = 4
    hspot = 0.1
    soil = pd.read_csv(r"./utils/soil_spectrum.csv", header=0, index_col=0)
    dataset = []
    parameters = []
    for row_name in soil.iterrows():
        print(row_name[0])
        rsoil0 = soil.loc[row_name[0]].values
        # rsoil0 = soil.loc["Wet clay"].values
        N = guass_distribution([1.5, 3], 1.5, 1, 4)
        Cab = guass_distribution([20, 80], 50, 30, 20)
        Cbr = uniform_distribution([0, 1.5], 2)
        Cw = guass_distribution([0, 0.07], 0.035, 0.01, 1)
        Cm = guass_distribution([0, 0.01], 0.008, 0.001, 1)
        Cant = guass_distribution([0, 5], 2.5, 1.5, 3)
        ALIA = uniform_distribution([30, 70], 3)
        SZA = uniform_distribution([20, 60], 3)
        LAI = uniform_distribution([0.05, 7], 20)
        index = 0
        for param in product(N, Cab, Cbr, Cw, Cm, Cant, ALIA, SZA, LAI):
            if index % 1000 == 0:
                print(index)
            index += 1
            n, cab, cbr, cw, cm, cant, lidfa, tts, lai = param
            s = prosail.run_prosail(n, cab, Car, cbr, cw, cm, lai, lidfa, hspot, tts, tto, phi,
                                    cant, alpha=40.0, typelidf=2, lidfb=0.0,
                                    factor="ALLALL", prospect_version='D', rsoil0=rsoil0,
                                    rsoil=None, psoil=None, soil_spectrum1=None, soil_spectrum2=None
                                    )
            # dataset.append(s[-4])
            # parameters.append(list(param + (1 - s[1],)))
            zy1e_ref = s[-4][wavelength]
            if not np.isnan(zy1e_ref[0]):
                fref.write(" ".join(map(lambda x: "%.4f" % x, zy1e_ref))+"\n")
                fparam.write(" ".join(map(lambda x: "%.4f"%x, param+(lai*cab*0.01, 1 - s[1],)))+"\n")  # CCC
        fref.flush()
        fparam.flush()
    fref.close()
    fparam.close()
    return dataset, parameters


hyper_spectrum, parameters = simulation()
# plt.rcParams['font.sans-serif'] = ['SimHei']

#
# print("光谱模拟数量：{}".format(len(hyper_spectrum)))
#
# plt.figure(figsize=(10, 5))
# mpl.rcParams['font.sans-serif'] = ['SimHei']
# for s in range(2000):
#     plt.plot(range(400, 2501), hyper_spectrum[s])
# plt.title("PROSAIL模拟光谱")
# plt.xlabel("wavelength(nm)")
# plt.ylabel("reflectance")
# plt.show()
