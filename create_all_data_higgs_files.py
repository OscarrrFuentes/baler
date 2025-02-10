import uproot
import numpy as np
import pandas as pd


def filter_len_2(series):
    lens = series.apply(len)
    mask = (lens == 2).to_numpy()
    return series[mask].to_numpy()


def cut_photon_ID(photon_isTightID):
    return photon_isTightID[0] == True and photon_isTightID[1] == True


data_path = "/gluster/data/atlas/baler/datasets/atlas/education_13TeV/GamGam/Data"
save_path = "/gluster/home/ofrebato/baler/workspaces/higgs/data"

samples = [data_path + "/data_" + letter + ".GamGam.root" for letter in ["A", "B", "C", "D"]]

all_pts = np.empty((0, 2))
all_etas = np.empty((0, 2))
all_phis = np.empty((0, 2))
all_Es = np.empty((0, 2))

for path in samples:
    with uproot.open(path + ":mini") as t:
        tree = t

    for data in tree.iterate(
        ["photon_pt", "photon_eta", "photon_phi", "photon_E", "photon_isTightID", "photon_etcone20"]
    ):
        data = data[np.vectorize(cut_photon_ID)(data.photon_isTightID)]

        all_pts = np.vstack((all_pts, filter_len_2(data.photon_pt)))
        all_etas = np.vstack((all_etas, filter_len_2(data.photon_eta)))
        all_phis = np.vstack((all_phis, filter_len_2(data.photon_phi)))
        all_Es = np.vstack((all_Es, filter_len_2(data.photon_E)))

np.savez(save_path + "/photon_pts.npz", data=all_pts, names=["photon_pt1", "photon_pt2"])
np.savez(save_path + "/photon_etas.npz", data=all_etas, names=["photon_eta1", "photon_eta2"])
np.savez(save_path + "/photon_phis.npz", data=all_phis, names=["photon_phi1", "photon_phi2"])
np.savez(save_path + "/photon_Es.npz", data=all_Es, names=["photon_E1", "photon_E2"])
