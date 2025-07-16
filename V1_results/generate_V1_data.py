import numpy as np
import uproot
import awkward
import pandas
import time
from tqdm import tqdm

def minmax(arr):
    return (arr-min(arr))/(max(arr)-min(arr)), (np.astype(min(arr), np.float32), np.astype(max(arr), np.float32))

def log_minmax(arr):
    return ((np.log10(arr)-np.log10(min(arr)))/(np.log10(max(arr))-np.log10(min(arr))),
            (np.astype(np.log10(min(arr)), np.float32),
             np.astype(np.log10(max(arr)), np.float32)))

tuple_path = "https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/GamGam/Data/"
save_path = "/Users/oscarfuentes/masters_project/baler_sem1/workspaces/higgs/data/"

samples = [tuple_path + "data_" + letter + ".GamGam.root" for letter in ["A", "B", "C", "D"]]

tightIDs = np.empty((0, 2))
etcones = np.empty((0, 2))
full_data = np.empty((0, 8))
full_norm_data = np.empty((0,8))
length = 0

total_data = 0

for path, letter in zip(samples, ["A", "B", "C", "D"]):
    print(f"Processing data {letter}")
    start = time.time()

    with uproot.open(path + ":mini") as t:
        tree = t
        print(f"File open, took {time.time() - start:.2f}s")

        for data in tree.iterate(
            [
                "photon_pt",
                "photon_eta",
                "photon_phi",
                "photon_E",
                "photon_isTightID",
                "photon_etcone20"
            ],
            library="pd",
        ):
            total_data += len(data)
            print("Applying cuts...")
            lens = data.photon_pt.apply(len)
            len_mask = (lens == 2).to_numpy()
            data = data[len_mask]

            print("Stacking...")
            full_data = np.vstack((full_data, np.column_stack([data.photon_pt,
                                                            data.photon_eta,
                                                            data.photon_phi,
                                                            data.photon_E])))
            tightIDs = np.vstack((tightIDs, data.photon_isTightID))
            etcones = np.vstack((etcones, data.photon_etcone20))

            length += len(data)

        print(f"Finished processing {letter}\n\nTook {time.time() - start:.2f}s")
        print(f"Current Length {length}\n")

print(f"Total precut length: {total_data}")

full_data = np.array(full_data, dtype=np.float32)
tightIDs = np.array(tightIDs, dtype=np.int32)
etcones = np.array(etcones, dtype=np.float32)

pt1, scale0 = log_minmax(full_data[:,0])
pt2, scale1 = log_minmax(full_data[:,1])
eta1, scale2 = minmax(full_data[:,2])
eta2, scale3 = minmax(full_data[:,3])
phi1, scale4 = minmax(full_data[:,4])
phi2, scale5 = minmax(full_data[:,5])
E1, scale6 = log_minmax(full_data[:,6])
E2, scale7 = log_minmax(full_data[:,7])
        
full_norm_data = np.vstack((full_norm_data, np.column_stack([pt1, pt2,
                                                        eta1, eta2,
                                                        phi1, phi2,
                                                        E1, E2])))
scales = np.column_stack([[scale0, scale1], [scale2, scale3], [scale4, scale5], [scale6, scale7]])

print(type(full_data))
print(type(full_data[0]))
print(type(full_data[0][0]))
print(f"Final data shape: {full_data.shape}")
print(type(tightIDs))
print(type(tightIDs[0]))
print(type(tightIDs[0][0]))
print(tightIDs[0][0])
print(f"Final tightIDs shape: {tightIDs.shape}")

names = ["pt1", "pt2", "eta1", "eta2", "phi1", "phi2", "E1", "E2"]

np.savez(save_path+"V1_precut_original.npz", data=full_data, names=names)
np.savez(save_path+"V1_precut_normalised.npz", data=full_norm_data, names=names)
np.savez(save_path+"V1_precut_scales.npz", data=scales, names=names)
np.savez(save_path+"V1_photon_isTightID.npz", data=tightIDs, names=np.array(["ID1", "ID2"]))
np.savez(save_path+"V1_photon_etcone20.npz", data=etcones, names=np.array(["etcone1", "etcone2"]))
