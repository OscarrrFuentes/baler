import numpy as np
import uproot
import awkward
import pandas
import time
from tqdm import tqdm


def cut_photon_pt(photon_pt):
    return (photon_pt[:,0] > 40000) & (photon_pt[:,1] > 30000)

def cut_photon_eta_transition(photon_eta):
    return ((abs(photon_eta[:,0])>1.52) | (abs(photon_eta[:,0])<1.37)) & ((abs(photon_eta[:,1])>1.52) | (abs(photon_eta[:,1])<1.37))

def cut_tightID(tightIDs):
    # with np.load("/Users/oscarfuentes/masters_project/baler_sem1/workspaces/higgs/data/V4_photon_isTightID.npz") as f:
    #     tightIDs = f["data"].astype(bool)
    return tightIDs[:,0] & tightIDs[:,1]

def cut_isolation_et(photon_etcone20):
    return (photon_etcone20[:,0]<4000) & (photon_etcone20[:,1]<4000)

def minmax(arr):
    return (arr-min(arr))/(max(arr)-min(arr)), (np.astype(min(arr), np.float32), np.astype(max(arr), np.float32))

def log_minmax(arr):
    return ((np.log10(arr)-np.log10(min(arr)))/(np.log10(max(arr))-np.log10(min(arr))),
            (np.astype(np.log10(min(arr)), np.float32),
             np.astype(np.log10(max(arr)), np.float32)))

tuple_path = "https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/GamGam/Data/"
save_path = "/Users/oscarfuentes/masters_project/baler_sem1/workspaces/higgs/data/"

samples = [tuple_path + "data_" + letter + ".GamGam.root" for letter in ["A", "B", "C", "D"]]

full_data = np.empty((0, 8))
full_norm_data = np.empty((0,8))
length = 0

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
            print("Applying cuts...\n")
            lens = data.photon_pt.apply(len)
            len_mask = (lens == 2).to_numpy()
            data = data[len_mask]
            print("Len cut complete")
            data = data[cut_tightID(data.photon_isTightID.to_numpy())]
            print("Tight ID cut complete")
            data = data[cut_isolation_et(data.photon_etcone20.to_numpy())]
            print("Isolation cut complete")
            data = data[cut_photon_pt(data.photon_pt.to_numpy())]
            print("pt cut complete")
            data = data[cut_photon_eta_transition(data.photon_eta.to_numpy())]
            print("eta transition cut complete")

            print("\nStacking...")
            full_data = np.vstack((full_data, np.column_stack([data.photon_pt,
                                                            data.photon_eta,
                                                            data.photon_phi,
                                                            data.photon_E])))

            length += len(data)

        print(f"Finished processing {letter}\n\nTook {time.time() - start:.2f}s")
        print(f"Current Length {length}\n")

full_data = np.array(full_data, dtype=np.float32)

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

names = ["pt1", "pt2", "eta1", "eta2", "phi1", "phi2", "E1", "E2"]

np.savez(save_path+"V4_precut_original.npz", data=full_data, names=names)
np.savez(save_path+"V4_precut_normalised.npz", data=full_norm_data, names=names)
np.savez(save_path+"V4_precut_scales.npz", data=scales, names=names)
