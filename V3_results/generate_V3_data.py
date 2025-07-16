import numpy as np

def load_data():
    with np.load("/Users/oscarfuentes/masters_project/baler_sem1/workspaces/higgs/data/V2_precut_original.npz") as datafile:
        data = datafile["data"]
        names = datafile["names"]
    with np.load("/Users/oscarfuentes/masters_project/baler_sem1/workspaces/higgs/data/photon_etcones.npz") as etconefile:
        etcones = etconefile["data"]
    return data, etcones, names

def log_minmax(arr):
    logmin = np.log10(np.min(arr))
    logmax = np.log10(np.max(arr))
    norm_arr = (np.log10(arr) - logmin)/(logmax - logmin)
    return norm_arr, (logmin, logmax)

def minmax(arr):
    norm_arr = (arr - np.min(arr))/(np.max(arr) - np.min(arr))
    return norm_arr, (np.min(arr), np.max(arr))

def cut_isolation(data, etcones):
    return data[(etcones[:,0] < 4000) & (etcones[:,1] < 4000)]

def main():
    data, etcones, names = load_data()
    if len(data[:,0]) == len(etcones[:,0]):
        print(f"len(data) == {len(data[:,0])}\nlen(etcones) == {len(etcones[:,0])}")
        cut_data = cut_isolation(data, etcones)

        norm_arr = []
        scales = []
        for i, arr in enumerate(cut_data.T):
            if (i < 2) or (i > 5): 
                dats, scale = log_minmax(arr)
            else:
                dats, scale = minmax(arr)
            norm_arr.append(dats)
            scales.append(np.array(scale))

        scales = np.array([np.stack([scales[0], scales[1]]),
                           np.stack([scales[2], scales[3]]),
                           np.stack([scales[4], scales[5]]),
                           np.stack([scales[6], scales[7]]),
                           ])

        np.savez("/Users/oscarfuentes/masters_project/baler_sem1/workspaces/higgs/data/V3_precut_normalised.npz",
                 data=np.column_stack(norm_arr),
                 names=names)
        np.savez("/Users/oscarfuentes/masters_project/baler_sem1/workspaces/higgs/data/V3_precut_original.npz",
                 data=cut_data,
                 names=names)
        np.savez("/Users/oscarfuentes/masters_project/baler_sem1/workspaces/higgs/data/V3_precut_scales.npz",
                 data=np.column_stack(scales),
                 names=names)
        return 0
    else:
        print(f"LenError:\nlen(data) == {len(data[:,0])}\nlen(etcones) == {len(etcones[:,0])}")

if __name__ == "__main__":
    main()
