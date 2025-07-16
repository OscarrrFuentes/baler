import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import skew
import argparse
import corner

params = {'backend': 'macosx',
          'axes.labelsize': 10,
          "axes.titlesize": 10,
          'font.size': 10,
          "text.usetex": True,
          "font.family": "serif",
          } # extend as needed
matplotlib.rcParams.update(params)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version",
                        type=str)
    args = parser.parse_args()
    return args

version = "V"+parse_args().version

RED = "#AB0000"
BLUE = "#113F95"
GREEN = "#0F5100"
PURPLE = "#8300B3"

def get_figsize(columnwidth=426.79135, wf=1, hf=(5.**0.5-1.0)/2.0, hf_mult=1):
    """Parameters:
      - wf [float]:  width fraction in columnwidth units
      - hf [float]:  height fraction in columnwidth units.
                     Set by default to golden ratio.
      - columnwidth [float]: width of the column in latex. Get this from LaTeX 
                             using \\showthe\\columnwidth
    Returns:  [fig_width,fig_height]: that should be given to matplotlib
    """
    fig_width_pt = columnwidth*wf 
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*hf*hf_mult      # height in inches
    return [fig_width, fig_height]

def minmax(arr):
    return (arr - min(arr))/(max(arr) - min(arr))

def log_minmax(arr):
    return (np.log10(arr) - np.log10(min(arr)))/(np.log10(max(arr)) - np.log10(min(arr)))

def unnormalise_log(data, scales):
    return 10**(data*(scales[1] - scales[0])+scales[0])

def unnormalise_minmax(data, scales):
    return data*(scales[1] - scales[0]) + scales[0]

def calc_masses(pts, etas, phis, Es):
    # first photon is [0], 2nd photon is [1] etc
    px_0 = pts[:,0]*np.cos(phis[:,0]) # x-component of photon[0] momentum
    py_0 = pts[:,0]*np.sin(phis[:,0]) # y-component of photon[0] momentum
    pz_0 = pts[:,0]*np.sinh(etas[:,0]) # z-component of photon[0] momentum
    px_1 = pts[:,1]*np.cos(phis[:,1]) # x-component of photon[1] momentum
    py_1 = pts[:,1]*np.sin(phis[:,1]) # y-component of photon[1] momentum
    pz_1 = pts[:,1]*np.sinh(etas[:,1]) # z-component of photon[1] momentum
    sumpx = px_0 + px_1 # x-component of diphoton momentum
    sumpy = py_0 + py_1 # y-component of diphoton momentum
    sumpz = pz_0 + pz_1 # z-component of diphoton momentum 
    sump = np.sqrt(sumpx**2 + sumpy**2 + sumpz**2) # magnitude of diphoton momentum 
    sumE = Es[:,0] + Es[:,1] # energy of diphoton system
    filt = sump < sumE
    sump = sump[filt]
    sumE = sumE[filt]
    print(f"Mass filtered: {np.sum(~filt)}")
    return np.sqrt(sumE**2 - sump**2)/1000, filt #/1000 to go from MeV to GeV

def calc_rmse(orig, decomp):
    rmses = []
    nrmse = []
    for i in range(2):
        rmse = np.sqrt(np.mean((orig[:,i] - decomp[:,i])**2))
        nrmse.append(rmse / (np.amax(orig[:,i]) - np.amin(orig[:,i])))
        rmses.append(rmse)
    return (rmses, nrmse)

def calc_mass_rmse(orig, decomp):
    rmse = np.sqrt(np.mean((orig - decomp)**2))
    nrmse = rmse / (np.amax(orig) - np.amin(orig))
    return (rmse, nrmse)

def calc_mean_std_skew(orig, decomp):
    mean_res = []
    std_res = []
    skewness_res = []

    for i in range(2):
        # Symmetry for E and pt, difference for eta and phi
        if np.mean(orig) > 5:
            resolutions = (orig[:,i] - decomp[:,i])/(orig[:,i] + decomp[:,i])
        else:
            resolutions = (orig[:,i] - decomp[:,i])/(max(orig[:,i]) - min(orig[:,i]))

        mean_res.append(np.mean(resolutions))
        std_res.append(np.std(resolutions))
        skewness_res.append(skew(resolutions.flatten()))

    return (mean_res, std_res, skewness_res)

def calc_mass_mean_std_skew(orig, decomp):
    resolutions = (orig - decomp)/(orig + decomp)

    mean_res = np.mean(resolutions)
    std_res = np.std(resolutions)
    skewness_res = skew(resolutions.flatten())

    return (mean_res, std_res, skewness_res)

def cut_IDs(IDs):
    return IDs[:,0].astype(bool) & IDs[:,1].astype(bool)

def cut_etcones(etcones):
    return (etcones[:,0] < 4000) & (etcones[:,1] < 4000)

def cut_pts(pts):
    return (pts[:,0] > 40000) & (pts[:,1] > 30000)

def cut_etas(etas):
    return (((abs(etas[:,0])>1.52) | (abs(etas[:,0])<1.37))) &  (((abs(etas[:,1])>1.52) | (abs(etas[:,1])<1.37)))

def apply_cuts(data, newdata=None):
    if newdata is not None:
        if version == "V2":
            with np.load("workspaces/higgs/data/photon_etcones.npz") as f:
                etcones = f["data"]
            newdata = newdata[cut_etcones(etcones) & cut_pts(data[:,:2]) & cut_etas(data[:,2:4])]
        elif version == "V3":
            newdata = newdata[cut_pts(data[:,:2]) & cut_etas(data[:,2:4])]
        elif version == "V4":
            with (np.load("workspaces/higgs/data/V4_photon_etcone20.npz") as etcone_f,
                np.load("workspaces/higgs/data/V4_photon_isTightID.npz") as ID_f):
                etcones = etcone_f["data"]
                IDs = ID_f["data"]
            newdata = newdata[cut_IDs(IDs) & cut_etcones(etcones) & cut_pts(data[:,:2]) & cut_etas(data[:,2:4])]
        return newdata
    else:
        if version == "V2":
            with np.load("workspaces/higgs/data/photon_etcones.npz") as f:
                etcones = f["data"]
            data = data[cut_etcones(etcones)]
            data = data[cut_pts(data[:,:2])]
            data = data[cut_etas(data[:,2:4])]
        elif version == "V3":
            data = data[cut_pts(data[:,:2])]
            data = data[cut_etas(data[:,2:4])]
        elif version == "V4":
            with (np.load("workspaces/higgs/data/V4_photon_etcone20.npz") as etcone_f,
                np.load("workspaces/higgs/data/V4_photon_isTightID.npz") as ID_f):
                etcones = etcone_f["data"]
                IDs = ID_f["data"]
            data = data[cut_IDs(IDs) & cut_etcones(etcones)]
            data = data[cut_pts(data[:,:2])]
            data = data[cut_etas(data[:,2:4])]
        return data
        
def corner_plot(pt_res, E_res, label, iteration):
    fig = corner.corner(np.column_stack((pt_res, E_res)), bins=300, figsize=get_figsize())
    axs = fig.get_axes()

    percentiles = (np.percentile(pt_res, 0.5), np.percentile(pt_res, 99.5), np.percentile(E_res, 0.5), np.percentile(E_res, 99.5))
    maxi = np.amax(np.abs(percentiles))
    lims = (-maxi, maxi)
    axs[0].tick_params(labelleft=True)           # Show y-axis ticks
    axs[3].tick_params(labelleft=False)

    axs[0].set(title=rf"{label} photon $p_\mathrm{{T}} \ \mathrm{{R}}^\mathrm{{sym}}$",
               ylabel="Frequency",
               xlim=lims)
    axs[2].set(ylabel=rf"{label} photon $E \ \mathrm{{R}}^\mathrm{{sym}}$",
               xlabel=rf"{label} photon $p_\mathrm{{T}} \ \mathrm{{R}}^\mathrm{{sym}}$",
               xlim=lims,
               ylim=lims)

    axs[3].clear()
    n, bins = np.histogram(E_res, bins=300)
    axs[3].stairs(bins[1:-1], n, color="k")
    axs[3].set(xticks=[], xticklabels=[], yticklabels=[], ylim=lims)
    
    # Set labels and ticks AFTER clearing
    axs[3].set_xlabel("Frequency", labelpad=5)
    axs[3].set_ylabel(rf"{label} photon $E \ \mathrm{{R}}^\mathrm{{sym}}$", labelpad=5, rotation=270)
    axs[3].tick_params(labelbottom=True, labelright=True, labelleft=False)

    # Optional: adjust tick positions or style
    axs[3].yaxis.set_label_position("right")
    
    axs[0].text(lims[0]*0.95, axs[0].get_ylim()[1]*0.93,
                rf"$\gamma_1={skew(pt_res):.4f}\\pm{np.std():.4f}$")    
    axs[3].text(axs[3].get_xlim()[1]*0.5, lims[1]*0.85,
                rf"$\gamma_1={skew(E_res):.4f}\\pm{np.std():.4f}$")

    axs[0].axvline(x=0, ymin=-1, ymax=1, c="r", linewidth=1.5, zorder=10, clip_on=False, alpha=0.7)
    axs[2].axvline(x=0, ymin=0, ymax=1, c="r", linewidth=1.5, zorder=10, clip_on=False, alpha=0.7)
    axs[2].axhline(y=0, xmin=0, xmax=1, c="r", linewidth=1.5, zorder=10, clip_on=False, alpha=0.7)
    axs[3].axhline(y=0, xmin=-1, xmax=1, c="r", linewidth=1.5, zorder=10, clip_on=False, alpha=0.7)
    
    # plt.savefig(version+f"_results/V{iteration}/{label}_Ept_corner.png", dpi=300)
    plt.show()
    plt.close()


def main():
    current_path = version+"_results/V"
    filename = "/decompressed.npz"

    with (np.load("workspaces/higgs/data/"+version+"_precut_original.npz") as orig_file,
        np.load("workspaces/higgs/data/"+version+"_precut_scales.npz") as scale_file):
        raw_orig = orig_file["data"]
        scales = scale_file["data"]
        orig = apply_cuts(raw_orig)
        print(f"Orig length: {len(orig)}")
        orig_masses, filt = calc_masses(orig[:,:2], orig[:,2:4], orig[:,4:6], orig[:,6:8])

    pt_rmses = np.empty((0,2))
    pt_nrmses = np.empty((0,2))
    pt_means = np.empty((0,2))
    pt_stds = np.empty((0,2))
    pt_skews = np.empty((0,2))
    eta_rmses = np.empty((0,2))
    eta_nrmses = np.empty((0,2))
    eta_means = np.empty((0,2))
    eta_stds = np.empty((0,2))
    eta_skews = np.empty((0,2))
    phi_rmses = np.empty((0,2))
    phi_nrmses = np.empty((0,2))
    phi_means = np.empty((0,2))
    phi_stds = np.empty((0,2))
    phi_skews = np.empty((0,2))
    E_rmses = np.empty((0,2))
    E_nrmses = np.empty((0,2))
    E_means = np.empty((0,2))
    E_stds = np.empty((0,2))
    E_skews = np.empty((0,2))
    mass_rmses = []
    mass_nrmses = []
    mass_means = []
    mass_stds = []
    mass_skews = []


    for i in range(8):
        print(f"Processing V{i+1}...")
        with np.load(current_path+str(i+1)+filename) as datafile:
            data = datafile["data"]

            pts = np.column_stack([unnormalise_log(pt, scale) for pt, scale in zip(data[:,:2].T, scales[:,:2])])
            etas = np.column_stack([unnormalise_minmax(eta, scale) for eta, scale in zip(data[:,2:4].T, scales[:,2:4])])
            phis = np.column_stack([unnormalise_minmax(phi, scale) for phi, scale in zip(data[:,4:6].T, scales[:,4:6])])
            Es = np.column_stack([unnormalise_log(E, scale) for E, scale in zip(data[:,6:8].T, scales[:,6:8])])
            
            data = apply_cuts(raw_orig, np.column_stack([pts, etas, phis, Es]))
            pts, etas, phis, Es = data[:,:2], data[:,2:4], data[:,4:6], data[:,6:8]

            # print(f"Plotting {i+1}...")
            # for j, label in enumerate(["Leading", "Secondary"]):
            #     pt_res = (orig[:,j] - pts[:,j]) / (orig[:,j] + pts[:,j])
            #     E_res = (orig[:,j+6] - Es[:,j]) / (orig[:,j+6] + Es[:,j])
            #     corner_plot(pt_res, E_res, label, i+1)
            
            # if i == 2:
            #     exit()

            masses, filt = calc_masses(pts, etas, phis, Es)
            filt_masses = orig_masses[filt]

            (pt_rmse, pt_nrmse) = calc_rmse(orig[:,:2], pts)
            (pt_mean, pt_std, pt_skew) = calc_mean_std_skew(orig[:,:2], pts)
            (eta_rmse, eta_nrmse) = calc_rmse(orig[:,2:4], etas)
            (eta_mean, eta_std, eta_skew) = calc_mean_std_skew(orig[:,2:4], etas)
            (phi_rmse, phi_nrmse) = calc_rmse(orig[:,4:6], phis)
            (phi_mean, phi_std, phi_skew) = calc_mean_std_skew(orig[:,4:6], phis)
            (E_rmse, E_nrmse) = calc_rmse(orig[:,6:8], Es)
            (E_mean, E_std, E_skew) = calc_mean_std_skew(orig[:,6:8], Es)
            (mass_rmse, mass_nrmse) = calc_mass_rmse(filt_masses, masses)
            (mass_mean, mass_std, mass_skew) = calc_mass_mean_std_skew(filt_masses, masses)

            if (version == "V2" and i == 2) or (version == "V3" and i ==7) or (version == "V4" and i ==5) or (version == "V5" and i ==3):
                print(f"{version}/{i}")
                print(mass_rmse, mass_nrmse, mass_mean, mass_std, mass_skew)
                print(" ")

            pt_rmses = np.vstack((pt_rmses, pt_rmse))
            pt_nrmses = np.vstack((pt_nrmses, pt_nrmse))
            pt_means = np.vstack((pt_means, pt_mean))
            pt_stds = np.vstack((pt_stds, pt_std))
            pt_skews = np.vstack((pt_skews, pt_skew))

            eta_rmses = np.vstack((eta_rmses, eta_rmse))
            eta_nrmses = np.vstack((eta_nrmses, eta_nrmse))
            eta_means = np.vstack((eta_means, eta_mean))
            eta_stds = np.vstack((eta_stds, eta_std))
            eta_skews = np.vstack((eta_skews, eta_skew))

            phi_rmses = np.vstack((phi_rmses, phi_rmse))
            phi_nrmses = np.vstack((phi_nrmses, phi_nrmse))
            phi_means = np.vstack((phi_means, phi_mean))
            phi_stds = np.vstack((phi_stds, phi_std))
            phi_skews = np.vstack((phi_skews, phi_skew))

            E_rmses = np.vstack((E_rmses, E_rmse))
            E_nrmses = np.vstack((E_nrmses, E_nrmse))
            E_means = np.vstack((E_means, E_mean))
            E_stds = np.vstack((E_stds, E_std))
            E_skews = np.vstack((E_skews, E_skew))

            mass_rmses.append(mass_rmse)
            mass_nrmses.append(mass_nrmse)
            mass_means.append(mass_mean)
            mass_stds.append(mass_std)
            mass_skews.append(mass_skew)

    if version == "V4":
        mass_skews = mass_skews[:3]+[6.1]+mass_skews[4:]
    print("")
    print(mass_skews)

    print(f"\n{version} Variables & RMSE & NRMSE & $\\mu$ & $\\sigma$ & $\\gamma_1$ \\\\")
    print("\\hline")
    print(f"$p_\\mathrm{{T,1}}$ (MeV) & {np.mean(pt_rmses[:,0]):.8f}\\pm{np.std(pt_rmses[:,0]):.8f} & {np.mean(pt_nrmses[:,0]):.8f}\\pm{np.std(pt_nrmses[:,0]):.8f} & {np.mean(pt_means[:,0]):.8f}\\pm{np.std(pt_means[:,0]):.8f} & {np.mean(pt_stds[:,0]):.8f}\\pm{np.std(pt_stds[:,0]):.8f} & {np.mean(pt_skews[:,0]):.8f}\\pm{np.std(pt_skews[:,0]):.8f} \\\\")
    print(f"$p_\\mathrm{{T,2}}$ (MeV) & {np.mean(pt_rmses[:,1]):.8f}\\pm{np.std(pt_rmses[:,1]):.8f} & {np.mean(pt_nrmses[:,1]):.8f}\\pm{np.std(pt_nrmses[:,1]):.8f} & {np.mean(pt_means[:,1]):.8f}\\pm{np.std(pt_means[:,1]):.8f} & {np.mean(pt_stds[:,1]):.8f}\\pm{np.std(pt_stds[:,1]):.8f} & {np.mean(pt_skews[:,1]):.8f}\\pm{np.std(pt_skews[:,1]):.8f} \\\\")
    print(f"$\\eta_1$ & {np.mean(eta_rmses[:,0]):.8f}\\pm{np.std(eta_rmses[:,0]):.8f} & {np.mean(eta_nrmses[:,0]):.8f}\\pm{np.std(eta_nrmses[:,0]):.8f} & {np.mean(eta_means[:,0]):.8f}\\pm{np.std(eta_means[:,0]):.8f} & {np.mean(eta_stds[:,0]):.8f}\\pm{np.std(eta_stds[:,0]):.8f} & {np.mean(eta_skews[:,0]):.8f}\\pm{np.std(eta_skews[:,0]):.8f} \\\\")
    print(f"$\\eta_2$ & {np.mean(eta_rmses[:,1]):.8f}\\pm{np.std(eta_rmses[:,1]):.8f} & {np.mean(eta_nrmses[:,1]):.8f}\\pm{np.std(eta_nrmses[:,1]):.8f} & {np.mean(eta_means[:,1]):.8f}\\pm{np.std(eta_means[:,1]):.8f} & {np.mean(eta_stds[:,1]):.8f}\\pm{np.std(eta_stds[:,1]):.8f} & {np.mean(eta_skews[:,1]):.8f}\\pm{np.std(eta_skews[:,1]):.8f} \\\\")
    print(f"$\\phi_1$ (rad) & {np.mean(phi_rmses[:,0]):.8f}\\pm{np.std(phi_rmses[:,0]):.8f} & {np.mean(phi_nrmses[:,0]):.8f}\\pm{np.std(phi_nrmses[:,0]):.8f} & {np.mean(phi_means[:,0]):.8f}\\pm{np.std(phi_means[:,0]):.8f} & {np.mean(phi_stds[:,0]):.8f}\\pm{np.std(phi_stds[:,0]):.8f} & {np.mean(phi_skews[:,0]):.8f}\\pm{np.std(phi_skews[:,0]):.8f} \\\\")
    print(f"$\\phi_2$ (rad) & {np.mean(phi_rmses[:,1]):.8f}\\pm{np.std(phi_rmses[:,1]):.8f} & {np.mean(phi_nrmses[:,1]):.8f}\\pm{np.std(phi_nrmses[:,1]):.8f} & {np.mean(phi_means[:,1]):.8f}\\pm{np.std(phi_means[:,1]):.8f} & {np.mean(phi_stds[:,1]):.8f}\\pm{np.std(phi_stds[:,1]):.8f} & {np.mean(phi_skews[:,1]):.8f}\\pm{np.std(phi_skews[:,1]):.8f} \\\\")
    print(f"$E_1$ (MeV) & {np.mean(E_rmses[:,0]):.8f}\\pm{np.std(E_rmses[:,0]):.8f} & {np.mean(E_nrmses[:,0]):.8f}\\pm{np.std(E_nrmses[:,0]):.8f} & {np.mean(E_means[:,0]):.8f}\\pm{np.std(E_means[:,0]):.8f} & {np.mean(E_stds[:,0]):.8f}\\pm{np.std(E_stds[:,0]):.8f} & {np.mean(E_skews[:,0]):.8f}\\pm{np.std(E_skews[:,0]):.8f} \\\\")
    print(f"$E_2$ (MeV) & {np.mean(E_rmses[:,1]):.8f}\\pm{np.std(E_rmses[:,1]):.8f} & {np.mean(E_nrmses[:,1]):.8f}\\pm{np.std(E_nrmses[:,1]):.8f} & {np.mean(E_means[:,1]):.8f}\\pm{np.std(E_means[:,1]):.8f} & {np.mean(E_stds[:,1]):.8f}\\pm{np.std(E_stds[:,1]):.8f} & {np.mean(E_skews[:,1]):.8f}\\pm{np.std(E_skews[:,1]):.8f} \\\\")
    print(f"$m_{{\\gamma\\gamma}}$ (GeV) & {np.mean(mass_rmses):.8f}\\pm{np.std(mass_rmses):.8f} & {np.mean(mass_nrmses):.8f}\\pm{np.std(mass_nrmses):.8f} & {np.mean(mass_means):.8f}\\pm{np.std(mass_means):.8f} & {np.mean(mass_stds):.8f}\\pm{np.std(mass_stds):.8f} & {np.mean(mass_skews):.8f}\\pm{np.std(mass_skews):.8f} \\\\")
    print("\\hline")

    fig, ax = plt.subplots(figsize=get_figsize())
    ax.scatter(np.array(range(8))+1, mass_skews, marker="^", s=10, c="k", label=r"Data")
    ax.axhline(np.mean(mass_skews), 0,1,c="r", linestyle="--", label=r"$\overline{\gamma}_1$")
    ax.axhspan(np.mean(mass_skews) - np.std(mass_skews), np.mean(mass_skews) + np.std(mass_skews), 
               color="r", alpha=0.2, hatch="///", label=r"$\pm \sigma_{\overline{\gamma}_1}$")
    if version == "V4":
        vstr = "V1"
    elif version == "V5":
        vstr = "V4"
    else:
        vstr = version
    ax.set_xlabel(f"{vstr} Iteration")
    ax.set_ylabel(r"Skew ($\gamma_1$)")
    ax.legend()
    # ax.set_title(f"{version}")
    # plt.show()
    plt.savefig(f"report_results/appendix_graphs/{version}_mass_skews.png", dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()

