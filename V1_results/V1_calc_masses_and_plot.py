import argparse
import numpy as np
import matplotlib.pyplot as plt # for plotting
from matplotlib.ticker import MaxNLocator,AutoMinorLocator # for minor ticks
import matplotlib
from lmfit.models import PolynomialModel, GaussianModel # for the signal and background fits
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec
import scipy
import corner
from scipy.stats import chisquare, entropy

 
params = {'backend': 'Agg',
          'axes.labelsize': 10,
          "axes.titlesize": 10,
          'font.size': 10,
          "text.usetex": True,
          "font.family": "serif",
          } # extend as needed
matplotlib.rcParams.update(params)

BALER_PATH = "/Users/oscarfuentes/masters_project/baler_sem1/"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v",
                       type=str)
    args = parser.parse_args()
    return args

def load_data(args):
    with (np.load(BALER_PATH+"workspaces/higgs/data/V1_precut_normalised.npz") as orig_file,
          np.load(BALER_PATH+f"V1_results/V{args.v}/decompressed.npz") as decomp_file,
          np.load(BALER_PATH+"workspaces/higgs/data/V1_precut_scales.npz") as scale_file):
        orig_data = orig_file["data"]
        decomp_data = decomp_file["data"]
        names = orig_file["names"]
        scales = scale_file["data"]
    
    return orig_data, decomp_data, names, scales

def cut_photon_pt(photon_pt):
    return (photon_pt[:,0] > 40000) & (photon_pt[:,1] > 30000)

def cut_photon_eta_transition(photon_eta):
    return ((abs(photon_eta[:,0])>1.52) | (abs(photon_eta[:,0])<1.37)) & ((abs(photon_eta[:,1])>1.52) | (abs(photon_eta[:,1])<1.37))

def cut_tightID():
    with np.load("/Users/oscarfuentes/masters_project/baler_sem1/workspaces/higgs/data/V1_photon_isTightID.npz") as f:
        tightIDs = f["data"].astype(bool)
    return tightIDs[:,0] & tightIDs[:,1]

def cut_isolation_et(photon_etcone20):
    return (photon_etcone20[:,0]<4000) & (photon_etcone20[:,1]<4000)

def apply_cuts(precut_data):
    pt_cuts = cut_photon_pt(precut_data[:,:2])
    eta_cuts = cut_photon_eta_transition(precut_data[:,2:4])
    tight_IDs_cuts = cut_tightID()
    etcone_path = "/Users/oscarfuentes/masters_project/baler_sem1/workspaces/higgs/data/V4_photon_etcone20.npz"
    with np.load(etcone_path) as etcone_file:
        etcone_cuts = cut_isolation_et(etcone_file["data"])
    return pt_cuts & eta_cuts & tight_IDs_cuts & etcone_cuts

def unnormalise_E_pt(arr, scales):
    return 10**((arr*(scales[1] - scales[0])) + scales[0])

def unnormalise_eta_phi(arr, scales):
    return (arr*(scales[1] - scales[0])) + scales[0]

def prepare_data(args, decomp_cut=False):
    orig_norm, decomp_norm, _, scales = load_data(args)
    orig, decomp = [], []

    # print("Preparing data...")
    orig.append(np.column_stack([unnormalise_E_pt(arr, scale).T for arr, scale in zip(orig_norm[:,:2].T, scales[:,:2])]))
    orig.append(np.column_stack([unnormalise_eta_phi(arr, scale).T for arr, scale in zip(orig_norm[:,2:4].T, scales[:,2:4])]))
    orig.append(np.column_stack([unnormalise_eta_phi(arr, scale).T for arr, scale in zip(orig_norm[:,4:6].T, scales[:,4:6])]))
    orig.append(np.column_stack([unnormalise_E_pt(arr, scale).T for arr, scale in zip(orig_norm[:,6:8].T, scales[:,6:8])]))

    decomp.append(np.column_stack([unnormalise_E_pt(arr, scale) for arr, scale in zip(decomp_norm[:,:2].T, scales[:,:2])]))
    decomp.append(np.column_stack([unnormalise_eta_phi(arr, scale) for arr, scale in zip(decomp_norm[:,2:4].T, scales[:,2:4])]))
    decomp.append(np.column_stack([unnormalise_eta_phi(arr, scale) for arr, scale in zip(decomp_norm[:,4:6].T, scales[:,4:6])]))
    decomp.append(np.column_stack([unnormalise_E_pt(arr, scale) for arr, scale in zip(decomp_norm[:,6:8].T, scales[:,6:8])]))

    orig = np.column_stack(orig)
    decomp = np.column_stack(decomp)

    if decomp_cut:
        cuts = np.array(apply_cuts(decomp))
        orig_cuts = np.array(apply_cuts(orig))
        same_cut_fraction = sum(orig_cuts == cuts)/len(orig_cuts)
        print(f"Same cut: {same_cut_fraction*100:.2f}%")

    else:
        cuts = apply_cuts(orig)

    orig = orig[cuts]
    decomp = decomp[cuts]

    print(len(orig), len(decomp))

    return orig, decomp

def calc_masses(arr):
    # first photon is [0], 2nd photon is [1] etc
    px_0 = arr[:,0]*np.cos(arr[:,4]) # x-component of photon[0] momentum
    py_0 = arr[:,0]*np.sin(arr[:,4]) # y-component of photon[0] momentum
    pz_0 = arr[:,0]*np.sinh(arr[:,2]) # z-component of photon[0] momentum
    px_1 = arr[:,1]*np.cos(arr[:,5]) # x-component of photon[1] momentum
    py_1 = arr[:,1]*np.sin(arr[:,5]) # y-component of photon[1] momentum
    pz_1 = arr[:,1]*np.sinh(arr[:,3]) # z-component of photon[1] momentum
    sumpx = px_0 + px_1 # x-component of diphoton momentum
    sumpy = py_0 + py_1 # y-component of diphoton momentum
    sumpz = pz_0 + pz_1 # z-component of diphoton momentum 
    sump = np.sqrt(sumpx**2 + sumpy**2 + sumpz**2) # magnitude of diphoton momentum 
    sumE = arr[:,6] + arr[:,7] # energy of diphoton system
    return np.sqrt(abs(sumE**2 - sump**2))/1000 #/1000 to go from MeV to GeV

def calc_rmse(orig, decomp):
    return np.sqrt(np.mean((orig - decomp)**2))

def zvalue_poisson(D, B):
    """
    Returns the Poisson z-value for observed count D and expected background B.
    For an excess (D >= B): p-value = gammainc(D, B)
    For a deficit (D < B): p-value = 1 - gammainc(D+1, B)
    """
    if D >= B:
        p = scipy.special.gammainc(D, B)
        z = np.sqrt(2.0) * scipy.special.erfinv(1.0 - 2.0 * p)
    else:
        p = 1.0 - scipy.special.gammainc(D + 1, B)
        z = -np.sqrt(2.0) * scipy.special.erfinv(1.0 - 2.0 * p)
    return z

def plot_both_data(orig_data, decomp_data, cut_type, args):   

    xmin = 100 # GeV
    xmax = 160 # GeV
    step_size = 3 # GeV
    
    bin_edges = np.arange(start=xmin, # The interval includes this value
                     stop=xmax+step_size, # The interval doesn't include this value
                     step=step_size ) # Spacing between values
    bin_centres = np.arange(start=xmin+step_size/2, # The interval includes this value
                            stop=xmax+step_size/2, # The interval doesn't include this value
                            step=step_size ) # Spacing between values

    orig_data_x,_ = np.histogram(orig_data, 
                            bins=bin_edges) # histogram the data
    orig_data_x_errors = np.sqrt(orig_data_x) # statistical error on the data

    decomp_data_x, _ = np.histogram(decomp_data,
                                 bins=bin_edges)

    decomp_data_x_errors = np.sqrt(decomp_data_x)

    # data fit
    orig_polynomial_mod = PolynomialModel( 4 ) # 4th order polynomial
    orig_gaussian_mod = GaussianModel() # Gaussian

    # set initial guesses for the parameters of the polynomial model
    # c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4
    orig_pars = orig_polynomial_mod.guess(orig_data_x, # data to use to guess parameter values
                                x=bin_centres, c0=orig_data_x.max(), c1=0,
                                c2=0, c3=0, c4=0 )
    
    if int(args.v) != 6:
        decomp_polynomial_mod = PolynomialModel( 4 ) # 4th order polynomial
        decomp_gaussian_mod = GaussianModel() # Gaussian
        decomp_pars = decomp_polynomial_mod.guess(decomp_data_x, # data to use to guess parameter values
                            x=bin_centres, c0=orig_data_x.max(), c1=0,
                            c2=0, c3=0, c4=0)
    else:
        decomp_polynomial_mod = PolynomialModel( 5 ) # 5th order polynomial
        decomp_gaussian_mod = GaussianModel() # Gaussian
        
        decomp_pars = decomp_polynomial_mod.guess(decomp_data_x, # data to use to guess parameter values
                                    x=bin_centres, c0=orig_data_x.max(), c1=0,
                                    c2=0, c3=0, c4=0, c5=0)
    
    # set initial guesses for the parameters of the Gaussian model
    orig_pars += orig_gaussian_mod.guess(orig_data_x, # data to use to guess parameter values
                               x=bin_centres, amplitude=100, 
                               center=125, sigma=2 )
    
    decomp_pars += decomp_gaussian_mod.guess(decomp_data_x, # data to use to guess parameter values
                               x=bin_centres, amplitude=100, 
                               center=125, sigma=2 )
    
    orig_model = orig_polynomial_mod + orig_gaussian_mod # combined model
    decomp_model = decomp_polynomial_mod + decomp_gaussian_mod
    
    # fit the model to the data
    orig_out = orig_model.fit(orig_data_x, # data to be fit
                    orig_pars, # guesses for the parameters
                    x=bin_centres, weights=1/orig_data_x_errors ) 
    decomp_out = decomp_model.fit(decomp_data_x, # data to be fit
                    decomp_pars, # guesses for the parameters
                    x=bin_centres, weights=1/decomp_data_x_errors ) 

    # background part of fit
    orig_params_dict = orig_out.params.valuesdict() # get the parameters from the fit to data

    cs = ["c0", "c1", "c2", "c3", "c4"]
    
    # get the background only part of the fit to data
    orig_background = np.sum([orig_params_dict[c_val]*(bin_centres**i) for i, c_val in enumerate(cs)], axis=0)

    # data fit - background fit = signal fit
    orig_signal_x = orig_data_x - orig_background 

        # background part of fit
    decomp_params_dict = decomp_out.params.valuesdict() # get the parameters from the fit to data
    
    # get the background only part of the fit to data
    if int(args.v) == 6:
        cs += ["c5"]
    decomp_background = np.sum([decomp_params_dict[c_val]*(bin_centres**i) for i, c_val in enumerate(cs)], axis=0)

    # data fit - background fit = signal fit
    decomp_signal_x = decomp_data_x - decomp_background

    # Calculating z-values
    orig_fit = orig_out.best_fit
    decomp_fit = decomp_out.best_fit

    orig_peak_bin_idxs = np.where(abs(orig_fit - orig_background > 1))[0]
    orig_z = zvalue_poisson(sum(orig_fit[orig_peak_bin_idxs]), sum(orig_background[orig_peak_bin_idxs]))
    orig_big_z = zvalue_poisson(sum(orig_fit), sum(orig_background))
    # orig_z = sum(np.vectorize(zvalue_poisson)(orig_fit, orig_background))

    decomp_peak_bin_idxs = np.where(abs(decomp_fit - decomp_background) > 1)[0]
    decomp_z = zvalue_poisson(sum(decomp_fit[decomp_peak_bin_idxs]), sum(decomp_background[decomp_peak_bin_idxs]))
    decomp_big_z = zvalue_poisson(sum(decomp_fit), sum(decomp_background))
    print(f"orig_z: {orig_z:.3f}\norig_big_z: {orig_big_z:.3f}")
    print(f"\ndecomp_z: {decomp_z:.3f}\ndecomp_big_z: {decomp_big_z:.3f}")

    # Calculating Chisquare

    orig_bk_chi_val = orig_background * (sum(orig_data_x)/sum(orig_background))
    orig_fit_chi_val = orig_fit * (sum(orig_data_x)/sum(orig_fit))
    decomp_bk_chi_val = decomp_background * (sum(decomp_data_x)/sum(decomp_background))
    decomp_fit_chi_val = decomp_fit * (sum(decomp_data_x)/sum(decomp_fit))

    bk_dof = len(orig_data_x) - 5
    fit_dof = len(orig_data_x) - len(orig_params_dict.keys())

    orig_bk_chi = chisquare(orig_bk_chi_val, orig_data_x).statistic / bk_dof
    orig_fit_chi = chisquare(orig_fit_chi_val, orig_data_x).statistic / fit_dof
    decomp_bk_chi = chisquare(decomp_bk_chi_val, decomp_data_x).statistic / bk_dof
    decomp_fit_chi = chisquare(decomp_fit_chi_val, decomp_data_x).statistic / fit_dof

    # *************
    # Main plot 
    # *************
    plt.axes([0.13,0.44,0.85,0.52]) # left, bottom, width, height 
    main_axes = plt.gca() # get current axes
    
    #ORIGINAL
    # plot the original data points
    main_axes.errorbar(x=bin_centres, y=orig_data_x, yerr=orig_data_x_errors,
                       color="#8B0000",
                       marker='o', # 'k' means black and 'o' means circles
                       markersize=3,
                       label='Original Data',
                       zorder=1,
                       alpha=0.6,
                       linestyle="None") 
    
    # plot the signal + background fit
    main_axes.plot(bin_centres, # x
                   orig_fit, # y
                   '-r', # single red line
                   label=f'Original Sig+Bkg Fit ($m_H={orig_params_dict["center"]:.2f}$ GeV)',
                   linewidth=2,
                   alpha=0.6)
    
    # plot the background only fit
    main_axes.plot(bin_centres, # x
                   orig_background, # y
                   '--r', # dashed red line
                   label='Original Bkg (4th order polynomial)',
                   linewidth=1,
                   alpha=0.6)
    
    #DECOMPRESSED
    main_axes.errorbar(x=bin_centres, y=decomp_data_x, yerr=decomp_data_x_errors,
                       color="#00008B",
                       marker='o', # 'k' means black and 'o' means circles
                       markersize=3,
                       label='Reconstructed Data',
                       zorder=1,
                       alpha=0.6,
                       linestyle="None") 
    
    # plot the signal + background fit
    main_axes.plot(bin_centres, # x
                   decomp_fit, # y
                   '-b', # single red line
                   label=f'Reconstructed Sig+Bkg Fit ($m_H={decomp_params_dict["center"]:.2f}$ GeV)',
                   linewidth=2,
                   alpha=0.6)
    
    if int(args.v) == 6:
        order = 5
    else:
        order = 4
    # plot the background only fit
    main_axes.plot(bin_centres, # x
                   decomp_background, # y
                   '--b', # dashed red line
                   label=f'Reconstructed Bkg ({order}th order polynomial)',
                   linewidth=1,
                   alpha=0.6)
    

    # set the x-limit of the main axes
    main_axes.set_xlim( left=xmin, right=xmax ) 
    
    # separation of x-axis minor ticks
    main_axes.xaxis.set_minor_locator( AutoMinorLocator() ) 
    
    # set the axis tick parameters for the main axes
    main_axes.tick_params(which='both', # ticks on both x and y axes
                          direction='in', # Put ticks inside and outside the axes
                          top=True, # draw ticks on the top axis
                          labelbottom=False, # don't draw tick labels on bottom axis
                          right=True ) # draw ticks on right axis
    
    # write y-axis label for main axes
    main_axes.set_ylabel('Events / '+str(step_size)+' GeV',
                         horizontalalignment='right',
                         ) 
    
    # set the y-axis limit for the main axes
    main_axes.set_ylim( bottom=0, top=np.amax(orig_data_x)*1.1 ) 
    
    # set minor ticks on the y-axis of the main axes
    main_axes.yaxis.set_minor_locator( AutoMinorLocator() ) 
    
    # avoid displaying y=0 on the main axes
    main_axes.yaxis.get_major_ticks()[0].set_visible(False) 

    # Add text 'ATLAS Open Data' on plot
    plt.text(0.24, # x
             0.92, # y
             'ATLAS Open Data', # text
             transform=main_axes.transAxes, # coordinate system used is that of main_axes
             fontsize=13 ) 
    
    # Add text 'for education' on plot
    plt.text(0.24, # x
             0.86, # y
             'for education', # text
             transform=main_axes.transAxes, # coordinate system used is that of main_axes
             style='italic',
             fontsize=8 ) 
    
    # Add energy and luminosity
    lumi_used = str(10) # luminosity to write on the plot
    plt.text(0.24, # x
             0.78, # y
             r'$\sqrt{s}$=13 TeV,$\int$L dt = '+lumi_used+r' fb$^{-1}$', # text
             transform=main_axes.transAxes ) # coordinate system used is that of main_axes 
    
    # Add a label for the analysis carried out
    plt.text(0.24, # x
             0.71, # y
             r'$H \rightarrow \gamma\gamma$', # text 
             transform=main_axes.transAxes ) # coordinate system used is that of main_axes
    
    # Add the calculated significances
    orig_z_str = r"$\sigma_\mathrm{original} =$"+f"{orig_z:.3f}\n"
    decomp_z_str = r"$\sigma_\mathrm{reconstructed} =$"+f"{decomp_z:.3f}"
    chi_bk_str = "\n\n"+rf"${{\chi_\nu^2}}_\mathrm{{Bkg}}={decomp_bk_chi:.3f}$"
    chi_fit_str = "\n"+rf"${{\chi_\nu^2}}_\mathrm{{Sig}}={decomp_fit_chi:.3f}$"
    kl_str = rf"$D_{{\mathrm{{KL}}}}(\mathrm{{Orig.}} \,\|\, \mathrm{{Rec.}})={entropy(orig_fit, decomp_fit):.2e}$"
    if np.isnan(decomp_z):
        decomp_z_str = r"$\sigma_\mathrm{reconstructed} =$"+"N/A"
    plt.text(0.65,
             0.44,
             orig_z_str+decomp_z_str+chi_bk_str+chi_fit_str,
             transform=main_axes.transAxes,
             fontsize=17)

    print(f"\n\n\n{args.v}: {entropy(orig_fit, decomp_fit)}\n\n\n")
    main_axes.legend()
    order = [4, 5, 1, 0, 3, 2]
    
    handles, labels = main_axes.get_legend_handles_labels()
    main_axes.legend([handles[idx] for idx in order],
                     [labels[idx] for idx in order],
                     frameon=False,
                     loc="lower left",
                     )


    # *************
    # Data-Bkg plot 
    # *************
    plt.axes([0.13,0.27,0.85,0.17]) # left, bottom, width, height
    fit_axes = plt.gca() # get the current axes
    
    # set the y axis to be symmetric about Data-Background=0
    fit_axes.yaxis.set_major_locator( MaxNLocator(nbins='auto', 
                                                  symmetric=True) )
    
    # plot Data-Background
    fit_axes.errorbar(x=bin_centres, y=orig_signal_x, yerr=orig_data_x_errors,
                      color="#8B0000",
                      marker="o",
                      markersize=3,
                      alpha=0.6,
                      linestyle="None") # 'k' means black and 'o' means circles
    
    # draw the fit to data
    fit_axes.plot(bin_centres, # x
                  orig_out.best_fit-orig_background, # y
                  '-r',
                  linewidth=2,
                  alpha=0.6) # single red line
    
    fit_axes.errorbar(x=bin_centres, y=decomp_signal_x, yerr=decomp_data_x_errors,
                      color="#00008B",
                      marker="o",
                      markersize=3,
                      alpha=0.6,
                      linestyle="None") # 'k' means black and 'o' means circles
    
    # draw the fit to data
    fit_axes.plot(bin_centres, # x
                  decomp_out.best_fit-decomp_background, # y
                  '-b',
                  linewidth=2,
                  alpha=0.6) # single red line
    
    # draw the background only fit
    fit_axes.plot(bin_centres, # x
                  orig_background-orig_background, # y
                  '--k',
                  alpha=0.6)  # dashed black line
    
    # set the x-axis limits on the sub axes
    fit_axes.set_xlim( left=xmin, right=xmax ) 
    
    # separation of x-axis minor ticks
    fit_axes.xaxis.set_minor_locator( AutoMinorLocator() ) 
    
    # # x-axis label
    # fit_axes.set_xlabel(r'di-photon invariant mass $\mathrm{m_{\gamma\gamma}}$ [GeV]',
    #                     x=1, horizontalalignment='right', 
    #                     fontsize=13 ) 
    
    # set the tick parameters for the sub axes
    fit_axes.tick_params(which='both', # ticks on both x and y axes
                         direction='in', # Put ticks inside and outside the axes
                         top=True, # draw ticks on the top axis
                          labelbottom=False, # don't draw tick labels on bottom axis
                         right=True ) # draw ticks on right axis 
    
    # separation of y-axis minor ticks
    fit_axes.yaxis.set_minor_locator( AutoMinorLocator() ) 
    
    # y-axis label on the sub axes
    fit_axes.set_ylabel( 'Events-Bkg') 

    ###
    # Difference between original and reconstructed
    ###
    plt.axes([0.13,0.1,0.85,0.17]) # left, bottom, width, height
    diff_axes = plt.gca() # get the current axes
    
    # set the y axis to be symmetric about Data-Background=0
    diff_axes.yaxis.set_major_locator( MaxNLocator(nbins='auto', 
                                                  symmetric=True) )
    
    # plot Data-Background
    diff_errs = np.sqrt(orig_data_x_errors**2 + decomp_data_x_errors**2)*0.6
    diff_axes.errorbar(x=bin_centres, y=decomp_signal_x - orig_signal_x, yerr=diff_errs,
                      color="#00008B",
                      marker="o",
                      markersize=3,
                      alpha=0.6,
                      linestyle="None") # 'k' means black and 'o' means circles
    
    # draw the background only fit
    diff_axes.plot(bin_centres, # x
                  orig_background-orig_background, # y
                  linestyle='--',
                  color="#8B0000",
                  alpha=0.6)  # dashed black line
    
    # set the x-axis limits on the sub axes
    diff_axes.set_xlim( left=xmin, right=xmax ) 
    
    # separation of x-axis minor ticks
    diff_axes.xaxis.set_minor_locator( AutoMinorLocator() ) 
    
    # x-axis label
    diff_axes.set_xlabel(r'di-photon invariant mass $\mathrm{m_{\gamma\gamma}}$ [GeV]',
                        x=1, horizontalalignment='right', 
                        fontsize=13 ) 
    
    # set the tick parameters for the sub axes
    diff_axes.tick_params(which='both', # ticks on both x and y axes
                         direction='in', # Put ticks inside and outside the axes
                         top=True, # draw ticks on the top axis
                         right=True ) # draw ticks on right axis 
    
    diff_ylims = diff_axes.get_yticks()[-2]
    diff_ytick_locs = [int(-diff_ylims*0.9), 0, int(diff_ylims*0.9)]
    diff_ytick_labels = [str(x) for x in diff_ytick_locs]

    diff_axes.set(yticks=diff_ytick_locs,
                  yticklabels=diff_ytick_labels)
    
    # separation of y-axis minor ticks
    diff_axes.yaxis.set_minor_locator( AutoMinorLocator() ) 
    
    # y-axis label on the sub axes
    diff_axes.set_ylabel("Rec. - Orig.\n  (Events)") 

    if np.mean(decomp_out.best_fit - decomp_background) > 10000:
        main_fit_x = -0.105
    else:
        main_fit_x = -0.085

    # Generic features for both plots
    main_axes.yaxis.set_label_coords( main_fit_x, 1 ) # x,y coordinates of the y-axis label on the main axes
    fit_axes.yaxis.set_label_coords( main_fit_x, 0.5 ) # x,y coordinates of the y-axis label on the sub axes
    diff_axes.yaxis.set_label_coords( main_fit_x + 0.03, 0.5 ) # x,y coordinates of the y-axis label on the sub axes
    
    plt.savefig(f"{cut_type}HyyAnalysis_comparison.png")
    # plt.show()
    plt.close()

    return bin_centres, (orig_out, orig_background, orig_params_dict, orig_model), (decomp_out, decomp_background, decomp_params_dict, decomp_model)

def plot_resolutions(
    original,
    reconstructed,
    variable,
    bins=100,
    hist_yscale="linear",
    save=False,
    savename="",
    norm=False,
    show=False,
):
    """
    Function which plots three types of comparisons when reconstructing Hyy data with Baler

    Parameters:
        original: 2xN numpy array, for N events after cuts for photon 1 and photon 2
        reconstructed: 2xN numpy array, which is the decompressed version of original
        variable: What variable you want plotted, options are "pt", "eta", "phi", "E"
        bins: Number of bins in the first histogram
        hist_yscale: Whether you want the first histogram to have a "log" y-axis, default "linear"
        save: bool whether to save the image created or not
        savename: If saving, what you want to name the file (don't include .png at the end)
        norm: Whether E and pt are normalised to [0,1] and eta and phi normalised to [-1,1]

    Practice use:
        plot_resolutions(orig_Es, new_Es, "E", bins=200, hist_yscale="log", save=True, savename="E_resolution_plots")

        Will output the three plots, the first histogram of the data comparison will be on a log scale, and the
        image will be saved to E_resolution_plots.png

    """

    if variable[-1].isnumeric():
        name = variable[:-1]
    else:
        name = variable

    # Initialise figure and axes objects
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 7))
    ax1, ax2, ax3, ax4 = axs.flatten()

    # Calculate the original and reconstructed histograms of distributions
    # print("Calculating original histogram...")
    orig_n, orig_bins = np.histogram(original, bins=bins)
    # print("Calculating reconstructed histogram...")
    rec_n, rec_bins = np.histogram(reconstructed, bins=orig_bins)
    # print("Histograms complete\n")

    # Calculate the resolution histogram
    diffs = original - reconstructed
    diff_n, diff_bins = np.histogram(diffs, bins=100)

    # Calculate the resolution histogram
    if np.any(abs(original) < 0.0001):
        hist_ratios = (original - reconstructed) / (original + np.amax(abs(original)) * 1.1)
    else:
        hist_ratios = (original - reconstructed) / original

    ratio_n, ratio_bins = np.histogram(hist_ratios, bins=100)

    # Plot results on all the axes
    # print("Plotting...")
    ax1.stairs(orig_n, orig_bins, label="Original")
    ax1.stairs(rec_n, rec_bins, label="Reconstructed")

    ax2.stairs(diff_n, diff_bins)

    ax3.scatter(np.hstack(original), np.hstack(reconstructed), label="Data", s=1)

    bounds = np.array([np.percentile(hist_ratios, 1), np.percentile(hist_ratios, 99)])
    resolution_xlims = [-np.amax(abs(bounds)), np.amax(abs(bounds))]
    cut_hist_ratios  = [x for x in hist_ratios if x > resolution_xlims[0] and x < resolution_xlims[1]]
    ratio_n, ratio_bins = np.histogram(cut_hist_ratios, bins=100)

    ax4.stairs(ratio_n, ratio_bins)
    # print("Plotting complete...\n")

    # Define the comparison x-axis and units for each variable
    if norm:
        xlims = (-0.1, 1.1)
        unit = ""
    elif name in ["E", "pt"] and hist_yscale == "log":
        xlims = [0, np.amax(original)]
        unit = "(MeV)"
    elif name in ["E", "pt"]:
        xlims = [0, np.amax(original) * 0.2]
        unit = "(MeV)"
    elif name == "phi":
        maxi = np.amax((abs(original), abs(reconstructed))) * 1.1
        xlims = [-maxi, maxi]
        unit = "(rad)"
    elif name == "eta":
        maxi = np.amax((abs(original), abs(reconstructed))) * 1.1
        xlims = [-maxi, maxi]
        unit = ""
    else:
        xlims = ax1.get_xlim()
        unit = ""
    
    if name == "eta":
        xlabel = r"$\eta$"
    elif name == "phi":
        xlabel = r"$\phi$"
    else:
        xlabel=name

    if name == "mass":
        unit = "(GeV)"

    # Set options for reconstruction comparison plot
    ax1.set(
        xlabel=f"{xlabel} {unit}",
        ylabel="Events",
        yscale=hist_yscale,
        xlim=xlims,
        title="Reconstruction comparison",
    )

    # Set options for difference plot
    max_err = np.amax(abs(diffs))
    ax2.set(
        xlabel=f"Original - Reconstructed {xlabel} {unit}",
        ylabel="Events",
        yscale="log",
        xlim=[-max_err * 1.1, max_err * 1.1],
        title="Difference plot",
    )
    # if name == "pt":
    #     xticks = ax2.get_xticks()[::2].astype(int)
    #     ax2.set(xticks=xticks, xticklabels=xticks)
    rmse = np.sqrt(np.mean(diffs**2))

    ax2.text(
        ax2.get_xlim()[0] + (abs(ax2.get_xlim()[0])) * 0.03,
        np.amax(diff_n) * 0.035,
        f"Max difference:\n{max_err:.2e} {unit}\n\nRMSE:\n{rmse:.2e} {unit}",
        fontsize=8,
    )

    # Set options for event-by-event comparison plot
    if name == "pt" or name == "E" or name == "mass":
        xlims = [0, np.amax((original, reconstructed)) * 1.1]
    else:
        xlims = [np.amin((original, reconstructed)) * 1.1, np.amax((original, reconstructed)) * 1.1]

    ax3.plot(xlims, xlims, color="r", linewidth=0.5, label="No difference")
    ax3.set(
        xlabel=f"Original {xlabel} {unit}",
        ylabel=f"Reconstructed {xlabel} {unit}",
        title="Event-by-event comparison",
        xlim=xlims,
        ylim=xlims,
    )

    # Set options for resolution plot
    iqr = np.percentile(hist_ratios, 75) - np.percentile(hist_ratios, 25)

    xlims = resolution_xlims
    xlabel = r"$\frac{(\mathrm{original}-\mathrm{reconstructed})}{\mathrm{original}}$"
    if name == "eta":
        xlabel = r"$\frac{(\mathrm{original}-\mathrm{reconstructed})}{(\mathrm{original}+3)}$"
    elif name == "phi":
        xlabel = r"$\frac{(\mathrm{original}-\mathrm{reconstructed})}{(\mathrm{original}+2\pi)}$"

    
    ax4.text(
        xlims[0] + (abs(xlims[0])) * 0.03,
        np.amax(ratio_n)*0.7,
        f"Max: {np.amax(hist_ratios):.3f}\nMin: {np.amin(hist_ratios):.3f}"
        + f"\nMean: {np.mean(hist_ratios):.3f}\nMedian: {np.median(hist_ratios):.3f}"
        + f"\nIQR: {iqr:.3f}",
        fontsize=8,
    )

    ax4.set(
        ylabel="Events",
        title="Resolution plot",
        xlim=xlims,
        # yscale=hist_yscale,
    )
    ax4.set_xlabel(xlabel, fontsize=14)

    # Include legends where necessary
    ax1.legend()
    ax3.legend(loc="upper left")
    
    # Sort out a bug in difference/resolution xlims
    # ax2.xaxis.set_major_locator(MaxNLocator(nbins=5))
    # ax4.xaxis.set_major_locator(MaxNLocator(nbins=5))
    # for tick in ax4.get_xticklabels():
    #     tick.set_rotation(15)

    # Give a title and tight_layout the figure
    fig.suptitle(f"{variable} plots")
    fig.tight_layout()

    # Sort out a bug in difference/resolution xlims
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax4.xaxis.set_major_locator(MaxNLocator(nbins=3))

    # Sort out a bug in the normalised xlims
    if norm:
        for ax in [ax1, ax3]:
            xlim_gap = ax.get_xlim()[1] - 1
            ax.set_xlim((-xlim_gap, 1+xlim_gap))

    # # Sort out a bug for the pt x-axis
    # if name == "pt":
    #     xticks = ax1.get_xticks()[::2].astype(int)
    #     ax1.set(xticks=xticks, xticklabels=xticks)

    # Save if you want
    if save or savename != "":
        plt.savefig(f"{savename}.png")

    if show:
        plt.show()
    else:
        plt.close()

def corner_plot(args, orig_data, decomp_data, orig_masses, decomp_masses, plot_all=False):

    pt_resolutions = (orig_data[:,0:2] - decomp_data[:,0:2])/(orig_data[:,0:2])
    eta_resolutions = (orig_data[:,2:4] - decomp_data[:,2:4])/(orig_data[:,2:4] + 10)
    phi_resolutions = (orig_data[:,4:6] - decomp_data[:,4:6])/(orig_data[:,4:6] + 5*np.pi)
    E_resolutions = (orig_data[:,6:8] - decomp_data[:,6:8])/(orig_data[:,6:8])
    if np.any(orig_masses < 0.001):
        mass_resolutions = (orig_masses - decomp_masses)/(orig_masses + np.amin(orig_masses) + 1)
    else:
        mass_resolutions = (orig_masses - decomp_masses)/(orig_masses)

    all_resolutions = [pt_resolutions[:,0],
                       eta_resolutions[:,0],
                       phi_resolutions[:,0],
                       E_resolutions[:,0],
                       pt_resolutions[:,1],
                       eta_resolutions[:,1],
                       phi_resolutions[:,1],
                       E_resolutions[:,1],
                       mass_resolutions]

    res_xlims = []
    for res in all_resolutions:
        bounds = np.array([np.percentile(res, 0.5), np.percentile(res, 99.5)])
        xlims = [-np.amax(abs(bounds)), np.amax(abs(bounds))]
        res_xlims.append(xlims)
        # cut_resolutions.append(res[(res > np.percentile(res, 1)) & (res < np.percentile(res, 99))])

    # cut_resolutions = np.column_stack(cut_resolutions)
    all_resolutions = np.column_stack(all_resolutions)

    if plot_all:
        fig = corner.corner(all_resolutions, bins=150)
        axes = np.array(fig.axes).reshape((9, 9))

        for i, label in enumerate(["leading pt", "leading E", "leading eta", "leading phi", "secondary pt", "secondary E", "secondary eta", "secondary phi", "mass"]):
            axes[i,0].set_ylabel(label, fontsize=10)
            axes[8,i].set_xlabel(label, fontsize=10)


        for i, xlim in enumerate(res_xlims):
            axes[i, i].set(xlim=xlim)

        for yi, xlim in enumerate(res_xlims):
            y = yi+1
            if y == 9:
                break
            for xi in range(y):
                axes[y, xi].set(xlim=res_xlims[xi], ylim=res_xlims[y])

        axes[0,6].text(0, 0, r"$E_{\mathrm{resolution}}\mathrm{,  }pt_{\mathrm{resolution}} = \frac{(\mathrm{original}-\mathrm{reconstructed})}{\mathrm{original}}$"+
                    "\n"+r"$\eta_{\mathrm{resolution}}=\frac{(\mathrm{original}-\mathrm{reconstructed})}{\mathrm{original}+3}$"+"\n"+
                    r"$\phi_{\mathrm{resolution}}=\frac{(\mathrm{original}-\mathrm{reconstructed})}{\mathrm{original}+2\pi}$", fontsize=12)


        fig.suptitle("Both photons", fontsize=14)
        fig.set_size_inches(fig.get_size_inches()*0.5)
        # fig.tight_layout()
        # plt.show()
        plt.close()

    fig2 = corner.corner(np.column_stack([all_resolutions[:,0], all_resolutions[:,3], all_resolutions[:,4], all_resolutions[:,7]]), bins=300)
    axes = np.array(fig2.axes).reshape((4, 4))

    res_xlims = [res_xlims[0], res_xlims[3], res_xlims[4], res_xlims[7]]

    for i, label in enumerate(["leading pt", "leading E", "secondary pt", "secondary E"]):
        axes[i,0].set_ylabel(label, fontsize=10)
        axes[3,i].set_xlabel(label, fontsize=10)


    for i, xlim in enumerate(res_xlims):
        axes[i, i].set(xlim=xlim)

    for yi, xlim in enumerate(res_xlims):
        y = yi+1
        if y == 4:
            break
        for xi in range(y):
            axes[y, xi].set(xlim=res_xlims[xi], ylim=res_xlims[y])
    
    axes[0,2].text(0.5,
                   0.5,
                   r"$\mathrm{Resolution}=\frac{\mathrm{original}-\mathrm{reconstructed}}{\mathrm{original}}$",
                   fontsize=14)
    
    fig2.suptitle("Energy and Transverse Momentum Resolution Relationship")
    fig2.savefig(f"V4_results/V{args.v}/Ept_corner_plot.png")
    # plt.show()
    plt.close()


def main():
    args = parse_args()

    print("\nPreparing data...")
    orig_data, decomp_data = prepare_data(args, decomp_cut=False)

    print("Calculating mass and rmse...")
    orig_masses = calc_masses(orig_data)
    decomp_masses = calc_masses(decomp_data)
    mass_rmse = calc_rmse(orig_masses, decomp_masses)

    print("Plotting Higgs analysis...")
    plot_both_data(orig_masses, decomp_masses, f"V4_results/V{args.v}/", args)

    # print("Plotting resolutions...")
    # plot_resolutions(orig_masses, decomp_masses, "mass", hist_yscale="log", savename=f"V4_results/V{args.v}/mass_resolution")

    # print("Plotting corner plot...")
    # corner_plot(args, orig_data, decomp_data, orig_masses, decomp_masses)

    print(f"\nMass RMSE: {mass_rmse:.2f}")

if __name__ == "__main__":
    main()
