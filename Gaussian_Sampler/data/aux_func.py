'''
This module contains several auxiliary functions used in the "Polymers Data Analysis - Salleo's OMIECs" notebooks, in addition to py4DSTEM's functions.

Dr. Yael Tsarfati
'''

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from matplotlib.colors import hsv_to_rgb
from matplotlib.colors import rgb_to_hsv
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages  
from emdfile import tqdmnd, PointList, PointListArray
from scipy.signal import medfilt2d
import py4DSTEM
import numpy as np
import fitz  
from pdf2image import convert_from_path
from PIL import Image


'''
Remove hot pixels from the 4DSTEM data
'''
def remove_hot_pixels(
    dataset,
    dp_mean_data,
    relative_threshold = 0.8,
    returnfig_hot_pixels = False
):
    dataset.get_dp_mean()
    mask_hot_pixels =  dp_mean_data - medfilt2d(dp_mean_data) > relative_threshold * dp_mean_data
    print('Total hot pixels = ' + str(mask_hot_pixels.sum())) 
    if returnfig_hot_pixels:
        py4DSTEM.show(
            mask_hot_pixels,
            figsize=(6,6),
        )
    # Apply mask - this step is not reversible!
    dataset.data *= (1 - mask_hot_pixels[None,None,:,:].astype('uint8'))
    dataset.get_dp_mean();
    dataset.get_dp_max();
    return dataset, mask_hot_pixels

'''
Filter a given set of Bragg peaks based on a q-range and an intensity threshold.
'''
def filter_bragg_peaks(
    bragg_peaks,
    R_Nx,
    R_Ny,
    q_range,
    inv_A_per_pixel_CL_corrected,threshold,
    boolean_peaks
):
    # Go over real space positions
    for rx in range(R_Nx):
        for ry in range(R_Ny):
            # Check point list per real space position : 
            # pl as the Bragg peaks at position rx,ry:
            pl = bragg_peaks.vectors.get_pointlist(rx,ry)
            if pl.length > 0:   # otherwise will stay False
                # Array of False with length of number of peaks
                # The peaks in a given positions
                boolean_braggpeak_rx_ry = np.zeros(pl.length, dtype=bool)
                for index in range(pl.length):     
                    # peak by peak 
                    # 1) check Q range
                    q = np.hypot(pl['qx'][index],pl['qy'][index])
                    # True if you are a backbone peak - by Q range
                    boolean_braggpeak_rx_ry[index]=~np.logical_or( q<q_range[0]*inv_A_per_pixel_CL_corrected,q>=q_range[1]*inv_A_per_pixel_CL_corrected)     
                    # Now, check the intensity threshold
                    #2)check intensity threshold
                    if  boolean_braggpeak_rx_ry[index]:
                        if pl['intensity'][index]<threshold:
                            boolean_braggpeak_rx_ry[index] = False
                        # right range and  pass Intensity threshold 
                        else :
                            # True if you are a backbone peak. An array of Yes/No per position (vs point list per position) 
                            boolean_peaks[rx,ry] = True        
            pl.remove(~boolean_braggpeak_rx_ry)
    return bragg_peaks,boolean_peaks

'''
Calibrate Bragg peaks based on the user's input parameters and pixel size derived from the calibrant.
'''
def calibrate_bragg_peaks(bragg_peaks,step_size,inv_A_per_pixel_CL_corrected,rotation_calibration):
    bragg_peaks.calibration.set_R_pixel_size(step_size)
    bragg_peaks.calibration.set_R_pixel_units('nm')
    bragg_peaks.calibration.set_Q_pixel_size(inv_A_per_pixel_CL_corrected )
    bragg_peaks.calibration.set_Q_pixel_units('A^-1')
    bragg_peaks.calibration.set_QR_rotation_degrees('rotation_calibration')
    bragg_peaks.calibrate()
    return bragg_peaks

'''
Center Bargg peaks.
'''
def centering_bragg_peaks(bragg_peaks):
    # Compute the origin position pattern-by-pattern
    origin_meas = bragg_peaks.measure_origin(mode = 'no_beamstop',)
    qx0_fit,qy0_fit,qx0_residuals,qy0_residuals = bragg_peaks.fit_origin(
        fitfunction='bezier_two',
        plot_range = 2.0,
    )
    bragg_peaks.calibration.set_origin((qx0_fit, qy0_fit))
    bragg_peaks.calibrate()
    return bragg_peaks
        
'''
Copied from py4DSTEM version 13.17 to be used in the plot_orientation_correlation_vis function, which was modified locally for visualization purposes.
'''
def orientation_correlation(
    orient_hist,
    radius_max=None,
    progress_bar=True,
):
    """
    Take in the 4D orientation histogram, and compute the distance-angle (auto)correlations

    Args:
        orient_hist (array):    3D or 4D histogram of all orientations with coordinates [x y radial_bin theta]
        radius_max (float):     Maximum radial distance for correlogram calculation. If set to None, the maximum
                                radius will be set to min(orient_hist.shape[0],orient_hist.shape[1])/2.

    Returns:
        orient_corr (array):          3D or 4D array containing correlation images as function of (dr,dtheta)
    """

    # Array sizes
    size_input = np.array(orient_hist.shape)
    if radius_max is None:
        radius_max = np.ceil(np.min(orient_hist.shape[1:3]) / 2).astype("int")
    size_corr = np.array(
        [
            np.maximum(2 * size_input[1], 2 * radius_max),
            np.maximum(2 * size_input[2], 2 * radius_max),
        ]
    )

    # Initialize orientation histogram
    orient_hist_pad = np.zeros(
        (
            size_input[0],
            size_corr[0],
            size_corr[1],
            size_input[3],
        ),
        dtype="complex",
    )
    orient_norm_pad = np.zeros(
        (
            size_input[0],
            size_corr[0],
            size_corr[1],
        ),
        dtype="complex",
    )

    # Pad the histogram in real space
    x_inds = np.arange(size_input[1])
    y_inds = np.arange(size_input[2])
    orient_hist_pad[:, x_inds[:, None], y_inds[None, :], :] = orient_hist
    orient_norm_pad[:, x_inds[:, None], y_inds[None, :]] = np.sum(
        orient_hist, axis=3
    ) / np.sqrt(size_input[3])
    orient_hist_pad = np.fft.fftn(orient_hist_pad, axes=(1, 2, 3))
    orient_norm_pad = np.fft.fftn(orient_norm_pad, axes=(1, 2))

    # Radial coordinates for integration
    x = (
        np.mod(np.arange(size_corr[0]) + size_corr[0] // 2, size_corr[0])
        - size_corr[0] // 2
    )
    y = (
        np.mod(np.arange(size_corr[1]) + size_corr[1] // 2, size_corr[1])
        - size_corr[1] // 2
    )
    ya, xa = np.meshgrid(y, x)
    ra = np.sqrt(xa**2 + ya**2)

    # coordinate subset
    sub0 = ra <= radius_max
    sub1 = ra <= radius_max - 1
    rF0 = np.floor(ra[sub0]).astype("int")
    rF1 = np.floor(ra[sub1]).astype("int")
    dr0 = ra[sub0] - rF0
    dr1 = ra[sub1] - rF1
    inds = np.concatenate((rF0, rF1 + 1))
    weights = np.concatenate((1 - dr0, dr1))

    # init output
    num_corr = (0.5 * size_input[0] * (size_input[0] + 1)).astype("int")
    orient_corr = np.zeros(
        (
            num_corr,
            (size_input[3] // 2 + 1).astype("int"),
            radius_max + 1,
        )
    )

    # Main correlation calculation
    ind_output = 0
    for a0, a1 in tqdmnd(
        range(size_input[0]),
        range(size_input[0]),
        desc="Calculate correlation plots",
        unit=" probe positions",
        disable=not progress_bar,
    ):
        # for a0 in range(size_input[0]):
        #     for a1 in range(size_input[0]):
        if a0 <= a1:
            # Correlation
            c = np.real(
                np.fft.ifftn(
                    orient_hist_pad[a0, :, :, :]
                    * np.conj(orient_hist_pad[a1, :, :, :]),
                    axes=(0, 1, 2),
                )
            )

            # Loop over all angles from 0 to pi/2  (half of indices)
            for a2 in range((size_input[3] / 2 + 1).astype("int")):
                orient_corr[ind_output, a2, :] = np.bincount(
                    inds,
                    weights=weights
                    * np.concatenate((c[:, :, a2][sub0], c[:, :, a2][sub1])),
                    minlength=radius_max,
                )

            # normalize
            c_norm = np.real(
                np.fft.ifftn(
                    orient_norm_pad[a0, :, :] * np.conj(orient_norm_pad[a1, :, :]),
                    axes=(0, 1),
                )
            )
            sig_norm = np.bincount(
                inds,
                weights=weights * np.concatenate((c_norm[sub0], c_norm[sub1])),
                minlength=radius_max,
            )
            orient_corr[ind_output, :, :] /= sig_norm[None, :]

            # increment output index
            ind_output += 1

    return orient_corr

# The modified version of py4DSTEM's plot_orientation_correlation from version 13.17 for visualization purposes.
def plot_orientation_correlation_vis(
    orient_corr,
    prob_range = [0.1, 10.0],
    calculate_coefs = False,
    fraction_coefs = 0.5,
    length_fit_slope = 10,
    plot_overlaid_coefs = True,
    inds_plot = None,  # You can now pass specific indices here
    pixel_size = None,
    pixel_units = None,
    fontsize = 24,
    figsize = (8, 6),
    returnfig = False,
    title = True,
    intensity_bar = False,
    save_to_pdf = False,  # New parameter to save the selected plot(s) to PDF
    pdf_filename = "output.pdf",  # Default PDF filename
    rotation_legend_num = 90,
    legend_font = 20,
    y_label = True,
):
    """
    Plot the distance-angle (auto)correlations in orient_corr.

    Parameters
    ----------
    orient_corr (array):
        3D or 4D array containing correlation images as function of (dr,dtheta)
        1st index represents each pair of rings.
    prob_range (array):
        Plotting range in units of "multiples of random distribution".
    calculate_coefs (bool):
        If this value is True, the 0.5 and 0.1 distribution fraction of the
        radial and annular correlations will be calculated and printed.
    fraction_coefs (float):
        What fraction to calculate the correlation distribution coefficients for.
    length_fit_slope (int):
        Number of pixels to fit the slope of angular vs radial intercept.
    plot_overlaid_coefs (bool):
        If this value is True, the 0.5 and 0.1 distribution fraction of the
        radial and annular correlations will be overlaid onto the plots.
    inds_plot (float):
        Which indices to plot for orient_corr.  Set to "None" to plot all pairs.
    pixel_size (float):
        Pixel size for x axis.
    pixel_units (str):
        units of pixels.
    fontsize (float):
        Font size.  Title will be slightly larger, axis slightly smaller.
    figsize (array):
        Size of the figure panels.
    returnfig (bool):
        Set to True to return figure axes.
    save_to_pdf (bool) : parameter to set the selected plot(s) to be saved to the file.
    pdf_filename="output.pdf" (syring) :  # PDF filename

    Returns
    --------
    fig, ax (handles):
        Figure and axes handles (optional).

    """
    prob_range = np.array(prob_range)

    if pixel_size is None:
        pixel_size = 1
    if pixel_units is None:
        pixel_units = "pixels"

    # Get the pair indices
    size_input = orient_corr.shape
    num_corr = (np.sqrt(8 * size_input[0] + 1) // 2 - 1 // 2).astype("int")
    ya, xa = np.meshgrid(np.arange(num_corr), np.arange(num_corr))
    keep = ya >= xa
    pair_inds = np.vstack((xa[keep], ya[keep]))

    if inds_plot is None:
        inds_plot = np.arange(size_input[0])
    elif np.ndim(inds_plot) == 0:
        inds_plot = np.atleast_1d(inds_plot)
    else:
        inds_plot = np.array(inds_plot)

    # Custom colormap
    N = 256
    cvals = np.zeros((N, 4))
    cvals[:, 3] = 1
    c = np.linspace(0.0, 1.0, int(N / 4))

    cvals[0 : int(N / 4), 1] = c * 0.4 + 0.3
    cvals[0 : int(N / 4), 2] = 1

    cvals[int(N / 4) : int(N / 2), 0] = c
    cvals[int(N / 4) : int(N / 2), 1] = c * 0.3 + 0.7
    cvals[int(N / 4) : int(N / 2), 2] = 1

    cvals[int(N / 2) : int(N * 3 / 4), 0] = 1
    cvals[int(N / 2) : int(N * 3 / 4), 1] = 1 - c
    cvals[int(N / 2) : int(N * 3 / 4), 2] = 1 - c

    cvals[int(N * 3 / 4) : N, 0] = 1 - 0.5 * c
    new_cmap = ListedColormap(cvals)

    if calculate_coefs:
        # Perform fitting
        def fit_dist(x, *coefs):
            int0 = coefs[0]
            int_bg = coefs[1]
            sigma = coefs[2]
            p = coefs[3]
            return (int0 - int_bg) * np.exp(np.abs(x) ** p / (-1 * sigma**p)) + int_bg

    # plotting
    num_plot = inds_plot.shape[0]
    fig, ax = plt.subplots(num_plot, 1, figsize=(figsize[0], num_plot * figsize[1]))

    # loop over indices
    for count, ind in enumerate(inds_plot):
        if num_plot > 1:
            p = ax[count].imshow(
                np.log10(orient_corr[ind, :, :]),
                vmin=np.log10(prob_range[0]),
                vmax=np.log10(prob_range[1]),
                aspect="auto",
                cmap=new_cmap,
            )
            ax_handle = ax[count]
        else:
            p = ax.imshow(
                np.log10(orient_corr[ind, :, :]),
                vmin=np.log10(prob_range[0]),
                vmax=np.log10(prob_range[1]),
                aspect="auto",
                cmap=new_cmap,
            )
            ax_handle = ax
            
        if intensity_bar:
            cbar = fig.colorbar(p, ax=ax_handle)
            t = cbar.get_ticks()
            t_lab = []
            t_lab = [f"{10**tick:.1f}" for tick in t]  # Round to one decimal place

            cbar.set_ticks(t)
            cbar.ax.set_yticklabels(t_lab, fontsize=fontsize * 1.)
            if y_label:
                cbar.ax.set_ylabel("Probability (m.r.d.)", fontsize=legend_font)
                # cbar.ax.yaxis.set_label_coords(-0.1, 0.5)  # Move the label to the right


             # Rotate colorbar tick labels by 90 degrees
            cbar.ax.tick_params(labelsize=legend_font, labelrotation=rotation_legend_num)
            cbar.ax.yaxis.set_label_coords(-0.1, 0.5)  # Adjust label position if needed
    
        ind_0 = pair_inds[0, ind]
        ind_1 = pair_inds[1, ind]

        if ind_0 != ind_1 and title:
            ax_handle.set_title(
                "Correlation of Rings " + str(ind_0) + " and " + str(ind_1),
                fontsize=fontsize * 1.,
            )
        else:
            if title:
                ax_handle.set_title(
                    "Autocorrelation of Ring " + str(ind_0), fontsize=fontsize * 1.
                )

        x_t = ax_handle.get_xticks()
        sub = np.logical_or(x_t < 0, x_t > orient_corr.shape[2])
        x_t_new = np.delete(x_t, sub)
        x_t_new_int = np.round(x_t_new * pixel_size).astype(int)
        ax_handle.set_xticks(x_t_new)
        ax_handle.set_xticklabels(x_t_new_int, fontsize=fontsize * 1.)
        ax_handle.set_xlabel("Radial Distance (" + pixel_units + ")", fontsize=fontsize)

        ax_handle.invert_yaxis()
        ax_handle.set_ylabel("Relative Orientation (°)", fontsize=fontsize)
        y_ticks = np.linspace(0, orient_corr.shape[1] - 1, 10, endpoint=True)
        ax_handle.set_yticks(y_ticks)
        ax_handle.set_yticklabels(
            ["0", "", "", "30", "", "", "60", "", "", "90"], fontsize=fontsize * 1.
        )

        if calculate_coefs:
            y = np.arange(orient_corr.shape[2])
            if orient_corr[ind, 0, 0] > orient_corr[ind, -1, 0]:
                z = orient_corr[ind, 0, :]
            else:
                z = orient_corr[ind, -1, :]
            coefs = [np.max(z), np.min(z), y[-1] * 0.25, 2]
            bounds = ((1e-3, 0, 1e-3, 1.0), (np.inf, np.inf, np.inf, np.inf))
            coefs = curve_fit(fit_dist, y, z, p0=coefs, bounds=bounds)[0]
            coef_radial = coefs[2] * (np.log(1 / fraction_coefs) ** (1 / coefs[3]))

            x = np.arange(orient_corr.shape[1])
            if orient_corr[ind, 0, 0] > orient_corr[ind, -1, 0]:
                z = orient_corr[ind, :, 0]
            else:
                z = np.flip(orient_corr[ind, :, 0], axis=0)
            z = np.maximum(z, 1.0)
            coefs = [np.max(z), np.min(z), x[-1] * 0.25, 2]
            bounds = ((1e-3, 0, 1e-3, 1.0), (np.inf, np.inf, np.inf, np.inf))
            coefs = curve_fit(fit_dist, x, z, p0=coefs, bounds=bounds)[0]
            coef_annular = coefs[2] * (np.log(1 / fraction_coefs) ** (1 / coefs[3]))
            if orient_corr[ind, 0, 0] <= orient_corr[ind, -1, 0]:
                coef_annular = orient_corr.shape[1] - 1 - coef_annular
            pixel_size_annular = 90 / (orient_corr.shape[1] - 1)

            x_slope = np.argmin(
                np.abs(orient_corr[ind, :, :length_fit_slope] - 1.0), axis=0
            )
            y_slope = np.arange(length_fit_slope)
            coefs_slope = np.polyfit(y_slope, x_slope, 1)

            if ind_0 != ind_1:
                print("Correlation of Rings " + str(ind_0) + " and " + str(ind_1))
            else:
                print("Autocorrelation of Ring " + str(ind_0))
            print(
                str(np.round(fraction_coefs * 100).astype("int"))
                + "% probability radial distance = "
                + str(np.round(coef_radial * pixel_size, 0))
                + " "
                + pixel_units
            )
            print(
                str(np.round(fraction_coefs * 100).astype("int"))
                + "% probability annular distance = "
                + str(np.round(coef_annular * pixel_size_annular, 0))
                + " degrees"
            )
            print(
                "Relative orientation increases by 1 degree every "
                + str(np.round(1.0 / coefs_slope[0], 0).astype("int"))
                + " "
                + pixel_units
            )
            print("\n")

        if plot_overlaid_coefs:
            if num_plot > 1:
                ax_handle = ax[count]
            else:
                ax_handle = ax

            if orient_corr[ind, 0, 0] > orient_corr[ind, -1, 0]:
                ax_handle.plot(
                    np.array((coef_radial, coef_radial, 0.0)),
                    np.array((0.0, coef_annular, coef_annular)),
                    color=(1.0, 1.0, 1.0),
                )
                ax_handle.plot(
                    np.array((coef_radial, coef_radial, 0.0)),
                    np.array((0.0, coef_annular, coef_annular)),
                    color=(0.0, 0.0, 0.0),
                    linestyle="--",
                )
            else:
                ax_handle.plot(
                    np.array((coef_radial, coef_radial, 0.0)),
                    np.array(
                        (
                            orient_corr.shape[1] - 1,
                            coef_annular,
                            coef_annular,
                        )
                    ),
                    color=(1.0, 1.0, 1.0),
                )
                ax_handle.plot(
                    np.array((coef_radial, coef_radial, 0.0)),
                    np.array(
                        (
                            orient_corr.shape[1] - 1,
                            coef_annular,
                            coef_annular,
                        )
                    ),
                    color=(0.0, 0.0, 0.0),
                    linestyle="--",
                )
            ax_handle.plot(
                y_slope,
                y_slope.astype("float") * coefs_slope[0] + coefs_slope[1],
                color=(0.0, 0.0, 0.0),
                linestyle="--",
            )

    fig.tight_layout()

    if save_to_pdf:
        fig.savefig(pdf_filename, bbox_inches="tight")

    if returnfig:
        return fig,ax

# for py4DSTEM for plot_orientation_correlation_vis
def get_intensity(orient, x, y, t):
    """
    Copy-pasted from py4DSTEM version 13.17 to be used in the plot_orientation_correlation function, which was modified for
    visualization purposes.
    """
    # utility function to get histogram intensites

    x = np.clip(x, 0, orient.shape[0] - 2)
    y = np.clip(y, 0, orient.shape[1] - 2)

    xF = np.floor(x).astype("int")
    yF = np.floor(y).astype("int")
    tF = np.floor(t).astype("int")
    dx = x - xF
    dy = y - yF
    dt = t - tF
    t1 = np.mod(tF, orient.shape[2])
    t2 = np.mod(tF + 1, orient.shape[2])

    int_vals = (
        orient[xF, yF, t1] * ((1 - dx) * (1 - dy) * (1 - dt))
        + orient[xF, yF, t2] * ((1 - dx) * (1 - dy) * (dt))
        + orient[xF, yF + 1, t1] * ((1 - dx) * (dy) * (1 - dt))
        + orient[xF, yF + 1, t2] * ((1 - dx) * (dy) * (dt))
        + orient[xF + 1, yF, t1] * ((dx) * (1 - dy) * (1 - dt))
        + orient[xF + 1, yF, t2] * ((dx) * (1 - dy) * (dt))
        + orient[xF + 1, yF + 1, t1] * ((dx) * (dy) * (1 - dt))
        + orient[xF + 1, yF + 1, t2] * ((dx) * (dy) * (dt))
    )

    return int_vals
    
# for py4DSTEM for plot_orientation_correlation_vis
def set_intensity(orient, xy_t_int):
    """
    Copy-pasted from py4DSTEM version 13.17 to be used in the plot_orientation_correlation function, which was modified for
    visualization purposes.
    """
    # utility function to set flowline intensites

    xF = np.floor(xy_t_int[:, 0]).astype("int")
    yF = np.floor(xy_t_int[:, 1]).astype("int")
    tF = np.floor(xy_t_int[:, 2]).astype("int")
    dx = xy_t_int[:, 0] - xF
    dy = xy_t_int[:, 1] - yF
    dt = xy_t_int[:, 2] - tF

    inds_1D = np.ravel_multi_index(
        [xF, yF, tF], orient.shape[0:3], mode=["clip", "clip", "wrap"]
    )
    orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:, 3] * (1 - dx) * (
        1 - dy
    ) * (1 - dt)
    inds_1D = np.ravel_multi_index(
        [xF, yF, tF + 1], orient.shape[0:3], mode=["clip", "clip", "wrap"]
    )
    orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:, 3] * (1 - dx) * (
        1 - dy
    ) * (dt)
    inds_1D = np.ravel_multi_index(
        [xF, yF + 1, tF], orient.shape[0:3], mode=["clip", "clip", "wrap"]
    )
    orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:, 3] * (1 - dx) * (
        dy
    ) * (1 - dt)
    inds_1D = np.ravel_multi_index(
        [xF, yF + 1, tF + 1], orient.shape[0:3], mode=["clip", "clip", "wrap"]
    )
    orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:, 3] * (1 - dx) * (
        dy
    ) * (dt)
    inds_1D = np.ravel_multi_index(
        [xF + 1, yF, tF], orient.shape[0:3], mode=["clip", "clip", "wrap"]
    )
    orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:, 3] * (dx) * (
        1 - dy
    ) * (1 - dt)
    inds_1D = np.ravel_multi_index(
        [xF + 1, yF, tF + 1], orient.shape[0:3], mode=["clip", "clip", "wrap"]
    )
    orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:, 3] * (dx) * (
        1 - dy
    ) * (dt)
    inds_1D = np.ravel_multi_index(
        [xF + 1, yF + 1, tF], orient.shape[0:3], mode=["clip", "clip", "wrap"]
    )
    orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:, 3] * (dx) * (dy) * (
        1 - dt
    )
    inds_1D = np.ravel_multi_index(
        [xF + 1, yF + 1, tF + 1], orient.shape[0:3], mode=["clip", "clip", "wrap"]
    )
    orient.ravel()[inds_1D] = orient.ravel()[inds_1D] + xy_t_int[:, 3] * (dx) * (dy) * (
        dt
    )

    return orient
    
# for py4DSTEM for plot_orientation_correlation_vis
def make_orientation_histogram(
    bragg_peaks = None,
    radial_ranges: np.ndarray = None,
    orientation_map = None,
    orientation_ind: int = 0,
    orientation_growth_angles: np.array = 0.0,
    orientation_separate_bins: bool = False,
    orientation_flip_sign: bool = False,
    upsample_factor = 4.0,
    theta_step_deg = 1.0,
    sigma_x = 1.0,
    sigma_y = 1.0,
    sigma_theta=3.0,
    normalize_intensity_image: bool = False,
    normalize_intensity_stack: bool = True,
    progress_bar: bool = True,
):
    """
    Copy-pasted from py4DSTEM to be used in plot_orientation_correlation function that was modified locally for visualization
    purpose.

    Create an 3D or 4D orientation histogram from a braggpeaks PointListArray
    from user-specified radial ranges, or from the Euler angles from a fiber
    texture OrientationMap generated by the ACOM module of py4DSTEM.

    Args:
        bragg_peaks (BraggVectors):         bragg_vectos containing centered peak locations.
        radial_ranges (np array):           Size (N x 2) array for N radial bins, or (2,) for a single bin.
        orientation_map (OrientationMap):   Class containing the Euler angles to generate a flowline map.
        orientation_ind (int):              Index of the orientation map (default 0)
        orientation_growth_angles (array):  Angles to place into histogram, relative to orientation.
        orientation_separate_bins (bool):   whether to place multiple angles into multiple radial bins.
        upsample_factor (float):            Upsample factor
        theta_step_deg (float):             Step size along annular direction in degrees
        sigma_x (float):                    Smoothing in x direction before upsample
        sigma_y (float):                    Smoothing in x direction before upsample
        sigma_theta (float):                Smoothing in annular direction (units of bins, periodic)
        normalize_intensity_image (bool):   Normalize to max peak intensity = 1, per image
        normalize_intensity_stack (bool):   Normalize to max peak intensity = 1, all images
        progress_bar (bool):                Enable progress bar

    Returns:
        orient_hist (array):                4D array containing Bragg peak intensity histogram
                                            [radial_bin x_probe y_probe theta]
    """

    # coordinates
    theta = np.arange(0, 180, theta_step_deg) * np.pi / 180.0
    dtheta = theta[1] - theta[0]
    dtheta_deg = dtheta * 180 / np.pi
    num_theta_bins = np.size(theta)

    if orientation_map is None:
        # Input bins
        radial_ranges = np.array(radial_ranges)
        if radial_ranges.ndim == 1:
            radial_ranges = radial_ranges[None, :]
        radial_ranges_2 = radial_ranges**2
        num_radii = radial_ranges.shape[0]
        size_input = bragg_peaks.shape
    else:
        orientation_growth_angles = np.atleast_1d(orientation_growth_angles)
        num_angles = orientation_growth_angles.shape[0]
        size_input = [orientation_map.num_x, orientation_map.num_y]
        if orientation_separate_bins is False:
            num_radii = 1
        else:
            num_radii = num_angles

    size_output = np.round(
        np.array(size_input).astype("float") * upsample_factor
    ).astype("int")

    # output init
    orient_hist = np.zeros([num_radii, size_output[0], size_output[1], num_theta_bins])

    # Loop over all probe positions
    for a0 in range(num_radii):
        t = "Generating histogram " + str(a0)
        # for rx, ry in tqdmnd(
        #         *bragg_peaks.shape, desc=t,unit=" probe positions", disable=not progress_bar
        #     ):
        for rx, ry in tqdmnd(
            *size_input, desc=t, unit=" probe positions", disable=not progress_bar
        ):
            x = (rx + 0.5) * upsample_factor - 0.5
            y = (ry + 0.5) * upsample_factor - 0.5
            x = np.clip(x, 0, size_output[0] - 2)
            y = np.clip(y, 0, size_output[1] - 2)

            xF = np.floor(x).astype("int")
            yF = np.floor(y).astype("int")
            dx = x - xF
            dy = y - yF

            add_data = False

            if orientation_map is None:
                p = bragg_peaks.vectors[rx, ry]#########
                r2 = p.data["qx"] ** 2 + p.data["qy"] ** 2
                sub = np.logical_and(
                    r2 >= radial_ranges_2[a0, 0], r2 < radial_ranges_2[a0, 1]
                )
                if np.any(sub):
                    add_data = True
                    intensity = p.data["intensity"][sub]
                    t = np.arctan2(p.data["qy"][sub], p.data["qx"][sub]) / dtheta
            else:
                if orientation_map.corr[rx, ry, orientation_ind] > 0:
                    if orientation_separate_bins is False:
                        if orientation_flip_sign:
                            t = (
                                np.array(
                                    [
                                        (
                                            -orientation_map.angles[
                                                rx, ry, orientation_ind, 0
                                            ]
                                            - orientation_map.angles[
                                                rx, ry, orientation_ind, 2
                                            ]
                                        )
                                        / dtheta
                                    ]
                                )
                                + orientation_growth_angles
                            )
                        else:
                            t = (
                                np.array(
                                    [
                                        (
                                            orientation_map.angles[
                                                rx, ry, orientation_ind, 0
                                            ]
                                            + orientation_map.angles[
                                                rx, ry, orientation_ind, 2
                                            ]
                                        )
                                        / dtheta
                                    ]
                                )
                                + orientation_growth_angles
                            )
                        intensity = (
                            np.ones(num_angles)
                            * orientation_map.corr[rx, ry, orientation_ind]
                        )
                        add_data = True
                    else:
                        if orientation_flip_sign:
                            t = (
                                np.array(
                                    [
                                        (
                                            -orientation_map.angles[
                                                rx, ry, orientation_ind, 0
                                            ]
                                            - orientation_map.angles[
                                                rx, ry, orientation_ind, 2
                                            ]
                                        )
                                        / dtheta
                                    ]
                                )
                                + orientation_growth_angles[a0]
                            )
                        else:
                            t = (
                                np.array(
                                    [
                                        (
                                            orientation_map.angles[
                                                rx, ry, orientation_ind, 0
                                            ]
                                            + orientation_map.angles[
                                                rx, ry, orientation_ind, 2
                                            ]
                                        )
                                        / dtheta
                                    ]
                                )
                                + orientation_growth_angles[a0]
                            )
                        intensity = orientation_map.corr[rx, ry, orientation_ind]
                        add_data = True

            if add_data:
                tF = np.floor(t).astype("int")
                dt = t - tF

                orient_hist[a0, xF, yF, :] = orient_hist[a0, xF, yF, :] + np.bincount(
                    np.mod(tF, num_theta_bins),
                    weights=(1 - dx) * (1 - dy) * (1 - dt) * intensity,
                    minlength=num_theta_bins,
                )
                orient_hist[a0, xF, yF, :] = orient_hist[a0, xF, yF, :] + np.bincount(
                    np.mod(tF + 1, num_theta_bins),
                    weights=(1 - dx) * (1 - dy) * (dt) * intensity,
                    minlength=num_theta_bins,
                )

                orient_hist[a0, xF + 1, yF, :] = orient_hist[
                    a0, xF + 1, yF, :
                ] + np.bincount(
                    np.mod(tF, num_theta_bins),
                    weights=(dx) * (1 - dy) * (1 - dt) * intensity,
                    minlength=num_theta_bins,
                )
                orient_hist[a0, xF + 1, yF, :] = orient_hist[
                    a0, xF + 1, yF, :
                ] + np.bincount(
                    np.mod(tF + 1, num_theta_bins),
                    weights=(dx) * (1 - dy) * (dt) * intensity,
                    minlength=num_theta_bins,
                )

                orient_hist[a0, xF, yF + 1, :] = orient_hist[
                    a0, xF, yF + 1, :
                ] + np.bincount(
                    np.mod(tF, num_theta_bins),
                    weights=(1 - dx) * (dy) * (1 - dt) * intensity,
                    minlength=num_theta_bins,
                )
                orient_hist[a0, xF, yF + 1, :] = orient_hist[
                    a0, xF, yF + 1, :
                ] + np.bincount(
                    np.mod(tF + 1, num_theta_bins),
                    weights=(1 - dx) * (dy) * (dt) * intensity,
                    minlength=num_theta_bins,
                )

                orient_hist[a0, xF + 1, yF + 1, :] = orient_hist[
                    a0, xF + 1, yF + 1, :
                ] + np.bincount(
                    np.mod(tF, num_theta_bins),
                    weights=(dx) * (dy) * (1 - dt) * intensity,
                    minlength=num_theta_bins,
                )
                orient_hist[a0, xF + 1, yF + 1, :] = orient_hist[
                    a0, xF + 1, yF + 1, :
                ] + np.bincount(
                    np.mod(tF + 1, num_theta_bins),
                    weights=(dx) * (dy) * (dt) * intensity,
                    minlength=num_theta_bins,
                )

    # smoothing / interpolation
    if (sigma_x is not None) or (sigma_y is not None) or (sigma_theta is not None):
        if num_radii > 1:
            print("Interpolating orientation matrices ...", end="")
        else:
            print("Interpolating orientation matrix ...", end="")
        if sigma_x is not None and sigma_x > 0:
            orient_hist = gaussian_filter1d(
                orient_hist,
                sigma_x * upsample_factor,
                mode="nearest",
                axis=1,
                truncate=3.0,
            )
        if sigma_y is not None and sigma_y > 0:
            orient_hist = gaussian_filter1d(
                orient_hist,
                sigma_y * upsample_factor,
                mode="nearest",
                axis=2,
                truncate=3.0,
            )
        if sigma_theta is not None and sigma_theta > 0:
            orient_hist = gaussian_filter1d(
                orient_hist, sigma_theta / dtheta_deg, mode="wrap", axis=3, truncate=2.0
            )
        print(" done.")

    # normalization
    if normalize_intensity_stack is True:
        orient_hist = orient_hist / np.max(orient_hist)
    elif normalize_intensity_image is True:
        for a0 in range(num_radii):
            orient_hist[a0, :, :, :] = orient_hist[a0, :, :, :] / np.max(
                orient_hist[a0, :, :, :]
            )

    return orient_hist
    
'''
Modified from py4DSTEM version 13.17 for visualization purposes with transparent background.
'''
def show_qprofile(
    q,
    intensity,
    ymax=None,
    figsize=(12,4),
    returnfig=False,
    color='k',
    xlabel='q (pixels)',
    ylabel='Intensity (A.U.)',
    labelsize=16,
    ticklabelsize=14,
    grid='on',
    label=None,
    linestyle='-',  # New parameter for line style (default is solid line)
    output_file=None,  # Optional parameter to specify file name for saving
    **kwargs
):
    """
    Plots a diffraction space radial profile.
    Params:
        q               (1D array) the diffraction coordinate / x-axis
        intensity       (1D array) the y-axis values
        ymax            (number) max value for the yaxis
        color           (matplotlib color) profile color
        xlabel          (str)
        ylabel          (str)
        labelsize       size of x and y labels
        ticklabelsize   size of tick labels
        grid            'off' or 'on'
        label           a legend label for the plotted curve
        output_file     (str) path to save the figure with transparent background
    """
    if ymax is None:
        ymax = np.max(intensity)*1.05

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(q, intensity, color=color, label=label,linestyle=linestyle)
    ax.grid(grid == 'on')
    ax.set_ylim(0, ymax)
    ax.tick_params(axis='x', labelsize=ticklabelsize)
    ax.set_yticklabels([])
    ax.set_xlabel(xlabel, size=labelsize)
    ax.set_ylabel(ylabel, size=labelsize)

    if output_file:
        # Save the figure with a transparent background
        fig.patch.set_alpha(0.0)  # Transparent figure background
        ax.patch.set_alpha(0.0)   # Transparent axes background
        plt.savefig(output_file, format='png', transparent=True, bbox_inches='tight')
        print(f"Figure saved as {output_file}")
    
    if not returnfig:
        plt.show()
    else:
        return fig, ax

# For overlay_image_on_pdf
def pdf_to_image(pdf_path, output_image_path, dpi=300):
    '''
    This function transforms a pdf file to an image,
    needed for function : overlay_image_on_pdf
    '''
    # Convert PDF to image
    images = convert_from_path(pdf_path, dpi=dpi)
    # Save the first page as an image
    images[0].save(output_image_path, 'PNG')

# For overlay_image_on_pdf
def remove_black_background(image_path, output_image_path):
    '''
    This function removes background from image,
    needed for function : overlay_image_on_pdf
    '''
    img = Image.open(image_path).convert("RGBA")
    datas = img.getdata()
    
    new_data = []
    for item in datas:
        # If the pixel is almost black
        if item[0] < 10 and item[1] < 10 and item[2] < 10:
            new_data.append((255, 255, 255, 0))  # Transparent
        else:
            new_data.append(item)
    
    img.putdata(new_data)
    img.save(output_image_path, 'PNG')

# For overlapping orientaion maps 
def overlay_image_on_pdf(base_pdf_path, image_path, output_pdf_path):
    """
    This function overlays two images.
    The function can be used to overaly lamellar orientation map (black and white and background removed)
    and colorful PI-PI orientation map.
    
    To use this function you can use:
    if __name__ == "__main__":
        # Define file paths
        black_white_pdf_path = 'lamellar.pdf'
        colorful_pdf_path = 'color.pdf'
        processed_image_path = 'processed_lamellar.png'
        output_pdf_path = 'overlay_output.pdf'
        # Convert the black and white PDF to an image
        pdf_to_image(black_white_pdf_path, 'black_and_white_map.png')
        # Remove background from the black and white image
        remove_black_background('black_and_white_map.png', processed_image_path)
        # Overlay the processed image on the colorful PDF
        overlay_image_on_pdf(colorful_pdf_path, processed_image_path, output_pdf_path)
    """
    # Open the base PDF
    base_pdf = fitz.open(base_pdf_path)
    # Load the base page
    base_page = base_pdf.load_page(0)
    
    # Create a new PDF to save the result
    output_pdf = fitz.open()
    # Create a new page in the output PDF with the same dimensions
    output_page = output_pdf.new_page(width=base_page.rect.width, height=base_page.rect.height)
    
    # Draw the base page on the output page
    output_page.show_pdf_page(output_page.rect, base_pdf, 0)
    
    # Draw the image on the output page
    img_rect = fitz.Rect(0, 0, output_page.rect.width, output_page.rect.height)
    output_page.insert_image(img_rect, filename=image_path, overlay=True)
    
    # Save the resulting PDF
    output_pdf.save(output_pdf_path)
    
    # Close all PDFs
    base_pdf.close()
    output_pdf.close()