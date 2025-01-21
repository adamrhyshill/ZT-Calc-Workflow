# zt_plot.py


"""
In this example, we prepare a plot to compare the ZT of p- and n-type Pnma SnS
and SnSe as a function of doping level and temperature.

The calculation combines data from multiple papers:

    Approximate models for the lattice thermal conductivity of alloy
        thermoelectrics
    J. M. Skelton
    J. Mater. Chem. C 9 (35), 11772 (2021), DOI: 10.1039/D1TC02026A

    Thermoelectric Properties of Pnma and Rocksalt SnS and SnSe
    J. M. Flitcroft, I. Pallikara and J. M. Skelton
    Solids 3 (1), 155-176 (2022), DOI: 10.3390/solids3010011
    
    Thermoelectric properties of Pnma and R3m GeS and GeSe
    M. Zhang, J. Flictroft, S. Guillemot and J. Skelton
    J. Mater. Chem. C 11, 14833 (2023, DOI: 10.1039/D3TC02938G
"""


import numpy as np
import re
import os
import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import FuncFormatter

import argparse

import sys

sys.path.append(r"C:/Users/mbdxpah2/VS_code_work/ZT-Calc-Workflow")

from zt_calc_workflow.amset import read_amset_csv
from zt_calc_workflow.dataset import zt_dataset_from_data, dataset_to_2d
from zt_calc_workflow.phono3py import read_phono3py_kappa_csv
from zt_calc_workflow.plotting import setup_matplotlib

##### Space to specify script inputs ######


#### Define all input files here
def define_input_files():
    """Function to edit filenames to be used in ZT 2D plot and other input parameters

    Returns:
        Dictionary: input files with system name as key, and input files as values
    """
    input_files = {
        "Pm3m": [
            r"amset-STO-Cubic-P.csv",
            r"amset-STO-Cubic-N.csv",
            r"STO_cubic_klatt_NAC_Wigner.csv",
        ],
        "Pnma": [
            r"amset-STO-Ortho-P.csv",
            r"amset-STO-Ortho-N.csv",
            r"STO_ortho_klatt_NAC_Wigner.csv",
        ],
        "I4/mcm": [
            r"amset-STO-Tetra-P.csv",
            r"amset-STO-Tetra-P.csv",
            r"STO_tetra_klatt_NAC_Wigner.csv",
        ],
    }
    return input_files


#### Define other inputs if required
def define_other_inputs():
    """Function to pass input data to script when not defined at runtime

    Returns:
        list: plot labels to assign to plots in order
        int: minimum temperature to model
        int: maximum temperature to model
        list: contour levels to plot on ZT plot
    """
    # Change your plot labels here if not passed at runtime
    plot_labels = [
        "Pm3m-p-type",
        "Pm3m-n-type",
        "Pnma-p-type",
        "Pnma-n-type",
        "test",
        "test2",
        "test3",
    ]

    # Set temperatures here if not passed at runtime
    t_min, t_max = 300.0, 1000.0

    # Set contour levels here if not passed at runtime
    zt_contour_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    # Pass information to
    return plot_labels, t_min, t_max, zt_contour_levels


def populate_input_dict(input_files):
    """Creates dictionary from input strings to fit in with exisiting script

    Args:
        input_files (list): list of strings containing paths to files and system names

    Returns:
        dict: dictionary with key corresponding to system name and values corresponding to input csv files
    """
    input_dict = {}
    print(len(input_files))
    for i in range(int((len(input_files) / 4))):
        input_dict = {
            **input_dict,
            input_files[i * 4]: input_files[i * 4 + 1 : i * 4 + 4],
        }

    print(input_dict)

    return input_dict


##### End of space to specify script inputs ######


def plot_2D_func(
    input_files,
    specify_temp,
    output_name,
    quality,
    gridrow,
    gridcolumn,
    input_labels,
    contour_range,
    figure_size,
    skip,
    padding,
    uniformity,
):
    """Function to plot 2D ZT data from n, p and klatt data

    Args:
        input_files (_type_): _description_
        specify_temp (_type_): _description_
        output_name (_type_): _description_
        quality (_type_): _description_

    Returns:
        _type_: _description_
    """

    zt_data = {}
    t_min, t_max = 10000, -10000

    for k, (amset_p, amset_n, kappa) in input_files.items():
        kappa_data = read_phono3py_kappa_csv(kappa)

        # The Phono3py calculations were performed on structures with the axes
        # oriented differently to those in the AMSET calculations. We deal with
        # this by dropping the off-diagonal elements and relabelling the
        # diagonal elements in kappa_data.

        kappa_data.drop(columns=["kappa_yz", "kappa_xz", "kappa_xy"], inplace=True)
        kappa_data.columns = ["t", "kappa_yy", "kappa_zz", "kappa_xx", "kappa_ave"]

        amset_data_p = read_amset_csv(amset_p, check_uniform=uniformity)
        amset_data_n = read_amset_csv(amset_n, check_uniform=uniformity)

        # checking to find highest and lowest temperature
        if amset_data_p["t"].max() > t_max:
            t_max = amset_data_p["t"].max()
        if amset_data_n["t"].max() > t_max:
            t_max = amset_data_n["t"].max()
        if amset_data_p["t"].min() < t_min:
            t_min = amset_data_p["t"].min()
        if amset_data_n["t"].min() < t_min:
            t_min = amset_data_n["t"].min()

        # Combine AMSET and Phono3py data into ZT datasets, write out CSVs
        # Can be generalised to not require p and n... grid, count etc.
        zt_p = zt_dataset_from_data(amset_data_p, kappa_data)
        zt_n = zt_dataset_from_data(amset_data_n, kappa_data)
        zt_p_name = str(k) + "_p_ZT"
        zt_n_name = str(k) + "_n_ZT"
        zt_p_name = re.sub(r"[^a-zA-Z0-9 \n\.]", "_", zt_p_name)
        zt_n_name = re.sub(r"[^a-zA-Z0-9 \n\.]", "_", zt_n_name)
        zt_p_name += ".csv"
        zt_n_name += ".csv"
        os.makedirs("CSVs", exist_ok=True)

        zt_p.to_csv(str("CSVs/" + zt_p_name))
        zt_n.to_csv(str("CSVs/" + zt_n_name))

        # then convert to 2D data
        zt_data[k] = (
            dataset_to_2d(zt_p),
            dataset_to_2d(zt_n),
        )

    # Setup Matplotlib.
    setup_matplotlib()

    # Collect inputs from top of script
    input_params = define_other_inputs()

    # Custom formatters for labels.#
    # Either read labels from script or from command line
    if len(input_labels) == 0:
        subplot_labels = input_params[0]

    else:
        subplot_labels = input_labels

    def contour_fmt(v):
        if v % 1.0 == 0:
            return "{0:.0f}".format(v)

        if v % 0.5 == 0:
            return "{0:.1f}".format(v)

        return "{0:.2f}".format(v)

    def log_fmt(v, pos):
        return "$10^{{{0:.0f}}}$".format(v)

    log_formatter = FuncFormatter(log_fmt)

    # To use a common colour bar, we need to determine the "global" ZT_max
    # across all datasets in the range (t_min, t_max). Once we've done this,
    # we can create a matplotlib.colors.Normalize to colour the 2D plots.

    # set temperature to values if not allowed to calculate automatically
    if specify_temp != "":
        t_min, t_max = int(specify_temp[0]), int(specify_temp[1])

    print("Plotting ZT between ", t_min, " and ", t_max)

    global_zt_max = 0.0

    for data_p, data_n in zt_data.values():
        for n, t, data in data_p, data_n:
            t_mask = np.logical_and(t >= t_min, t <= t_max)
            global_zt_max = max(global_zt_max, data["zt_ave"][:, t_mask].max())

    print(global_zt_max)
    norm = Normalize(vmin=0.0, vmax=global_zt_max)

    # Plot.
    # Contour levels - if not provided, use pre-defined
    if len(contour_range) == 0:
        zt_contour_levels = input_params[3]
    # otherwise make a new list based on the increments
    else:
        # find number of decimal places of increment, for rounding
        decimal_str = str(contour_range[2])
        dec_places = len(str(decimal_str).split(".")[1])
        print(dec_places)
        # Create numpy array over contour range
        contour_array = np.arange(contour_range[0], contour_range[1], contour_range[2])
        # round to number of decimal places required for increment
        contour_array = np.round(contour_array, dec_places)
        # convert to list
        zt_contour_levels = contour_array.tolist()

    print("Plotting contours: ", zt_contour_levels)

    plt.figure(figsize=(figure_size[0] / 2.54, figure_size[1] / 2.54))

    # Use a GridSpec to divide the plot into a four subplot axes and a colour
    # bar axis.
    # rows = 4 * r + 1 and columns = columns
    # use gridrow and gridcolumn from inputs

    # check whether skip has been specified
    # plot differently if skipping files requested
    offset = 2
    if skip:
        offset = 1

    grid_spec = GridSpec((gridrow * 4), gridcolumn)
    grid_spec.update(top=0.90, bottom=0.08, wspace=padding[0], hspace=padding[1])

    col_grid = GridSpec(1, gridcolumn)
    col_grid.update(top=0.99, bottom=0.94)

    subplot_axes = []

    counter = 0
    for r in range(gridrow):
        for c in range(gridcolumn):
            if not skip or (skip and counter < len(input_files.keys())):
                subplot_axes.append(plt.subplot(grid_spec[4 * r : 4 * (r + 1), c]))
            counter += 1

    # Loop over SnS/SnSe and p- and n-type doping and draw a 2D colour plot
    # with contour lines.

    for i, k in enumerate(list(input_files.keys())):
        data_p, data_n = zt_data[k]

        for j, (n, t, data) in enumerate([data_p, data_n]):
            # if skip is enabled, skip the second file - fix to fit in with script (assumes n and p data always present)
            if skip and j == 1:
                continue
            # check if there are enough axes to data plot on - avoids crash
            if (offset * i + j) < len(subplot_axes):
                axes = subplot_axes[offset * i + j]

                t_mask = np.logical_and(t >= t_min, t <= t_max)

                x = np.log10(n)
                y = t[t_mask]
                z = data["zt_ave"].T[t_mask, :]

                axes.pcolormesh(x, y, z, norm=norm, shading="gouraud")

                cs = axes.contour(x, y, z, levels=zt_contour_levels, colors="r")
                axes.clabel(cs, cs.levels, inline=True, fmt=contour_fmt)

    # Adjust axis ranges and labels.

    for axes in subplot_axes:
        axes.set_ylim(t_min, t_max)
        axes.set_yticks(np.linspace(t_min, t_max, 6))

    for axes in subplot_axes:
        axes.xaxis.set_major_formatter(log_formatter)

    if skip:
        for axes in subplot_axes:
            axes.set_xlabel(r"Doping Level $n$ [cm$^{-3}$]")
        for axes in subplot_axes:
            axes.set_ylabel(r"$T$ [K]")
    else:
        for axes in subplot_axes[-gridcolumn:]:
            axes.set_xlabel(r"Doping Level $n$ [cm$^{-3}$]")
        for axes in subplot_axes[::gridcolumn]:
            axes.set_ylabel(r"$T$ [K]")

    # Add subplot labels.

    for i, (axes, label) in enumerate(zip(subplot_axes, subplot_labels)):
        subplot_label = AnchoredText(
            r"({0}) {1}".format(chr(97 + i), label), loc="lower left", frameon=True
        )

        subplot_label.patch.set_edgecolor("k")
        subplot_label.patch.set_facecolor((1.0, 1.0, 1.0, 0.5))

        axes.add_artist(subplot_label)

    # Add colour bar.
    cbar_axes = plt.subplot(col_grid[0, 0:gridcolumn])
    plt.colorbar(ScalarMappable(norm=norm), orientation="horizontal", cax=cbar_axes)
    cbar_axes.set_ylabel(r"$ZT$")
    # cbar_axes.set_in_layout(False)

    # Finalise and save.

    # plt.tight_layout(pad=float(padding))
    img_name = output_name + ".png"
    pdf_name = output_name + ".pdf"
    plt.savefig(img_name, dpi=quality)
    plt.savefig(pdf_name, dpi=quality)
    plt.close()
    print("Plot saved as ", img_name)
    print("Vector image saved as ", pdf_name)


### Running the ZT plot code - arguments are parsed here
if __name__ == "__main__":

    #### Defining script arguments for plotting ####

    # start setting up arguments
    parser = argparse.ArgumentParser(
        prog="general_zt_plot.py",
        description="Process arguments for ZT plotting script",
    )

    parser.add_argument(
        "-i",
        "--input_files",
        nargs="*",
        default="",
        help="Specify input files, separated by spaces. First entry is system name, followed by p data, n data and klatt data.",
    )

    # specify temperature range
    parser.add_argument(
        "-t",
        "--temp_specify",
        nargs="*",
        default="",
        help="Specify minimum and maximum temperature to plot",
    )

    # specify output path
    parser.add_argument(
        "-o",
        "--output",
        default="zt_plot",
        help="Specify filename for graph image",
    )

    # specify image output quality
    parser.add_argument(
        "-q",
        "--quality",
        default=300,
        help="Specify DPI for graph image quality",
    )

    # specify grid row layout
    parser.add_argument(
        "-gr",
        "--gridrow",
        default=3,
        help="Specify number of rows of graphs for plotting grid ",
    )

    # specify grid column layout
    parser.add_argument(
        "-gc",
        "--gridcolumn",
        default=3,
        help="Specify number of rows of graphs for plotting grid ",
    )

    # specify labels for plots
    parser.add_argument(
        "-l",
        "--labels",
        nargs="*",
        default="",
        help="Specify labels for plots, separated by spaces.",
    )

    # specify contour list for plots
    parser.add_argument(
        "-c",
        "--contour",
        nargs="*",
        default="",
        help="Define range for contour plot, first two values are minimum and maximum for the range, third value is the increment amount.",
    )

    # specify figure size in cm
    parser.add_argument(
        "-f",
        "--fig_size",
        nargs=2,
        default=[14, 14],
        help="Specify size in cm for figure in x and y",
    )

    # plot only one item
    parser.add_argument(
        "-s",
        "--skip",
        action="store_true",
        help="Plot only one file from set, default is first file given per item.",
    )

    # specify padding for layout
    parser.add_argument(
        "-p",
        "--padding",
        nargs=2,
        default=[0.24, 0.94],
        help="Specify padding for layout of plots, height then width ",
    )

    # plot only one item
    parser.add_argument(
        "-u",
        "--uniform",
        action="store_false",
        help="Forces data to be uniform, otherwise will flag non-uniform data and stop the calculation.",
    )

    # load arguments from script
    args = parser.parse_args()
    contour_float = list(map(float, args.contour))
    fig_size_float = list(map(float, args.fig_size))
    padding_float = list(map(float, args.padding))

    #### Checking routines for arguments ######

    # read input files from function
    if len(args.input_files) > 0:
        input_files = populate_input_dict(args.input_files)
    else:
        input_files = define_input_files()

    # check two temperatures are provided if custom range requested
    if len(args.temp_specify) == 1 or len(args.temp_specify) > 2:
        print("Two temperatures are required to specify plot temperature range")
        print(
            "Please re-run with two temperatures specified or leave blank for automatic temperature range"
        )
        exit()

    # check that max temp is higher than min temp, otherwise rearrange
    if len(args.temp_specify) > 0:
        if int(args.temp_specify[0]) > int(args.temp_specify[1]):
            print(
                "Minimum temperature is higher than maximum temperature, values have been swapped."
            )
            # save min temperature as variable
            temp = args.temp_specify[0]
            # swap min to max temperature
            args.temp_specify[0] = args.temp_specify[1]
            # swap max to min temperature saved in variable
            args.temp_specify[1] = temp

    # check enough information provided to create contours
    if len(args.contour) > 0 and len(args.contour) < 3:
        print("A start, end and increment value are required to specify contour range")
        print(
            "Please re-run with a start, end and increment value or leave blank to use default contour range"
        )
        exit()

    # check if contour needs rearranging like temperatures
    if len(contour_float) > 0:
        if contour_float[0] > contour_float[1]:
            print(
                "Contour minimum is higher than contour maximum, values have been swapped."
            )
            # save min contour as variable
            temp = contour_float[0]
            # swap min to max contour
            contour_float[0] = contour_float[1]
            # swap max to min contour saved in variable
            contour_float[1] = temp

    # remove . from output names
    if "." in args.output:
        args.output = ".".join(args.output.split(".")[:-1])

    # Read input data.
    plot_2D_func(
        input_files,
        args.temp_specify,
        args.output,
        args.quality,
        int(args.gridrow),
        int(args.gridcolumn),
        args.labels,
        contour_float,
        fig_size_float,
        args.skip,
        padding_float,
        args.uniform,
    )
