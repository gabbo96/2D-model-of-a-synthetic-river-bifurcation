import os
import json
import time
import xarray
import shutil  # to delete png files before the simulation starts
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import dorado.particle_track as pt
import dorado.routines as rt
from matplotlib.path import Path
from datetime import datetime  # just to name the simulations
from PIL import Image, ImageDraw  # PIL stands for the Pillow library,

# used here for making gifs and drawing transects
# used to generate points along a transect
from shapely.geometry import LineString
from anuga import plot_utils


def init_dorado_params(dx, theta, my_swwvals, gamma=0, diff_coeff=0, dry_depth=0.01):
    """Initialize an object of dorado's modelParams class.

    Parameters
    ----------
    dx : float
        spatial resolution of dorado grid
    theta : _type_
        _description_
    my_swwvals : dict
        dictionary containing the different variables retrieved from the ANUGA simulation.
        Must include 'depth'
    gamma : int, optional
        _description_, by default 0
    diff_coeff : int, optional
        _description_, by default 0
    dry_depth : float, optional
        _description_, by default 0.01

    Returns
    -------
    modelParams object (usually named "params" in codes)
    """
    params = pt.modelParams()
    params.dx = dx
    params.theta = theta
    params.gamma = gamma
    params.diff_coeff = diff_coeff
    params.model = "Anuga"
    params.topography = my_swwvals["depth"]
    params.dry_depth = dry_depth
    params.verbose = False

    # add ANUGA flow field
    params.depth = my_swwvals["depth"]
    params.stage = my_swwvals["stage"]
    params.qx = my_swwvals["qx"]
    params.qy = my_swwvals["qy"]

    return params


def plot_ANUGA_sim_domain_maps(my_swwvals, out_dir_path):
    """Creates a map for water depth, stage, bathymetry and momentum components.
    Output .png files are saved in the folder prescribed in the inputs.

    Parameters
    ----------
    my_swwvals : dict
        dictionary returned by the import_ANUGA_flow_field function, containing
        the outputs of the ANUGA simulation interpolated on dorado's regular grid
    out_dir_path : str
        Path to the directory where the output maps are saved.

    Returns
    -------
    None
    """
    for var in my_swwvals.keys():
        if not var in ["x", "y"]:
            plt.figure(figsize=(10, 10))
            plt.title(var)
            plt.imshow(my_swwvals[var], cmap="viridis")
            plt.colorbar(fraction=0.018, label=var)
            plt.savefig(
                os.path.join(out_dir_path, f"{var}.png"), dpi=300, bbox_inches="tight"
            )

    return None


def plot_ANUGA_sim_maps(
    my_swwvals,
    out_dir_path,
    t_pts,
    win_width=60,
    ncols=2,
    center_point=None,
    colormap="viridis",
    bathy_colormap="terrain",
    plot_transects="downstream",
    lw_tr=2,
    suptitle=None,
):
    """Plots and saves to file maps of water depth, bathymetry and
    water stage. Each image is composed of one subplot for each bifurcation,
    centered on the centroid of the transect of the upstream channel.

    Parameters
    ----------
    my_swwvals : dict
        dictionary returned by the import_ANUGA_flow_field function, containing
        the outputs of the ANUGA simulation interpolated on dorado's regular grid
    t_pts : dict
        Dictionary containing the coordinates of the transects, that are plotted over the map if plot_transects is true.
        The transect on the upstream branch is used to center the plot if center_point is not given.
    out_dir_path : str
        path to the folder where to save the output maps
    win_width : int, default 60
        Width and height (in dorado units) of the map for each bifurcation,
        centered around the centroid of the upstream transect
    ncols: int, default 2
        Number of columns for the subplots
    center_point: tuple, default None
        (x,y) of a point. If given, the subplot of each bifurcation is centered on that point.
    colormap: str, optional
        colormap used in the maps. Default is "viridis"
    plot_transects: str, optional
        option to plot transects. Available options: "all", "downstream" (upstream transect not plotted),
        "none"
    lw_rt: float, optional
        linewidth for the transects
    suptitle: str, optional
        String used as a suptitle for the subplots. Default is None

    Returns
    -------
    None
    """
    # Extract bathymetry and put it in a new element of my_swwvals
    my_swwvals["bathy"] = my_swwvals["stage"] - my_swwvals["depth"]
    for var in ["depth", "stage", "bathy"]:
        # Create subplots
        nrows = int(np.ceil(len(t_pts) / 2))
        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows)
        )

        if ncols == 1:
            axs_iter = [axs]
        else:
            axs_iter = axs.flatten()

        for ax, bif in zip(axs_iter, t_pts):
            if center_point is not None:
                x_c, y_c = center_point
            else:
                # Center map on centroid of the upstream transect
                for t in t_pts[bif]:
                    if t[-1] == "0":
                        x_c = int(0.5 * (t_pts[bif][t][0][0] + t_pts[bif][t][1][0]))
                        y_c = int(0.5 * (t_pts[bif][t][0][1] + t_pts[bif][t][1][1]))

            # upper left corner (lower x, lower y in dorado reference system) of the window
            win_ul = [x_c - int(win_width / 2), y_c - int(win_width / 2)]

            # Extract values of the current variable nearby the current bifurcation
            bif_map = my_swwvals[var][
                win_ul[0] : win_ul[0] + win_width, win_ul[1] : win_ul[1] + win_width
            ]

            # Plot the extracted values
            if var == "bathy":
                cbar_label = "Bed elevation"
                cmap = bathy_colormap
            else:
                cbar_label = var
                cmap = colormap
            im = ax.imshow(bif_map, cmap=cmap)
            fig.colorbar(im, ax=ax, fraction=0.018, label=f"{cbar_label} [m]")

            # Plot the transects
            if plot_transects == "all" or plot_transects == "downstream":
                for t in t_pts[bif]:
                    if not (plot_transects == "downstream" and t[-1] == "0"):
                        x_tr = np.array(
                            [
                                t_pts[bif][t][0][0] - win_ul[0],
                                t_pts[bif][t][1][0] - win_ul[0],
                            ]
                        )
                        y_tr = np.array(
                            [
                                t_pts[bif][t][0][1] - win_ul[1],
                                t_pts[bif][t][1][1] - win_ul[1],
                            ]
                        )
                        ax.plot(y_tr, x_tr, "o-", lw=lw_tr, color="black")

            # Esthetics
            ax.set_title(f"{bif}")

        if suptitle is not None:
            fig.suptitle(suptitle)

        if len(t_pts) % 2 != 0:
            fig.delaxes(axs_iter.flat[-1])

        plt.tight_layout()

        plt.savefig(
            os.path.join(out_dir_path, f"{var}.png"), dpi=300, bbox_inches="tight"
        )
    return None


def plot_flow_field(
    my_swwvals,
    out_dir_path,
    t_pts,
    subsample_step=4,
    win_width=60,
    ncols=2,
    center_point=None,
    arrow_dict={"scale": 5, "hwidth": 16, "hlength": 14},
    plot_transects=True,
    lw_tr=1,
    suptitle=None,
):
    """Plots a vector map of an ANUGA flow field converted to a cartesian grid

    Parameters
    ----------
    anuga_sim_name: str
        Title of the figure, used only if suptitle==True
    my_swwvals: dict
        dictionary returned by the import_ANUGA_flow_field function
    out_dir_path: str
        path to the folder that will containg the output png file
    subsample_step: int, optional
        not all cells are considered to draw the vector field. Instead, flow field is
        resampled taking one value each subsample_step cells
    win_width : int, default 50
        Width and height (in dorado units) of the map for each bifurcation,
        centered around the centroid of the upstream transect
    ncols: int, default 2
        Number of columns for the subplots
    plot_transects: bool, optional
        if true, transects are plotted over the flow field
    lw_rt: float, optional
        linewidth for the transects (used only if plot_transects is True)
    suptitle: str, optional
        String used as a suptitle for the subplots. Default is None

    Returns
    -------
    None
    """
    # Arrows parameters
    scale = arrow_dict["scale"]  # the lower the value, the longer the arrows
    hwidth = arrow_dict["hwidth"]  # width of the head of the arrow. Default is 3
    hlength = arrow_dict["hlength"]  # length of the heaf of the arrow
    haxlength = hlength - 0.5

    # Create subplots
    nrows = int(np.ceil(len(t_pts) / 2))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))

    if ncols == 1:
        axs_iter = [axs]
    else:
        axs_iter = axs.flatten()

    for ax, bif in zip(axs_iter, t_pts):
        if center_point is not None:
            x_c, y_c = center_point
        else:
            # Center map on centroid of the upstream transect
            for t in t_pts[bif]:
                if t[-1] == "0":
                    x_c = int(0.5 * (t_pts[bif][t][0][0] + t_pts[bif][t][1][0]))
                    y_c = int(0.5 * (t_pts[bif][t][0][1] + t_pts[bif][t][1][1]))

        # upper left corner (lower x, lower y in dorado reference system) of the window
        win_ul = [x_c - int(win_width / 2), y_c - int(win_width / 2)]

        my_swwvals_cut = {}
        for var in ["qx", "qy", "depth"]:
            my_swwvals_cut[var] = my_swwvals[var][
                win_ul[0] : win_ul[0] + win_width, win_ul[1] : win_ul[1] + win_width
            ]

        # Extract x and y components of flow velocity
        # we must change the sign of the y component
        # for consistency with the (x, y) reference system
        u = my_swwvals_cut["qx"] / my_swwvals_cut["depth"]
        v = my_swwvals_cut["qy"] / my_swwvals_cut["depth"]

        # Create a grid of coordinates using np.meshgrid.
        x_vect = np.arange(u.shape[1])
        y_vect = np.arange(u.shape[0])
        X, Y = np.meshgrid(x_vect, y_vect)

        # Subsample the data
        step = subsample_step
        X_subsampled = X[::step, ::step]
        Y_subsampled = Y[::step, ::step]
        u_subsampled = u[::step, ::step]
        v_subsampled = v[::step, ::step]

        # Plot flow field
        ax.quiver(
            X_subsampled,
            Y_subsampled,
            u_subsampled,
            v_subsampled,
            scale=scale,
            units="width",
            headwidth=hwidth,
            headaxislength=haxlength,
            headlength=hlength,
        )

        # Plot the transects
        if plot_transects:
            for t in t_pts[bif]:
                x_tr = np.array(
                    [t_pts[bif][t][0][0] - win_ul[0], t_pts[bif][t][1][0] - win_ul[0]]
                )
                y_tr = np.array(
                    [t_pts[bif][t][0][1] - win_ul[1], t_pts[bif][t][1][1] - win_ul[1]]
                )
                ax.plot(y_tr, x_tr, linestyle="dotted", lw=lw_tr, color="orange")

        ax.invert_yaxis()

        # Set axes boundaries to "zoom in" to the current bifurcation
        # ax.set_xlim([y_c - int(win_width / 2), y_c + int(win_width / 2)])
        # ax.set_ylim([x_c + int(win_width / 2), x_c - int(win_width / 2)])

        # Esthetics
        ax.set_title(f"{bif}")

    if suptitle is not None:
        fig.suptitle(suptitle)

    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir_path, "flow_field.png"),
        bbox_inches="tight",
        dpi=600,
    )
    return None


def plot_cross_sections(
    t_pts,
    params,
    out_dir_path,
    ncols=3,
    plot_velocity_xs=False,
    plot_lims="WLD",
    xs_name="",
    suptitle=None,
):
    """Plots all cross-sections identified by the given transects.

    Parameters
    ----------
    t_pts : `dict``
        dictionary containing the coordinates of each transect, organized as
        t_pts[<bifurcation name>][<transect name>] = [(x1, y1), (x2, y2)]

    params : `dorado.particle_track.modelParams` object
        Object dorado's modelParams class

    out_dir_path : `str`
        path to the folder where to save the output image

    ncols : int, optional
        by default 3

    plot_velocity_xs: bool, optional
        Plot the transverse profile of normal flow velocity. Default is False.

    plot_lims : str, optional
        Sets xlim and ylim. Available options: "WLD". By default "WLD"

    xs_name : str, optional
        If given, this string is appended to the name of the output png file.

    Returns
    -------
    None
    """
    print("Plotting cross-sections...")

    MainCamp_idx = None  # index for Main-Campground cross-sections, used to rescale
    # only these subplots

    fig, axs = plt.subplots(nrows=len(t_pts), ncols=ncols, figsize=(15, 9))

    # If nrows is 1, make axs a 2D array for consistent indexing
    if len(t_pts) == 1:
        axs = np.array(axs)[np.newaxis, :]

    for i, bif in enumerate(t_pts):
        if bif == "Main Pass - Campground Pass":
            MainCamp_idx = i
        for j, transect in enumerate(t_pts[bif]):
            sample_int = 1  # meters
            tr_line = LineString(t_pts[bif][transect])

            # Interpolate points at intervals along line
            dist_along_line = 0  # start at length 0
            tr_pts = []  # list of new, equally spaced points
            while dist_along_line < tr_line.length:
                tr_pt = tr_line.interpolate(dist_along_line)
                tr_pts.append(tr_pt)
                dist_along_line += sample_int / params.dx  # move to next distance/point

            # Computes components of unit vector normal to the transect
            t_dx = tr_pts[-1].x - tr_pts[0].x
            t_dy = tr_pts[-1].y - tr_pts[0].y
            n_x = t_dy / (t_dy**2 + t_dx**2) ** 0.5
            n_y = -t_dx / (t_dy**2 + t_dx**2) ** 0.5

            # Iterate over the points
            q_norm = []
            h = []
            D = []
            eta = []
            for tr_pt in tr_pts:
                # Retrieve coordinates and unit discharges of nearest cell
                x_near_cell = int(round(tr_pt.x))
                y_near_cell = int(round(tr_pt.y))

                qx_cell = params.qx[x_near_cell, y_near_cell]
                qy_cell = params.qy[x_near_cell, y_near_cell]

                # Transform qx and qy in the same reference system of x and y
                qx_transf = -qy_cell
                qy_transf = qx_cell

                # Perform dot product to compute the projection of the local momentum
                # vector on the unit vector normal to the transect
                q_norm.append(abs(qx_transf * n_x + qy_transf * n_y))
                h.append(params.stage[x_near_cell, y_near_cell])
                D.append(params.depth[x_near_cell, y_near_cell])
                eta.append(
                    params.stage[x_near_cell, y_near_cell]
                    - params.depth[x_near_cell, y_near_cell]
                )

            # Plot cross sections
            lines = []
            labels = []  # just for the legend
            y = np.linspace(0, tr_line.length * params.dx, num=len(h))
            (line1,) = axs[i, j].plot(y, h)
            (line2,) = axs[i, j].plot(y, eta)
            label1 = r"Free surface $h$"
            label2 = r"Bed elevation $\eta$"
            lines.extend([line1, line2])
            labels.extend([label1, label2])
            ylabel = r"$h$, $\eta$ [m]"
            if plot_velocity_xs:
                (line3,) = axs[i, j].plot(y, np.array(q_norm) / np.array(D))
                label3 = r"Normal velocity $u$"
                lines.append(line3)
                labels.append(label3)
                ylabel += r"$u$ [m/s]"
            axs[i, j].set_xlabel("Distance along transect [m]")
            axs[i, j].set_ylabel(ylabel)
            axs[i, j].set_title(f"{transect}")
            axs[i, j].grid()
            if plot_lims == "WLD":
                axs[i, j].set_xlim([0, 1200])
                axs[i, j].set_ylim([-5, 2])

    if MainCamp_idx is not None:
        for ax in axs[MainCamp_idx, :]:
            ax.set_ylim([-20, 2])
    fig.legend(
        handles=lines,
        labels=labels,
        loc="upper center",
        ncols=3,
    )

    if suptitle is not None:
        fig.suptitle(suptitle)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(
        os.path.join(out_dir_path, f"cross_sections{xs_name}.png"),
        dpi=500,
        bbox_inches="tight",
    )
    return None


def plot_long_profiles(t_pts, params, out_dir_path, xs_name="", plot_eta=False):
    """Plots longitudinal profiles for a set of transects

    Parameters
    ----------
    t_pts : `dict`
        dictionary containing the coordinates of each transect, organized as
        t_pts[<transect name>] = [(x1, y1), (x2, y2)]

    params : `dorado.particle_track.modelParams` object
        Object dorado's modelParams class

    out_dir_path : `str`
        path to the folder where to save the output image

    plot_lims : `str`, optional
        Sets xlim and ylim. Available options: "WLD". By default "WLD"

    xs_name : `str`, optional
        If given, this string is appended to the name of the output png file.

    plot_eta : `bool`, optional
        If set to True, the bed elevation along the transect is plotted along with
        the water surface elevation

    Returns
    -------
    None
    """
    fig, axs = plt.subplots(nrows=len(t_pts), ncols=1, figsize=(12, 9))

    for j, transect in enumerate(t_pts):
        sample_int = 1  # meters
        tr_line = LineString(t_pts[transect])

        # Interpolate points at intervals along line
        dist_along_line = 0  # start at length 0
        tr_pts = []  # list of new, equally spaced points
        while dist_along_line < tr_line.length:
            tr_pt = tr_line.interpolate(dist_along_line)
            tr_pts.append(tr_pt)
            dist_along_line += sample_int / params.dx  # move to next distance/point

        # Iterate over the points
        h = []
        D = []
        eta = []
        for tr_pt in tr_pts:
            # Retrieve coordinates and unit discharges of nearest cell
            x_near_cell = int(round(tr_pt.x))
            y_near_cell = int(round(tr_pt.y))

            h.append(params.stage[x_near_cell, y_near_cell])
            D.append(params.depth[x_near_cell, y_near_cell])
            eta.append(
                params.stage[x_near_cell, y_near_cell]
                - params.depth[x_near_cell, y_near_cell]
            )

        # Plot cross sections
        lines = []
        labels = []  # just for the legend
        y = np.linspace(0, tr_line.length * params.dx, num=len(h))

        (line,) = axs[j].plot(y, h)
        label = r"Free surface $h$"
        lines.append(line)
        labels.append(label)

        if plot_eta:
            (line,) = axs[j].plot(y, eta)
            label = r"Bed elevation $\eta$"
            lines.append(line)
            labels.append(label)
        axs[j].set_xlabel("Distance along transect [m]")
        axs[j].set_ylabel("Elevation [m]")
        axs[j].set_title(f"{transect}")
        axs[j].grid()
        axs[j].legend(handles=lines, labels=labels)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(
        os.path.join(out_dir_path, f"cross_sections{xs_name}.png"),
        bbox_inches="tight",
        dpi=600,
    )
    return None


def clear_temp_folder(figs_path):
    folder = figs_path
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))
    return None


def create_sim_folder(
    in_d,
    out_dir_path,
    sim_name,
    sim_par_dir_name=None,
    datetime_sim_dir=True,
    make_fig_subdir=True,
    exist_ok=False,
):
    """Create a directory for a simulation, copying the given input dictionary as a json file.
    By default, the simulation folder is created as a subfolder of the folder located at the out_dir_path,
    i.e. at out_dir_path/sim_name.

    If a name for the parent directory is given, the simulation folder is instead created at
    out_dir_path/sim_par_dir_name/sim_name.

    If a name for a datetime subfolder is given, the simulation folder is created at
    out_dir_path/sim_par_dir_name/sim_name/datetime.

    Parameters
    ----------
    in_d : dict
        Input dictionary of the simulation. A copy of it is saved in the simulation
        folder as a json file.
    out_dir_path : str
        Path to the root folder containing all outputs of all simulations
    sim_dir_name : str
        Name to the folder containing the simulation outputs.


    Returns
    -------
    sim_dir_path
        Path to the directory where all the outputs of the simulations will be saved.
    figs_temp_path
        Path where the figures will be saved. If make_fig_subdir is False, it is equal
        to sim_dir_path
    """
    if sim_par_dir_name is None:
        sim_dir_path = os.path.join(out_dir_path, sim_name)
    else:
        sim_dir_path = os.path.join(out_dir_path, sim_par_dir_name, sim_name)

    if datetime_sim_dir:
        sim_dir_path = os.path.join(
            sim_dir_path, datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
        )
    os.makedirs(sim_dir_path, exist_ok=exist_ok)

    if make_fig_subdir:
        figs_temp_path = os.path.join(sim_dir_path, "figs_temp")
        os.makedirs(figs_temp_path, exist_ok=True)
    else:
        figs_temp_path = sim_dir_path

    # Store input dictionary as json file in the simulation folder
    with open(os.path.join(sim_dir_path, "inputs.json"), "w") as f:
        json.dump(in_d, f, indent=4)

    return sim_dir_path, figs_temp_path


def plot_bathymetry_seedpts(bathy, seedinds):
    plt.figure(figsize=(5, 5))
    plt.imshow(bathy, cmap="jet")
    plt.colorbar(fraction=0.018, label="water depth")
    plt.title("Gridded Depth Array with seeding points")
    plt.scatter(seedinds[0][1], seedinds[0][0], c="r")
    plt.show()
    return None


def import_ANUGA_flow_field(
    anuga_file_path, dx, UTM_region=None, bounds_path=None, t_idx=-1
):
    """
    Reads a .sww or .nc NetCDF file produced by an ANUGA simulation
    and returns a dictionary containing the coordinates, the water depth and
    the velocity commponents at each mesh centroid.

    Parameters
    ----------
    anuga_file_path : str
        path to the NetCDF file (either .sww or .nc)  produced by ANUGA simulations

    dx : float,
        dorado grid resolution in meters

    UTM_region : str, default None
        name of rectangular region to use for the first crop of the anuga output. Available options: "WLD".
        If None, no cropping is performed.

    bounds_path : str, default None
        path to the csv file containing the list of UTM coordinates of the vertices of the opping polygon for dorado's
        unstruct2grid() function. If None, no cropping is performed.

    t_idx: int, default -1
        index of the timestep of the ANUGA simulation to use for the flow field.
        -1 uses the last timestep. Use 29 for comparison with Siyoon's Spring and Fall campaign estimates.

    Returns
    -------
    my_swwvals: dict
        dictionary containing, for each centroid of the mesh elements,
    the coordinates of the centroids (keys 'x' and 'y'), the bathymetry ('bathy'), the water depth ('depth') and water level ('stage'),
    and the components of the local momentum ('qx' and 'qy'), defined as the product of the local water depth
    and the corresponding component of the local flow velocity.
    """

    # Load dataset
    swwvals_all = xarray.load_dataset(anuga_file_path)

    # Extract coordinates of UTM grid
    x_sww = swwvals_all["x"].data
    y_sww = swwvals_all["y"].data

    # Retrieve file type and extract interpolation points accordingly
    anuga_file_type = os.path.splitext(anuga_file_path)[-1]
    if anuga_file_type == ".sww":
        # Retrieve matrix containing water depth
        swwvals_all["depth"] = swwvals_all["stage"] - swwvals_all["elevation"]
        # Use directly the coordinates of the vertexes
        x_pts = x_sww[:]
        y_pts = y_sww[:]
    elif anuga_file_type == ".nc":
        # Compute centroid coordinates
        mesh = swwvals_all["mesh"].data
        x_pts = (x_sww[mesh[0, :]] + x_sww[mesh[1, :]] + x_sww[mesh[2, :]]) / 3
        y_pts = (y_sww[mesh[0, :]] + y_sww[mesh[1, :]] + y_sww[mesh[2, :]]) / 3

    # Create list of (x, y) tuples
    coordinates = [(x, y) for x, y in zip(x_pts, y_pts)]

    # If a region is set, create a new list of coordinates
    if UTM_region == "WLD":
        # Use Path to find the interpolation points in the WLAD region
        coord_WLD = [
            [639000.0, 3278800.0],
            [659600.0, 3278800.0],
            [659600.0, 3251600.0],
            [639000.0, 3251600.0],
        ]
        path = Path(coord_WLD)
        # Crop region using matplotlib's Path
        inside = path.contains_points(coordinates)
        x_inside = x_pts[inside]
        y_inside = y_pts[inside]
        coordinates = [(x, y) for x, y in zip(x_inside, y_inside)]
    elif UTM_region == "WLD_small":
        coord_WLD = [
            [645500.0, 3270094.0],
            [654091.0, 3270094.0],
            [654091.0, 3260741.0],
            [645500.0, 3260741.0],
        ]
        path = Path(coord_WLD)
        # Crop region using matplotlib's Path
        inside = path.contains_points(coordinates)
        x_inside = x_pts[inside]
        y_inside = y_pts[inside]
        coordinates = [(x, y) for x, y in zip(x_inside, y_inside)]

    else:
        inside = np.ones_like(x_pts).astype(bool)
        x_inside = x_pts
        y_inside = y_pts

    # Initialize dictionary that extract the timestep and variables of
    # interest from ANUGA's big "swwvals_all" output file, along with the
    # centroid coordinates
    my_swwvals = {"x": x_inside, "y": y_inside}
    my_vars = ["depth", "stage", "qx", "qy"]  # keys of output dictionary

    # List the variable names of the dataset (i.e. of the sww/nc file)
    # corresponding to the variables defined in my_sww_vars
    if anuga_file_type == ".nc":
        anuga_vars = ["depth", "stage", "xmom", "ymom"]
    elif anuga_file_type == ".sww":
        anuga_vars = ["depth", "stage", "xmomentum", "ymomentum"]

    # Transform the map of water depth to a uniform grid and plot it
    if bounds_path is not None:
        bounds = np.loadtxt(bounds_path, delimiter=",")
    else:
        bounds = None
    myInterp, _ = pt.unstruct2grid(
        coordinates, swwvals_all["depth"][t_idx, inside], dx, 3, bounds
    )

    # Extract timesteps & variables of interest using the myInterp function
    for my_var, anuga_var in zip(my_vars, anuga_vars):
        my_swwvals[my_var] = myInterp(np.array(swwvals_all[anuga_var][t_idx, inside]))

    # Add bathymetry to output dictionary
    my_swwvals["bathy"] = my_swwvals["stage"] - my_swwvals["depth"]

    return my_swwvals


def extract_ch_tips_transects_UTM_coord(coords_file_path):
    df_t = pd.read_csv(coords_file_path)

    t_coord = {}
    for branch_name, branch_group in df_t.groupby("Branch"):
        t_coord[branch_name] = [(x, y) for x, y in zip(branch_group.X, branch_group.Y)]

    return t_coord


def extract_ch_tips_transects_index_coord(transects_coord_path, x, y, bathy, dx):
    """
    Reads a csv file and reads the coordinates of the transects.
    Then groups them in a dictionary

    coords_file_path: path to the csv file containing the transect coordinates
    x, y: matrices containing the coordinates of the dorado grid
    bathy: any matrix having the same dimensions of the dorado grid
    dx: dorado grid spacing
    """
    # Extract x and y UTM coordinates of transect vertexes from csv file and
    # convert them in dorado 'index' coordinate
    t_coord = extract_ch_tips_transects_UTM_coord(transects_coord_path)

    t_pts = {}
    for branch in t_coord:
        t_pts[branch] = pt.coord2ind(
            t_coord[branch], (min(x), min(y)), np.shape(bathy), dx
        )
    return t_pts


def extract_transects_UTM_coord(coords_file_path):
    """
    Reads a .csv file containing the coordinates of the transects,
    then returns a dictionary where each transect is stored as a
    list of two (x, y) tuples, and grouped by bifurcation
    """
    df_t = pd.read_csv(coords_file_path)

    t_coord = {}
    for bif_name, bif_group in df_t.groupby("Bif_name"):
        t_coord[bif_name] = {}
        for t_name, t_group in bif_group.groupby("label"):
            t_coord[bif_name][t_name] = [(x, y) for x, y in zip(t_group.X, t_group.Y)]

    return t_coord


def extract_transects_index_coord(transects_coord_path, my_swwvals, dx):
    """Reads a csv file and reads the coordinates of the transects.
    Then groups them in a dictionary

    Parameters
    ----------
    transects_coord_path : str
        path to the coordinates (in ANUGA's reference system) of the transects
    my_swwvals : dict
        dictionary containing the different variables retrieved from the ANUGA simulation.
        Must include 'x', 'y' and 'depth'
    dx : float
       dx: dorado grid spacing

    Returns
    -------
    dict
        Dictionary that contains the transect coordinates, organised as dict[bif][t]
    """
    # Extract x and y UTM coordinates of transect vertexes from csv file and
    # convert them in dorado 'index' coordinate
    t_coord = extract_transects_UTM_coord(transects_coord_path)

    t_pts = {}
    for bif in t_coord:
        t_pts[bif] = {}
        for t in t_coord[bif]:
            t_pts[bif][t] = pt.coord2ind(
                t_coord[bif][t],
                (min(my_swwvals["x"]), min(my_swwvals["y"])),
                np.shape(my_swwvals["depth"]),
                dx,
            )
    return t_pts


def transect_get_Q_D_W(transect, params, sample_int=1.0):
    """
    Uses the ANUGA output converted to a uniform grid by dorado's unstruct2grid function to compute
    water discharge Q, average depth D and wetted width W along a given transect.

    Used by the compute_DeltaQ_DeltaD function.

    Inputs:
    ------
    - transect: list of two tuples, each containing the (x, y) coordinates of one end of the transect
                expressed in the local "index" reference system of the dorado grid
    - params: instance of dorado's params class, containing the information on the flow field
    - sample_int : space interval (in meters) at which points are sampled along the transect

    """
    tr_line = LineString(transect)

    # Interpolate points at intervals along line
    dist_along_line = 0  # start at length 0
    tr_pts = []  # list of new, equally spaced points
    while dist_along_line < tr_line.length:
        tr_pt = tr_line.interpolate(dist_along_line)
        tr_pts.append(tr_pt)
        dist_along_line += sample_int / params.dx  # move to next distance/point

    # Computes components of unit vector normal to the transect
    t_dx = tr_pts[-1].x - tr_pts[0].x
    t_dy = tr_pts[-1].y - tr_pts[0].y
    n_x = t_dy / (t_dy**2 + t_dx**2) ** 0.5
    n_y = -t_dx / (t_dy**2 + t_dx**2) ** 0.5

    # Initialize lists
    q_norm = []
    D = []
    u_x = []  # x component of flow velocity
    u_y = []  # y component of flow velocity

    # Iterate over the points
    for pt in tr_pts:
        # Retrieve coordinates and unit discharges of nearest cell
        x_near_cell = int(round(pt.x))
        y_near_cell = int(round(pt.y))
        qx_cell = params.qx[x_near_cell, y_near_cell]
        qy_cell = params.qy[x_near_cell, y_near_cell]

        if np.isnan(qx_cell) or np.isnan(qy_cell):
            # Skip this point
            continue

        # Transform qx and qy in the same reference system of x and y
        qx_transf = -qy_cell
        qy_transf = qx_cell

        # Perform dot product to compute the projection of the local momentum
        # vector on the unit vector normal to the transect
        q_norm.append(qx_transf * n_x + qy_transf * n_y)

        # Retrieve water depth of the nearest cell
        D_cell = params.depth[x_near_cell, y_near_cell]
        if D_cell > params.dry_depth:
            # Append to water depth list and compute local flow velocity
            D.append(D_cell)
            u_x.append(qx_transf / D_cell)
            u_y.append(qy_transf / D_cell)

    # Integrate to compute the total discharge in m3/s
    Q_tot = np.sum(q_norm) * sample_int

    # Compute average water depth and total wetted width
    D_avg = np.mean(D)
    W = len(D) * sample_int

    # Compute width perpendicular to average flow direction
    u_norm = ((np.mean(u_x)) ** 2 + (np.mean(u_y)) ** 2) ** 0.5
    nu_x = np.mean(u_x) / u_norm  # x component of average velocity unit vector
    nu_y = np.mean(u_y) / u_norm  # y component of average velocity unit vector
    # cosine of the angle between average flow direction
    cos_alpha = nu_x * n_x + nu_y * n_y
    # and unit vector normal to the transect
    W_perp_to_flow = W * cos_alpha
    return abs(Q_tot), D_avg, abs(W_perp_to_flow)


def compute_flow_asymmetries(transects_coord_path, my_swwvals, params, out_path):
    """
    Uses the "transect_get_Q_D_W" function to retrieve the water discharge,
    water depth and wetted width asymmetries at each bifurcation.

    Assumes that the column "labels" of the transects in the file at transects_coord_path
    end with "L", "R" or "0"

    Creates a dictionary for each asymmetry and saves it to a json file in the specificed path.

    All dictionaries share the same structure: dict[bifurcation_name][var_name],
    where var_name is either *_L, *_R, *_0 or Delta*
    - DeltaQ: dictionary for water discharge asymmetry (Q_L-Q_R)/Q_0
    - DeltaD: dictionary for water depth asymmetry (D_L-D_R)/(D_L+D_R)
    - DeltaW: dictionary for wetted width asymmetry (W_L-W_R)/(W_L+W_R)

    Returns
    -------
    None
    """
    print("Computing flow asymmetries...")

    t_pts = extract_transects_index_coord(transects_coord_path, my_swwvals, params.dx)

    DeltaQ = {}
    DeltaD = {}
    DeltaW = {}
    for bif in t_pts:
        DeltaQ[bif] = {}
        DeltaD[bif] = {}
        DeltaW[bif] = {}
        for t in t_pts[bif]:
            Q, D, W = transect_get_Q_D_W(t_pts[bif][t], params)
            DeltaQ[bif][f"Q_{t[-1]}"] = np.round(Q, 1)
            DeltaD[bif][f"D_{t[-1]}"] = np.round(D, 1)
            DeltaW[bif][f"W_{t[-1]}"] = np.round(W, 1)
        DeltaD[bif]["DeltaD_0"] = np.round(
            ((DeltaD[bif]["D_L"] - DeltaD[bif]["D_R"]) / DeltaD[bif]["D_0"]), 3
        )
        DeltaD[bif]["DeltaD_sum"] = np.round(
            (
                (DeltaD[bif]["D_L"] - DeltaD[bif]["D_R"])
                / (DeltaD[bif]["D_L"] + DeltaD[bif]["D_R"])
            ),
            3,
        )
        DeltaQ[bif]["DeltaQ"] = np.round(
            ((DeltaQ[bif]["Q_L"] - DeltaQ[bif]["Q_R"]) / DeltaQ[bif]["Q_0"]), 3
        )
        DeltaQ[bif]["(Q_L+Q_R)/Q_0"] = np.round(
            ((DeltaQ[bif]["Q_L"] + DeltaQ[bif]["Q_R"]) / DeltaQ[bif]["Q_0"]), 3
        )
        DeltaW[bif]["DeltaW"] = np.round(
            (
                (DeltaW[bif]["W_L"] - DeltaW[bif]["W_R"])
                / (DeltaW[bif]["W_L"] + DeltaW[bif]["W_R"])
            ),
            3,
        )

    # Save dictionaries to json files
    json.dump(DeltaQ, open(os.path.join(out_path, "Delta_Q.json"), "w"), indent=4)
    json.dump(DeltaD, open(os.path.join(out_path, "Delta_D.json"), "w"), indent=4)
    json.dump(DeltaW, open(os.path.join(out_path, "Delta_W.json"), "w"), indent=4)

    print("Asymmetries successfully written to json files\n")
    return None


def run_dorado_simulation(
    in_d, my_swwvals, seedinds, sim_folder_path, figs_temp_path, area_xylims="WLD"
):
    """Runs a dorado simulation on the given flow field

    Parameters
    ----------
    in_d : dict
        Dictionary containing the input parameters of the simulation.
        Must have the keys dx, theta, dt, n_step and save_walk_data.
        The duration of the simulation is set by the product dt*n_step. They are defined separately just
        to have a better control on the output figures (which are produced at each step).
    my_swwvals : dict
        dictionary containing the different variables retrieved from the ANUGA simulation.
        Must include 'x', 'y' and 'depth'
    seedinds : list
        list of raster index coordinates [(x1, y1), (x2, y2)] of the seeding points
    sim_folder_path : str
        path to the dorado simulation folder
    figs_temp_path : str
        path of the folder where the screenshots of the particle positions (used by the gif creator)
    area_xylims : str, optional
        defines xlims and ylims of the plot, by default "WLD" (which is the only option available)

    Returns
    -------
    params : modelParams
        output object of dorado's modelParams class
    walk_data : dict
        output dictionary containing the history of each particle
    """
    # Initialize an instance of the `params` class, and use ANUGA outputs and
    # the dorado parameters to populate its attributes
    params = pt.modelParams()
    params.dx = in_d["dx"]
    params.theta = in_d["theta"]
    params.gamma = 0.0
    params.diff_coeff = 0.0
    params.model = "Anuga"
    params.topography = my_swwvals["depth"]
    params.dry_depth = 0.01
    params.verbose = False

    # target times are in seconds
    target_times = np.linspace(
        in_d["dt"], in_d["n_step"] * in_d["dt"], num=in_d["n_step"]
    )
    walk_data = {}

    for i, t in enumerate(target_times):
        # ----------------------------- DORADO ITERATION ----------------------------- #
        # update water depth, stage and momentum components
        params.depth = my_swwvals["depth"]
        params.stage = my_swwvals["stage"]
        params.qx = my_swwvals["qx"]
        params.qy = my_swwvals["qy"]

        # generate particles
        particles = pt.Particles(params)
        if i == 0:
            particles.generate_particles(in_d["Np"], [seedinds[0][0]], [seedinds[0][1]])
        else:
            particles.generate_particles(0, [], [], 0, "random", walk_data)

        walk_data = particles.run_iteration(t)

        # ----------------------------------- PLOTS ---------------------------------- #
        # Generate new plots
        xiam, yiam, tiam = rt.get_state(walk_data)  # Most recent locations

        show_depth = params.depth.copy()
        show_depth[show_depth <= params.dry_depth] = np.nan

        if i == 0:
            x0a, y0a, t0a = rt.get_state(walk_data, 0)  # Starting locations

            # Initialize figure
            fig = plt.figure(dpi=300, figsize=(5, 5))
            ax = fig.add_subplot(111)
            if area_xylims == "WLD":
                ax.set_ylim([900, 300])
                ax.set_xlim([150, 750])

            # Topo background
            im_background = ax.imshow(
                params.topography, vmin=0, vmax=1.5, cmap="gist_gray"
            )
            cax = fig.add_axes(
                [
                    ax.get_position().x1 + 0.01,
                    ax.get_position().y0,
                    0.02,
                    ax.get_position().height,
                ]
            )
            cbar = plt.colorbar(im_background, cax=cax)
            cbar.set_label("Topography")

            # Depth foreground
            im = ax.imshow(show_depth, cmap="Blues", vmin=0, vmax=5, alpha=0.2)

            # New location
            newloc_a_M = ax.scatter(
                yiam, xiam, facecolors="brown", edgecolors=None, s=0.08, alpha=0.9
            )

        # Update figure with new locations, save hourly
        im.set_data(show_depth)
        newloc_a_M.set_offsets(np.array([yiam, xiam]).T)

        plt.draw()
        ax.set_title("Hour %s" % (t / 3600))

        plot_filename = f"output{i}.png"
        plt.savefig(os.path.join(figs_temp_path, plot_filename), bbox_inches="tight")

        print(f"Particles routed to hour {t/3600:.2f}")
    plt.close()

    # Save walk_data to a json file in the simulation folder
    if in_d["save_walk_data"]:
        json.dump(
            walk_data,
            open(os.path.join(sim_folder_path, "walk_data.json"), "w"),
            indent=4,
        )
    return params, walk_data


def create_gif(figs_path, sim_name_folder_path, overwrite_gif=False):
    """Creates the gif for the dorado simulation.

    Parameters
    ----------
    figs_path : str
        path to the folder where the png figures created during the dorado simulation
        are saved
    sim_name_folder_path : str
        parent to the folder containing all simulations sharing the same name (i.e.,
        the same set of parameters). If the simulation folder is a datetime subfolder,
        this parameter corresponds to the parent folder of the simulation's
    overwrite_gif : bool, optional
        overwrite an existing gif, by default False

    Returns
    -------
    None
    """
    out_gif_path = os.path.join(sim_name_folder_path, "animation.gif")
    if not os.path.isfile(out_gif_path) or overwrite_gif:
        print("Generating gif...\n")
        image_filenames = os.listdir(figs_path)
        images = []
        for i in range(len(image_filenames)):
            images.append(Image.open(os.path.join(figs_path, f"output{i}.png")))
        images[0].save(
            out_gif_path, save_all=True, append_images=images[1:], duration=100, loop=0
        )
    else:
        print("Gif already created for this group of dorado simulations.\n")
    return None


def fluxline(walk_data, pt1, pt2, raster_size, method="fractional"):
    """Measure the net flux through a transect defined by two points. Result
    should only reflect net flux, so bi-directional flows shouldn't be a problem.

    Inputs:
        walk_data: Dorado travel history
        pt1: First point of transect, tuple given in dorado index coordinates
        pt2: Second point of transect, tuple given in dorado index coordinates,
        raster_size: Shape of underlying domain rasters, e.g. stage.shape
        method: "fractional" or "total"
    Outputs:
        flux_net: Net flux through transect, expressed as fraction of total particles
    """

    # Draw the transect using PIL library.
    # width=2 ensures that perfeclty 45° inclined transects are not an issue

    # Create a new  B/W image with the given size
    image = Image.new("L", raster_size)
    draw = ImageDraw.Draw(image)  # Create an ImageDraw object
    draw.line([pt1[0], pt1[1], pt2[0], pt2[1]], fill=1, width=2)  # Draw the line

    transect = np.array(image).T  # PIL's reference system is (x,y),
    # contrary to OpenCV's cv2 (y, x)
    # transect = cv2.line(np.zeros(raster_size),
    #                    (pt1[1], pt1[0]), (pt2[1], pt2[0]), 1, 2)

    # Delineate upstream from downstream by breaking domain along line
    grid_y, grid_x = np.meshgrid(
        np.arange(0, raster_size[1]), np.arange(0, raster_size[0])
    )
    sides = np.array(
        (pt2[0] - pt1[0]) * (grid_y - pt1[1]) - (pt2[1] - pt1[1]) * (grid_x - pt1[0])
        < 0
    ).astype(float)

    # Measure net flux
    flux_net = 0.0
    Np_tracer = len(walk_data["xinds"])
    non_fluxes = []

    # Loop through particles
    for i in list(range(Np_tracer)):
        flux_occurred = False
        # Give lists shorter names to clean up code
        x = walk_data["xinds"][i]
        y = walk_data["yinds"][i]
        # Loop through iterations
        for j in list(range(len(x))):
            # Nelson's version of error handling
            if x[j] >= transect.shape[0] or y[j] >= transect.shape[1]:
                print(
                    "Warning: index %s is out of bounds with shape %s"
                    % ((x[j], y[j]), transect.shape)
                )
            else:
                try:
                    if transect[x[j], y[j]] == 1 and transect[x[j - 1], y[j - 1]] == 0:
                        # If particle just entered transect, record entrance side
                        entered_from = sides[x[j - 1], y[j - 1]]
                    elif (
                        transect[x[j], y[j]] == 0 and transect[x[j - 1], y[j - 1]] == 1
                    ):
                        # If particle just exited transect, save to flux measurement
                        exited_to = sides[x[j], y[j]]
                        flux_net += exited_to - entered_from
                        if exited_to - entered_from > 0:
                            flux_occurred = True
                        elif exited_to - entered_from < 0:
                            flux_occurred = False
                except:
                    print((x[j], y[j]), transect.shape)
        if not flux_occurred:
            non_fluxes.append(i)

    if method == "fractional":
        flux_net = flux_net / Np_tracer
        # print('Net fractional flux out of transect: %s' % flux_net)

    return flux_net, non_fluxes


def compute_DeltaQS(
    transects_coord_file_path,
    walk_data,
    dx,
    my_swwvals,
    sim_folder_path,
    plot_bif_maps=False,
    sim_name=None,
):
    """Estimate sediment partitioning using ADCP-like transects. Saves the resulting values
    of sediment discharge asymmetry in a json file named "DeltaQ_S.json", and optionally plots a map for each bifurcation.

    Parameters
    ----------
    transects_coord_file_path : str
        path to the transesct coordinates
    walk_data : dict
        dictionary returned by the run_dorado_simulation function
    dx : float
        gridsize of the dorado simulation
    my_swwvals : dict
        dictionary containing the flow field of the ANUGA simulation projected onto
        dorado's cartesian grid, e.g. by the `import_ANUGA_flow_field` function
    sim_folder_path : str
        path to the dorado simulation folder
    sim_name : str
        name of the simulation. Used only as a title for the plot, thus only if plot_bif_maps is True.

    Returns
    -------
    None
    """
    print("Computing DeltaQS...\n")
    time_deltaQ_S = time.time()

    t_pts = extract_transects_index_coord(
        transects_coord_file_path,
        my_swwvals,
        dx,
    )

    # Use the fluxline function to compute the sediment discharge asymmetry for each bifurcation, and save the output to a json file
    DeltaQS = {}
    for bif in t_pts:
        print(f"Computing DeltaQS for {bif}...")
        DeltaQS[bif] = {}
        for t in t_pts[bif]:
            flux_net, non_fluxes = fluxline(
                walk_data, *t_pts[bif][t], my_swwvals["depth"].shape
            )
            DeltaQS[bif][f"Qs_{t[-1]}"] = flux_net
        DeltaQS[bif]["DeltaQ_S"] = (
            DeltaQS[bif]["Qs_L"] - DeltaQS[bif]["Qs_R"]
        ) / DeltaQS[bif]["Qs_0"]
        DeltaQS[bif]["(Qs_L+Qs_R)/Qs_0"] = (
            DeltaQS[bif]["Qs_L"] + DeltaQS[bif]["Qs_R"]
        ) / DeltaQS[bif]["Qs_0"]

    with open(os.path.join(sim_folder_path, "DeltaQ_S.json"), "w") as f:
        json.dump(DeltaQS, f, indent=4)

    if plot_bif_maps:
        # Plot the detail of each bifurcation, along with the transects and the sediment discharge asymmetry
        fig, axs = plt.subplots(nrows=int(np.ceil(len(t_pts) / 2)), ncols=2)
        cell_buffer = 50  # buffer for the axis bounds in the plots of the transects

        cmap = mpl.colormaps["viridis"]
        colors = cmap(np.linspace(0.2, 0.8, len(DeltaQS)))

        for ax, bif in zip(axs.flatten(), DeltaQS):
            # Plot water depth as background
            im = ax.imshow(my_swwvals["depth"], cmap="viridis")
            fig.colorbar(im, ax=ax, fraction=0.018, label="Water depth")

            # Plot the transects
            for color, t in zip(colors, t_pts[bif]):
                x_tr = [t_pts[bif][t][0][1], t_pts[bif][t][1][1]]
                y_tr = [t_pts[bif][t][0][0], t_pts[bif][t][1][0]]
                ax.plot(x_tr, y_tr, "o-", color=color)

            # Esthetics
            ax.set_xlim([max(x_tr) - cell_buffer, max(x_tr) + cell_buffer])
            ax.set_ylim([max(y_tr) + cell_buffer, max(y_tr) - cell_buffer])
            ax.set_title(
                f"{bif}\n"
                + r"$(Q_{sL}-Q_{sR})/Q_{s0}$ = "
                + rf"${DeltaQS[bif]['DeltaQ_S']:.2f}$",
                fontsize=14,
            )
        if sim_name is not None:
            fig.suptitle(sim_name, fontsize=14)

        plt.tight_layout()
        plt.savefig(
            os.path.join(sim_folder_path, "bifurcations_DeltaQS.png"),
            bbox_inches="tight",
            dpi=300,
        )

    print(
        f"\nDeltaQS computation completed after {(time.time()-time_deltaQ_S)/60:.1f} minutes\n"
    )
    return None


def compute_Qs_ch_tips(chtip_t_coord_path, walk_data, my_swwvals, out_path, dx):
    """_summary_

    Parameters
    ----------
    chtip_t_pts : dict
        dictionary containing the coordinates of the channel tip transects.
    walk_data : dict
        output dictionary of the dorado simulation.
    my_swwvals : dict
        dictionary containing the different variables retrieved from the ANUGA simulation.
        Must include 'x', 'y' and 'depth'
    out_path : str
        path to the folder where the output json file will be saved
    dx : float
        dorado gridsize in meters

    Returns
    -------
    None
    """
    print("Computing Qs at channel tips...\n")
    time_start = time.time()

    chtip_t_pts = extract_ch_tips_transects_index_coord(
        chtip_t_coord_path, my_swwvals["x"], my_swwvals["y"], my_swwvals["depth"], dx
    )
    Qs_ch_tips = {}
    for t in chtip_t_pts:
        flux_net, non_fluxes = fluxline(
            walk_data, *chtip_t_pts[t], my_swwvals["depth"].shape
        )
        Qs_ch_tips[t] = flux_net
    with open(os.path.join(out_path, "chtips_Qs.json"), "w") as f:
        json.dump(Qs_ch_tips, f, indent=4)

    print(f"\nComputation completed after {(time.time()-time_start)/60:.1f} minutes\n")
    return None


def manualFlowThruTransect(
    transect_coord: list, anuga_file_path: str, sww_ind=-1, sample_int=1, verbose=False
) -> float:
    """Function created by N. Tull (and slightly modified by me in the error handling), much faster than anuga's builtin get_flow_throug_cross_section.
    Only considers one timestep of the ANUGA simulation, selected by means of the sww_ind
    keyword argument.

    Parameters
    ----------
    transect_coord : list
        List containing (x,y) tuples of the two ends of the transects
    anuga_file_path : str
        Path to anuga output as .sww file
    sww_ind : int, optional
        Model time index, e.g., sww_ind=4*24*4 would be at the 4-hour mark of the
        simulation if the anuga yield step is 900 seconds, by default -1
    sample_int : int, optional
        Interval between points along the transect, in meters, by default 1

    Returns
    -------
    float
        Flow discharge through the transect
    """

    # sww_file = xarray.load_dataset(anuga_file_path)
    sww_file = anuga_file_path

    # create shapely LineString object from two end points of transect
    tr_line = LineString(transect_coord)

    # interpolate points at intervals along line
    distAlongLine = 0  # start at length 0
    tr_pts = []  # list of new, equally spaced points
    while distAlongLine < tr_line.length:
        tr_pt = tr_line.interpolate(distAlongLine)
        tr_pts.append(tr_pt)
        distAlongLine += sample_int  # move to next distance/point

    # get x/y components of transect
    dx = tr_pts[-1].x - tr_pts[0].x
    dy = tr_pts[-1].y - tr_pts[0].y
    # get x/y components of normal unit vector, which is always clockwise from transect direction
    n_x = dy / (dy**2 + dx**2) ** 0.5
    n_y = -1 * dx / (dy**2 + dx**2) ** 0.5

    # get anuga output
    p = plot_utils.get_output(sww_file)
    pc = plot_utils.get_centroids(p)

    # loop thru transect points, compute normal fluxes
    pts_not_in_triangles = False
    fluxes = []
    for pt in tr_pts:
        try:
            # get anuga triangle containing point
            triangle = plot_utils.get_triangle_containing_point(p, (pt.x, pt.y))
            # get x/y momentum (m^2/s)
            xmom_i = pc.xmom[sww_ind, triangle]
            ymom_i = pc.ymom[sww_ind, triangle]
            # calculate momentum in direction of normal vector (dot product)
            mom_i_n = xmom_i * n_x + ymom_i * n_y
            # append to list of momentums, to integrate later
            fluxes.append(mom_i_n)
        except Exception as e:
            pts_not_in_triangles = True
            pass
    # integrate flows by summing across transect and multiplying by interval length
    Q_tot = np.sum(fluxes) * sample_int

    if verbose and pts_not_in_triangles:
        print("Some points were not found within a triangle")

    return Q_tot


def compute_deltaQ_from_flow_field(anuga_file_path, transects_coords_path):
    # Retrieve list of UTM coordinate tuples (x, y) for each transect
    t_coord = extract_transects_UTM_coord(transects_coords_path)

    # Compute water discharge for each transect and water discahrge asymmetry
    # for each bifurcation
    DeltaQ = {}
    for bif in t_coord:
        DeltaQ[bif] = {}
        for t in t_coord[bif]:
            time, Q = manualFlowThruTransect(t_coord[bif][t], anuga_file_path)
            DeltaQ[bif][f"Q_{t[-1]}"] = Q[-1]
        DeltaQ[bif]["DeltaQ"] = (DeltaQ[bif]["Q_L"] - DeltaQ[bif]["Q_R"]) / DeltaQ[bif][
            "Q_0"
        ]

    return DeltaQ
