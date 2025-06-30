[![DOI](https://zenodo.org/badge/1011017346.svg)](https://doi.org/10.5281/zenodo.15771789)

This folder contains Python scripts, libraries, and Jupyter notebooks used for the research paper "Controlling factors on water and sediment partitioning at deltaic bifurcations", currently in press at Journal of Geophysical Research - Earth Surface.

Below is a concise documentation of the contents of the repository.

#### WLD stands for Wax Lake Delta, while SDB stands for Simplified Deltaic Bifurcation (the synthetic geometry mentioned in the paper)

## Python Scripts
- **dorado_on_WLD.py**: Script for running dorado simulations on the Wax Lake Delta (WLD) model.
- **dorado_on_SDB.py**: Script for running dorado simulations on the synthetic geometry representing a Simplified Deltaic Bifurcation (SDB).
- **dorado_atx.py**: Contains various functions for importing ANUGA flow fields, plotting profiles, creating simulation folders, and running dorado simulations. Used for both WLD and SDB simulations.
- **geometry.py**: Contains functions for handling objects in the 3D space. Used in the creation of the SDB mesh. This file is also available in a dedicated [public repository](https://github.com/gabbo96/geometry)
  
## Jupyter Notebooks
- **bifo_mesh.ipynb**: Notebook for creating and analyzing the SDB computational mesh and perform ANUGA simulations on it.

## Auxiliary Files
- **ENV.yml**: Conda environment file for replicating the python virtual environment used in the study.
- **science.mplstyle**: Matplotlib style settings for plotting.
- **SDB_domain.png**: Image of the SDB domain.
- **seedloc_coords.json**: JSON file containing coordinates of seeding locations.
- **transects_coords.csv**: CSV file containing coordinates of the transects for the analysis on the WLD bifurcations, used to compute the flux partitioning and plot the cross-sections.
- **WLAD_boundary.csv**: CSV file containing the UTM coordinates of the vertices of the polygon used to crop the output of the Wax Lake Delta ANUGA simulations.
- the folder **dorado_simulations** contains an example of a `dorado` simulation performed on the Wax Lake Delta and presented in the paper.
- the folder **SDB_simulations** contains an example of the ANUGA and `dorado` simulations performed on the Simplified Deltaic Bifurcation. The length of the bifurcates of this sample simulation is considerably lower than that employed in the paper (see the supplementary information of the manuscript).

## `dorado` simulations on the Wax Lake Delta require the outputs of the calibrated ANUGA simulations
The ANUGA simulations on the Wax Lake Delta are not included in this repository, but they are freely available at:

Wright, K., & Passalacqua, P. (2024). Delta-x: Calibrated ANUGA hydrodynamic outputs for the atchafalaya basin, MRD, LA. ORNL DAAC, Oak Ridge, Tennessee, USA. doi: 10.3334/ORNLDAAC/2306

The simulation output files used in the paper are named "Hydro_WLAD_20210401_DXS.nc" (Spring Campaign) and "Hydro_WLAD_20210821_DXF.nc" (Fall Campaign).

## Configuration Files
### ANUGA simulations on SDB (inputs_anuga_SDB.json), used by `bifo_mesh.ipynb`
- `save_outputs`: Boolean flag to save the ANUGA outputs.
- `path_output_anuga_sims_dir`: Path to the directory where the output of the ANUGA simulations will be saved.
- `compute_t_Q_time`: Boolean flag to compute the time series of the flow discharge at each transect.
- `camp`: select the parameters of the function used to compute the input flow discharge (see `pars`), retrieved from the fits on the Fall and Spring DeltaX campaign
- `pars`: parameters of the function used to compute the input flow discharge as $Q = a \cdot W_U + b$, where $W_U$ is the width of the upstream channel and $a$ and $b$ are the first and second element of the list of the selected campaign.
- `max_triangle_area`: Maximum area of the triangles in the ANUGA mesh.
- `stage_diff`: parameter of the transmissive boundary condition.
- `yieldstep`: ANUGA yieldstep parameter of the `domain.evolve()` function.
- `duration`: Duration of the ANUGA simulation [s].
- `dx_dorado`: Grid spacing of the dorado grid, used to project the flow field of the simulation [m].
- `W_U`: Width of the upstream channel [m].
- `E_W`: ratio between the width of the downstream channels and the width of the upstream channel (`E_W=(W_L+W_R)/W_U`).
- `DeltaW`: ratio between the difference in the width of the downstream channels and their sum (`DeltaW=(W_L-W_R)/(W_L+W_R)`).
- `phi_sum_deg`: sum of the bifurcation angles (`phi_sum_deg=(phi_L+phi_R)`) [deg].
- `phi_diff_deg`: difference of the bifurcation angles (`phi_diff_deg=phi_L-phi_R`) [deg].
- `bathy_step`: Dimensionless step in the bathymetry, scaled with the upstream water depth.
- `S_U`, `S_L`, `S_R`: Slopes of the upstream and downstream channels [m/m].
- `L_U`: Length of the upstream channel, scaled with the backwater length.
- `L_LR`: Length of the downstream channels, scaled with the backwater length.
- `Manning_n`: Manning's roughness coefficient [s m^-1/3].

### `dorado` simulations on SDB (inputs_dorado_SDB.json), used by `dorado_on_SDB.py`
- `anuga_sims_dir_name`: Name of the directory containing the ANUGA simulations.
- `anuga_sim_name`: Name of the ANUGA simulation, which must correspond to the name of a subfolder of the folder specified in `anuga_sims_dir_name`.
- `overwrite_dorado_sim`: Boolean flag to overwrite the dorado simulation.
- `transects_filename`: Name of the file containing the coordinates of the transects. The file will be searched within the `coords` subfolder of the folder of the ANUGA simulation. Both the subfolder `coords` and the files containing the transects coordinates are generated in `bifo_mesh.ipynb`.
- `seedloc_length`: Distance of the seeding location from the inlet of the upstream channel, as a percentage of the upstream channel length. If set to 0, the longitudinal x coordinate of the seeding point (which originates at the upstream end of the upstream channel) is computed by default as equal to one channel width `W_U`.
- `Np`: Number of particles to release.
- `theta`: `dorado` parameter controlling the dependence of the routing direction on the water depth of the neighboring cells
- `dx`: Grid spacing of the dorado grid [m].
- `dt`: Time step of the dorado simulation [s]. Note that this parameter only determines the frequency with which the position of the particles is stored in the output dictionary `walk_data.json`. At each step, multiple iterations of the random walk algorithm are performed until the time of each particle reaches the prescribed value for time
- `n_step`: Number of time steps of the dorado simulation. This parameter mainly defines the duration of the simulation, which equals `dt * n_step`.
- `save_walk_data`: Boolean flag to save the output dictionary `walk_data.json` in the folder of the `dorado` simulation.

### `dorado` simulations on WLD (inputs_dorado_WLD.json), used by `dorado_on_WLD.py`
- `path_anuga_sims`: Path to the directory containing the output files of the ANUGA simulations.
- `anuga_file_name`: Name of the ANUGA simulation, which must correspond to the name of a subfolder of the folder specified in `path_anuga_sims`.
- `anuga_timestep`: Time step of the ANUGA simulation corresponding to the flow field employed in the `dorado` simulation.
- `transects_filename`: Name of the file containing the UTM coordinates of the transects of the WLD bifurcations, which must be saved in the same folder of this script.
- `path_output_dorado_sims_dir`: Path to the directory where the output of the `dorado` simulations will be saved.
- `test_mode`: if true, the simulation will be named "test"; otherwise, it will be named according to the inputs.
- `seedloc_point`: UTM coordinates of the particle seeding location.
- See the paragraph regarding `inputs_dorado_SDB.json` for the rest of the parameters.
