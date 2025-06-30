"""Script to run a dorado simulation on the flow field returned by an ANUGA simulation."""

# %%
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import dorado_atx as atx
import dorado.particle_track as pt

current_path = os.path.dirname(os.path.realpath(__file__))
plt.style.use(os.path.join(current_path, "science.mplstyle"))

# Import configuration file in a dictionary
with open(os.path.join(current_path, "inputs_dorado_SDB.json")) as f:
    in_d = json.load(f)

# Create simulation name based on input parameters
dorado_sim_name = f"Np{in_d['Np']}_theta{in_d['theta']:.1f}_nstep{in_d['n_step']}_dt{in_d['dt']}_dx{in_d['dx']:.1f}"
if in_d["seedloc_length"] != 0:
    dorado_sim_name += f"_seedloc{in_d['seedloc_length']}"

# Retrieve path to parent folder of the anuga simulation
anuga_sim_name = in_d["anuga_sim_name"]
anuga_sims_dir_path = os.path.join(
    current_path, in_d["anuga_sims_dir_name"], anuga_sim_name
)

# Retrieve path to coordinates subfolder
coords_path = os.path.join(anuga_sims_dir_path, "coords")

# Retrieve path to the json configuration file of the anuga simulation
# and import it as a dictionary
anuga_inputs_path = os.path.join(anuga_sims_dir_path, "inputs.json")
with open(anuga_inputs_path, "r") as f:
    anuga_ind = json.load(f)

# Create directory for the dorado simulation
try:
    dorado_sim_dir_path, fig_temp_path = atx.create_sim_folder(
        in_d,
        os.path.join(anuga_sims_dir_path, "dorado_simulations"),
        dorado_sim_name,
        datetime_sim_dir=False,
        make_fig_subdir=True,
        exist_ok=in_d["overwrite_dorado_sim"],
    )
except OSError:
    print(
        f"The dorado simulation {dorado_sim_name} has already been performed on the {anuga_sim_name} simulation "
        + f"of the bifurcation {anuga_sim_name}. To allow overwriting, enable the overwrite flag."
    )

my_swwvals = atx.import_ANUGA_flow_field(
    os.path.join(anuga_sims_dir_path, f"swwvals.sww"),
    in_d["dx"],
    bounds_path=os.path.join(coords_path, "bounds.csv"),
)
# %%
seedinds = pt.coord2ind(
    [tuple(np.loadtxt(os.path.join(coords_path, "seed_point.csv")))],
    (min(my_swwvals["x"]), min(my_swwvals["y"])),
    np.shape(my_swwvals["depth"]),
    in_d["dx"],
)
# %%
# Modify seedind longitudinal coordinate (y coordinate in dorado's index reference system)
# according to the input parameter seedloc_length
if in_d["seedloc_length"] > 0:
    L_backw = anuga_ind["RefFlow"]["D_U"] / anuga_ind["S_U"]
    seed_y = (anuga_ind["L_U"] * L_backw * in_d["seedloc_length"]) / in_d["dx"]
    seedinds = [(seedinds[0][0], seed_y)]

# Run dorado simulation, generate gif and compute fluxes
params, walk_data = atx.run_dorado_simulation(
    in_d, my_swwvals, seedinds, dorado_sim_dir_path, fig_temp_path, area_xylims="SDB"
)
atx.create_gif(fig_temp_path, dorado_sim_dir_path, overwrite_gif=True)
atx.clear_temp_folder(fig_temp_path)

transects_coord_file_path = os.path.join(coords_path, in_d["transects_filename"])
atx.compute_DeltaQS(
    transects_coord_file_path, walk_data, in_d["dx"], my_swwvals, dorado_sim_dir_path
)
