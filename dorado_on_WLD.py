import gc  # garbage collector, used to delete a big variable to save RAM
from dorado_atx import *

# I/O paths for dorado simulation
DORADO_CONF_FILENAME = "inputs_dorado_WLD.json"  # relative path to configuration file

# Retrieve path of current file
dir_path = os.path.dirname(os.path.realpath(__file__))

# Load inputs from json file as a dictionary
with open(os.path.join(dir_path, DORADO_CONF_FILENAME)) as f:
    in_d = json.load(f)

# Retrieve path of transects coordinates file and seedloc coordinates
transects_coord_path = os.path.join(dir_path, in_d["transects_filename"])

# Create simulation name
if in_d["test_mode"]:
    sim_name = "test"
else:
    sim_name = (
        f"Np{in_d['Np']}_theta{in_d['theta']:.1f}_nstep{in_d['n_step']}"
        + f"_dt{in_d['dt']}_dx{in_d['dx']:.1f}_{in_d['seedloc_point']}"
    )

# ------------------------------------ I/O ----------------------------------- #
anuga_file_path = os.path.join(in_d["path_anuga_sims"], in_d["anuga_file_name"])
my_swwvals = import_ANUGA_flow_field(
    anuga_file_path,
    in_d["dx"],
    UTM_region="WLD",
    bounds_path=os.path.join(dir_path, "WLAD_Boundary.csv"),
    t_idx=in_d["anuga_timestep"],
)

# Initialize folder for simulation
sim_folder_path, figs_temp_path = create_sim_folder(
    in_d,
    in_d["path_output_dorado_sims_dir"],
    sim_name,
    sim_par_dir_name=in_d["anuga_file_name"],
)

# ----------------------------- dorado simulation ---------------------------- #
# Retrieve the coordinates of the seeding point
seedinds = pt.coord2ind(
    [tuple(in_d["seedloc_point"])],
    (min(my_swwvals["x"]), min(my_swwvals["y"])),
    np.shape(my_swwvals["depth"]),
    in_d["dx"],
)

# Create and route the particles
params, walk_data = run_dorado_simulation(
    in_d, my_swwvals, seedinds, sim_folder_path, figs_temp_path
)

# Create the gif for this group of dorado simulations
create_gif(figs_temp_path, os.path.dirname(sim_folder_path))
clear_temp_folder(figs_temp_path)

# Compute sediment partitioning at bifurcations
compute_DeltaQS(
    transects_coord_path, walk_data, in_d["dx"], my_swwvals, sim_folder_path
)

# Delete the big dictionary containing the travel paths of all particles
del walk_data
gc.collect()
