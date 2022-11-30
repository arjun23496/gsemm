import sys

import open3d.cpu.pybind.io
import seaborn as seaborn
import matplotlib.pyplot as plt
import matplotlib

from episodic_memory.utils.all_imports import *
from decouple import config  # obtain environment files

from scipy.spatial import distance
from scipy.optimize import approx_fprime

import argparse
import os
import numpy as np
import time
import open3d as o3d

# sys.path.append('/media/arjun/Shared/research_projects/episodic_hopfield/python_modules')

from episodic_memory.experiment import temporal_pattern_from_output, find_best_chain_length, \
    pattern_changes_from_temporal_pattern
from episodic_memory.experiment import train_episodic_network
from episodic_memory.utils import normalize
from episodic_memory.networks.AdiabaticEpisodicExp import AdiabaticEpisodicExponential
from bokeh.models import Range1d
from episodic_memory.utils import add_experiment_id

from episodic_memory.plotting import quick_plot
from tqdm import tqdm

from sklearn.decomposition import PCA

import matplotlib.animation as animation
from mpl_toolkits import mplot3d
import networkx as nx
import matplotlib

# matplotlib.use("TkAgg")

parser = argparse.ArgumentParser(description='Demo for energy based networks.')

parser.add_argument('--alpha_c', type=float, default=4.9,
                    help="alpha c parameter")
parser.add_argument('--test_duration', type=float, default=1.5,
                    help="How long to test for")
parser.add_argument('--cue_id', type=int, default=0,
                    help="id of the cue that is set")

parser.add_argument('--hop_cue_id', type=int, default=0,
                    help="id of the cue for constant energy hopfield traversal")

parser.add_argument('--refresh', action="store_true",
                    help="whether to refresh the saved versions")

args = parser.parse_args()

# CONTOUR_RANGE = 1.2
# CONTOUR_STEP = 0.15

CONTOUR_RANGE = 1.2
CONTOUR_STEP = 0.10
N_PATTERNS = 3
N = 2
PATTERN_ORDER = [1, 2, 3]
ALPHA_C = args.alpha_c
ALPHA_S = 1.2
CUE_ID = args.cue_id

# create and initialize network and memories
mnet = AdiabaticEpisodicExponential()
mnet.initialize(seed=0,
                T_d=20,
                # alpha_s=1.2,
                alpha_s=ALPHA_S,
                alpha_c=ALPHA_C,
                beta_d=1.0,
                gamma=100,
                N=2,
                n_patterns=N_PATTERNS,
                approximation="range-kutta")

patterns = np.array([
    [1, 1],
    [1, -1],
    [-1, 1],
])
G = np.identity(N_PATTERNS)
G = np.roll(G, 1, axis=1)

mnet.xi = patterns.T
mnet.phi = G.T

# test network
mnet_test = copy.deepcopy(mnet)
simulated = []
energy_trajectory = []

# change this when working with static or dynamic energy surfaces

if args.alpha_c > 0.0:
    # cue = patterns[CUE_ID, :].astype('float')
    cue = np.array([-0.3, -0.8])
else:
    hop_patterns = np.array([
        [0.55, 0.55],
        [-0.7, -0.4],
        [-0.3, -0.8]
    ])
    cue = hop_patterns[args.hop_cue_id]

mnet_test.v = cue.flatten()

diag_freq = 5
for i in tqdm(range(int(args.test_duration * N_PATTERNS * 10000) // 6)):
    mnet_test.update()

    mnet_test.v = np.clip(mnet_test.v, -1, 1)
    if i < 100:
        per = diag_freq
    else:
        per = diag_freq

    if i % per == 0:
        model_output = mnet_test.v.reshape((-1, 1))
        model_output_vd = mnet_test.v_d.reshape((-1, 1))

        simulated.append((np.clip(model_output.flatten(), -1, 1), mnet_test.get_energy(),
                          np.clip(model_output_vd.flatten(), -1, 1),
                          mnet_test.f(mnet_test.h)))

## Step 4: Save plots of output
print("Computing energy surfaces")
energy_surfaces = []

array_size = int(2 * CONTOUR_RANGE / CONTOUR_STEP) + 3
energy_plot = None

energy_surface_directory = "energy_surface_{}_{}_{}".format(args.alpha_c,
                                                            args.test_duration,
                                                            args.cue_id)

if os.path.exists(energy_surface_directory) and not args.refresh:

    with open(os.path.join(energy_surface_directory, "x.pkl"), "rb") as f:
        x = pkl.load(f)

    with open(os.path.join(energy_surface_directory, "y.pkl"), "rb") as f:
        y = pkl.load(f)

    with open(os.path.join(energy_surface_directory, "energy_surfaces.pkl"), "rb") as f:
        energy_surfaces = pkl.load(f)

else:
    for time_id in tqdm(range(len(simulated))):
        energy_surface = np.zeros((array_size, array_size))
        x = np.zeros((array_size, array_size))
        y = np.zeros((array_size, array_size))

        for i, v1 in enumerate(np.arange(-CONTOUR_RANGE, CONTOUR_RANGE + 2 * CONTOUR_STEP, CONTOUR_STEP)):
            for j, v2 in enumerate(np.arange(-CONTOUR_RANGE, CONTOUR_RANGE + 2 * CONTOUR_STEP, CONTOUR_STEP)):
                x[i, j] = v1
                y[i, j] = v2
                energy_surface[i, j] = mnet_test.get_energy(np.array([v1, v2]), simulated[time_id][2])

        energy_surface = np.clip(energy_surface, np.min(energy_surface) + 0.035, None)

        energy_surfaces.append(energy_surface.copy())

    if not os.path.exists(energy_surface_directory):
        os.makedirs(energy_surface_directory)

    with open(os.path.join(energy_surface_directory, "x.pkl"), "wb") as f:
        pkl.dump(x, f)

    with open(os.path.join(energy_surface_directory, "y.pkl"), "wb") as f:
        pkl.dump(y, f)

    with open(os.path.join(energy_surface_directory, "energy_surfaces.pkl"), "wb") as f:
        pkl.dump(energy_surfaces, f)


################# Open3D section


######### Intialize energy point cloud data
energy_xyz = np.zeros((np.size(x), 3))
energy_xyz[:, 0] = np.reshape(x, -1)
energy_xyz[:, 2] = np.reshape(y, -1)
energy_xyz[:, 1] = np.reshape(energy_surfaces[0], -1)

normals = np.zeros((np.size(x), 3))
normals[:, 1] = 1

print("Intiailizing surface")
energy_meshes = []

meshes_directory = "energy_meshes_{}_{}_{}".format(args.alpha_c,
                                                   args.test_duration,
                                                   args.cue_id)

print("Computing triangles")
triangles_1 = []
triangles_2 = []
triangles_3 = []

triangle_normals_x = []
triangles_normals_y = []
triangles_normals_z = []
for i in tqdm(range(np.size(x) - array_size)):
    if i % array_size >= array_size-1:
        continue
    else:
        # triangle 1
        triangles_1.append(i)
        triangles_2.append(i+1)
        triangles_3.append(i + array_size+1)

        # triangle 2
        triangles_1.append(i)
        triangles_2.append(i + array_size+1)
        triangles_3.append(i + array_size)


energy_mesh_triangles = np.zeros((len(triangles_1), 3))
energy_mesh_triangles[:, 0] = np.array(triangles_1)
energy_mesh_triangles[:, 1] = np.array(triangles_2)
energy_mesh_triangles[:, 2] = np.array(triangles_3)

print("computing meshes")
if os.path.exists(meshes_directory) and not args.refresh:
# if False:
    for frame_number in tqdm(range(len(simulated))):
        mesh_aux = open3d.io.read_triangle_mesh(os.path.join(meshes_directory,
                                                             "energy_mesh_{}.ply".format(frame_number)))
        mesh_aux.compute_vertex_normals()
        energy_meshes.append(mesh_aux)
else:

    if not os.path.exists(meshes_directory):
        os.makedirs(meshes_directory)

    for frame_number in tqdm(range(len(simulated))):
        energy_xyz[:, 1] = np.reshape(energy_surfaces[frame_number], -1)

        alpha = 1.0

        # compute triangle normals
        triangle_normals = np.zeros(energy_mesh_triangles.shape)
        for triangle_index in range(energy_mesh_triangles.shape[0]):
            # print(energy_mesh_triangles[triangle_index, 0])
            vec1 = energy_xyz[int(energy_mesh_triangles[triangle_index, 0]), :]
            vec2 = energy_xyz[int(energy_mesh_triangles[triangle_index, 1]), :]
            vec3 = energy_xyz[int(energy_mesh_triangles[triangle_index, 2]), :]

            triangle_normals[triangle_index, :] = np.cross(vec2-vec1, vec3-vec1)

        # create mesh
        mesh_aux = o3d.geometry.TriangleMesh()
        mesh_aux.vertices = o3d.utility.Vector3dVector(energy_xyz)
        mesh_aux.triangles = o3d.utility.Vector3iVector(energy_mesh_triangles)
        mesh_aux.triangle_normals = o3d.utility.Vector3dVector(triangle_normals)
        mesh_aux.vertex_normals = o3d.utility.Vector3dVector(normals)

        # filter mesh
        mesh_aux = mesh_aux.filter_smooth_taubin(number_of_iterations=100)
        mesh_aux.compute_vertex_normals()

        energy_meshes.append(copy.deepcopy(mesh_aux))

        # save mesh in file
        open3d.cpu.pybind.io.write_triangle_mesh(os.path.join(meshes_directory,
                                                              "energy_mesh_{}.ply".format(frame_number)),
                                                 mesh_aux)

######### State position information
state_pos = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
state_pos.paint_uniform_color([0.75, 0.75, 0.75])
# state_pos.paint_uniform_color([0.5, 0.5, 0.5])
state_pos.compute_vertex_normals()

############  Visualizer section
# initialize visualizer
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()

# add geometry to visualization
mesh = copy.deepcopy(energy_meshes[0])
vis.add_geometry(mesh)
vis.add_geometry(state_pos)

ctr = vis.get_view_control()
parameters = o3d.io.read_pinhole_camera_parameters("pov_cam.json")
ctr.convert_from_pinhole_camera_parameters(parameters)

ANIMATION_PLAY = True


def pauseplay_animation(*args, **kwargs):
    global ANIMATION_PLAY
    ANIMATION_PLAY = not ANIMATION_PLAY


i = 0


def step_animation_forward(*args, **kwargs):
    global ANIMATION_PLAY, i
    if not ANIMATION_PLAY:
        i = (i + 1) % len(simulated)
        update_energy_mesh(i)
        update_state_pos(i)


def step_animation_backward(*args, **kwargs):
    global ANIMATION_PLAY, i
    if not ANIMATION_PLAY:
        i = (i - 1) % len(simulated)
        update_energy_mesh(i)
        update_state_pos(i)


def get_energy_surface_mesh(j):
    mesh.vertices = energy_meshes[j].vertices
    mesh.triangles = energy_meshes[j].triangles
    mesh.triangle_normals = energy_meshes[j].triangle_normals
    mesh.vertex_normals = energy_meshes[j].vertex_normals
    mesh.textures = energy_meshes[j].textures

    return mesh


def get_vertex_colors(emap):
    vertex_colors = np.zeros((emap.shape[0], 3))

    cmap = matplotlib.cm.get_cmap('jet')

    # max_offset = 1.6
    max_offset = 1.95
    e_min = np.min(emap)
    e_max = np.max(emap)

    for i in range(emap.shape[0]):
        vertex_colors[i, :] = np.array(cmap((np.clip(emap[i], e_min, e_max - max_offset) - e_min)
                                            /(e_max - max_offset - e_min)))[:3]

    return vertex_colors


def update_energy_mesh(j):
    mesh.vertices = energy_meshes[j].vertices
    mesh.triangles = energy_meshes[j].triangles
    mesh.triangle_normals = energy_meshes[j].triangle_normals
    mesh.vertex_normals = energy_meshes[j].vertex_normals
    mesh.textures = energy_meshes[j].textures
    mesh.vertex_colors = o3d.utility.Vector3dVector(get_vertex_colors(energy_surfaces[j].flatten()))

    vis.update_geometry(mesh)


def update_state_pos(j):
    global state_pos

    # update state position
    t = np.array([simulated[i][0][0], simulated[j][1] + 0.15, simulated[j][0][1]])
    state_pos = state_pos.translate(t - state_pos.get_center())
    state_pos.compute_vertex_normals()

    vis.update_geometry(state_pos)


# register callbacks
vis.register_key_callback(32, pauseplay_animation)
vis.register_key_callback(262, step_animation_forward)
vis.register_key_callback(263, step_animation_backward)

while True:
    if ANIMATION_PLAY:
        # update energy landscape

        update_energy_mesh(i)
        update_state_pos(i)

        i = (i + 1) % len(simulated)

    # update the rendering engine
    vis.poll_events()
    vis.update_renderer()
