######## IMPORTS ########
# General purpose imports
import os
import sys
import numpy as np

# Optimization specific imports
from lumopt import CONFIG
from lumopt.geometries.topology import TopologyOptimization3DLayered
from lumopt.utilities.load_lumerical_scripts import load_from_lsf
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.optimization import Optimization
from lumopt.optimizers.generic_optimizers import ScipyOptimizers
from lumopt.utilities.wavelengths import Wavelengths

######## DEFINE BASE SIMULATION ########

def runSim(params, eps_bg, eps_wg, x_pos, y_pos, z_pos, size_x, size_y, size_z, filter_R, working_dir, beta):

    ######## DEFINE GEOMETRY ########
    geometry = TopologyOptimization3DLayered(
        params=params,
        eps_min=eps_bg,
        eps_max=eps_wg,
        x=x_pos,
        y=y_pos,
        z=z_pos,
        filter_R=filter_R,
        beta=beta
    )

    ######## DEFINE FIGURE OF MERIT FOR EACH OUTPUT WAVEGUIDE ########
    fom3 = ModeMatch(monitor_name='fom_3', mode_number='Fundamental TE mode', direction='Forward', norm_p=2, target_fom=1)
    fom4 = ModeMatch(monitor_name='fom_4', mode_number='Fundamental TE mode', direction='Forward', norm_p=2, target_fom=1)

    ######## DEFINE OPTIMIZATION ALGORITHM ########
    optimizer = ScipyOptimizers(max_iter=400, method='L-BFGS-B', pgtol=1e-6, ftol=1e-5, scale_initial_gradient_to=0.25)

    ######## DEFINE SETUP SCRIPT AND INDIVIDUAL OPTIMIZERS ########
    script = load_from_lsf('CWDM_splitter_1310_4ch_3D_TE_topology.lsf')
    script = script.replace('opt_size_x=6e-6', 'opt_size_x={:1.6g}'.format(size_x))
    script = script.replace('opt_size_y=6e-6', 'opt_size_y={:1.6g}'.format(size_y))
    script = script.replace('opt_size_z=6e-6', 'opt_size_z={:1.6g}'.format(size_z))

    wavelengths3 = Wavelengths(start=1280e-9, stop=1290e-9, points=11)
    opt3 = Optimization(
        base_script=script,
        wavelengths=wavelengths3,
        fom=fom3,
        geometry=geometry,
        optimizer=optimizer,
        use_deps=False,
        hide_fdtd_cad=True,
        plot_history=False,
        store_all_simulations=False,
        save_global_index=False
    )
    wavelengths4 = Wavelengths(start=1265e-9, stop=1275e-9, points=11)
    opt4 = Optimization(
        base_script=script,
        wavelengths=wavelengths4,
        fom=fom4,
        geometry=geometry,
        optimizer=optimizer,
        use_deps=False,
        hide_fdtd_cad=True,
        plot_history=False,
        store_all_simulations=False,
        save_global_index=False
    )

    ######## PUT EVERYTHING TOGETHER AND RUN ########
    opt = opt3 + opt4
    opt.run(working_dir=working_dir)


if __name__ == '__main__':
    ######## SIMULATION SETTINGS ########
    size_x = 6000          # Length of design region in x (nm)
    size_y = 6000          # Length of design region in y (nm)
    size_z = 220           # Thickness of design region in z (nm)
    filter_R = 200e-9      # Filter radius (m)

    ######## DEFINE GRID ########
    x_points = int(size_x / 20) + 1
    y_points = int(size_y / 20) + 1
    z_points = int(size_z / 20) + 1  # Higher resolution in z for accuracy

    eps_wg = 3.45**2       # Waveguide permittivity (silicon)
    eps_bg = 1.44**2       # Background permittivity (SiO2)
    x_pos = np.linspace(-size_x / 2 * 1e-9, size_x / 2 * 1e-9, x_points)  # x-coordinates
    y_pos = np.linspace(-size_y / 2 * 1e-9, size_y / 2 * 1e-9, y_points)  # y-coordinates
    z_pos = np.linspace(0, size_z * 1e-9, z_points)                      # z-coordinates

    ######## INITIAL PARAMETERS ########
    # prev_filename = 'C:/Users/William/Desktop/results_master/wdm/wdm/cwds_20241124/CWDM_y_1310_4ch_2D_TE_x6000_y6000_f0200_5/parameters_624.npz'  #< Load previous optimization state
    # prev_geom = TopologyOptimization3DLayered.from_file(prev_filename, filter_R=filter_R)
    # params = prev_geom.last_params
    # beta = prev_geom.beta  # Use the structure defined in the project file as the initial condition
    params = 0.5*np.ones((x_points,y_points))     #< Start with the domain filled with (eps_wg+eps_bg)/2
    beta = 1
    ######## RUN OPTIMIZATION ########
    working_dir = 'CWDM_y_1310_4ch_3D_TE_x{:04d}_y{:04d}_z{:04d}_f{:04d}'.format(size_x, size_y, size_z, int(filter_R * 1e9))
    runSim(params, eps_bg, eps_wg, x_pos, y_pos, z_pos, size_x * 1e-9, size_y * 1e-9, size_z * 1e-9, filter_R, working_dir=working_dir, beta=beta)
