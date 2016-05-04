#!/usr/bin/env python

########################################################################
# Notes:
# - General configuration appears below in the "config" section
# - Model files, depths to plot, saturation levels, etc, supplied as
#   command-line arguments
########################################################################

import argparse
import os.path
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap

from Model1D import Model1D
from ModelA3d import ModelA3d
import pyspl
from UCBColorMaps import cmapSved, cmapXved

from FigTools import plot_plates, plot_hotspots

########################################################################
# const

# earth radius
r_earth = 6371.0

# spherical spline interpolant cache
sspl_cache_file = 'cache.pkl'


########################################################################
# config

# configuration arguments for figure windows (e.g. size in inches)
fig_config = {'figsize': (8,4)}

# basemap configuration
map_config = {
    'projection': 'robin',   # type of map projection
    'lon_0': 135.0,          # center longitude
    'resolution': 'c'}       # resolution of coastline data ('c'oarse fine for global)

# type of mean used with mean-removal turned on (command-line arg)
#mean_type = 'harmonic'
mean_type = 'arithmetic'

# compute and save model RMS at each depth
save_rms = True

# output-save config
## eps
save_ext = 'eps'
#save_kwargs = {}
## png
#save_ext = 'png'
save_kwargs = {'dpi': 150, 'transparent': True}


########################################################################
# support routines

def recenter_cmap(cmap, new_center, old_center=0.5):
    '''Recenters the matplotlib colormap `cmap` around `new_center` (assuming
    it was previously centered at kwarg `old_center`, which defaults to 0.5).
    '''
    segmentdata = cmap._segmentdata
    new_segmentdata = {}
    for color in segmentdata:
        new_levels = []
        for level in segmentdata[color]:
            x, above, below = level
            if x <= old_center:
                new_x = x * new_center / old_center
            else:
                new_x = new_center + (1.0 - new_center) * (x - old_center) / (
                        1.0 - old_center)
            new_levels.append((new_x, above, below))
        new_segmentdata[color] = tuple(new_levels)
    return LinearSegmentedColormap(cmap.name + '-recentered', new_segmentdata)


def get_interp(fname, dx = 1.0):
    """Build or load the spherical-spline interpolant for the specified
    knot-grid file.
    """
    cache = {}
    if os.path.exists(sspl_cache_file):
        with open(sspl_cache_file) as f:
            cache = pickle.load(f)
    if fname in cache:
        if cache[fname]['dx'] == dx:
            return cache[fname]['interp']
    grid = np.loadtxt(fname, skiprows=1)
    sspl = pyspl.SphericalSplines(grid[:,0], grid[:,1], grid[:,2])
    lons, lats = np.meshgrid(-180.0 + np.arange(0, 360.0 + 1e-5 * dx, dx),
                             -90.0  + np.arange(0, 180.0 + 1e-5 * dx, dx))
    H = sspl.evaluate(lons.ravel(), lats.ravel())
    cache[fname] = {'dx': dx, 'interp': (lons, lats, H)}
    with open(sspl_cache_file, 'w') as f:
        pickle.dump(cache, f)
    return lons, lats, H

def plot_model(modelfile, param, gridfile, rmean, recenter_xi,
        depths_and_levels, ref_model):
    """Plot the specified A3d file
    """
    # load the A3d model
    model = ModelA3d(modelfile)
    model.load_from_file()

    # fetch the spherical spline interpolant
    lons, lats, H = get_interp(gridfile)

    # extract model coefficients
    coefs = model.get_parameter_by_name(param).get_values()

    # build the radial b-spline interpolant
    bspl = pyspl.CubicBSplines(model.get_bspl_knots())
    r = np.asarray([r_earth - depth[0] for depth in depths_and_levels])
    V = bspl.evaluate(r)

    # sample the model (relative perturbations)
    x = H * (V * coefs).T

    # load the reference model, derive reference parameters
    ref = Model1D(ref_model)
    ref.load_from_file()
    vsv = ref.get_values(1000 * r, parameter='vsv')
    vsh = ref.get_values(1000 * r, parameter='vsh')
    vs0 = 0.001 * np.sqrt((2 * vsv ** 2 + vsh ** 2) / 3)
    xi0 = vsh ** 2 / vsv ** 2

    # xi is plotted with respect to isotropy
    if param == 'X':
        x = xi0 * (1.0 + x) - 1.0

    # rms strength at each depth
    weights = np.cos(np.radians(lats))
    weights = weights.reshape((weights.size,1))
    if save_rms:
        wrms = np.sqrt((weights * x ** 2).sum(axis = 0) / weights.sum())
        fname_rms = '%s.%s.rms' % (modelfile, param)
        with open(fname_rms, 'w') as f:
            for ix in range(len(depths_and_levels)):
                f.write('%f %f\n' % (depths_and_levels[ix][0], wrms[ix]))

    # mean removal (only valid for Vs)
    if rmean:
        assert param == 'S', 'Mean removal is only supported for Vs'
        vs = (1.0 + x) * vs0
        if mean_type == 'arithmetic':
            mean = (weights * vs).sum(axis = 0) / weights.sum()
        elif mean_type == 'harmonic':
            mean = weights.sum() / (weights / vs).sum(axis = 0)
        else:
            raise ValueError('Unrecognized mean type "%s"' % (mean_type))

    # loop over depths / saturation levels
    for ix in range(len(depths_and_levels)):
        # has a reference value been specified?
        if len(depths_and_levels[ix]) == 3:
            depth, level, x1 = depths_and_levels[ix]
            if param == 'S':
                vs = (x[:,ix] + 1.0) * vs0[ix]
                x[:,ix] = (vs - x1) / x1
            elif param == 'X':
                # reminder: xi shifted to isotropy
                # back to absolute
                xi = x[:,ix] + 1.0
                x[:,ix] = (xi - x1) / x1
            else:
                raise ValueError('Unrecognized parameter "%s"' % (param))
        else:
            depth, level = depths_and_levels[ix]
            # only remove mean if we have not used an alternative reference
            if rmean:
                vs = (x[:,ix] + 1.0) * vs0[ix]
                x[:,ix] = (vs - mean[ix]) / mean[ix]

        # to percent
        x[:,ix] *= 100

        # colormap / saturation
        if param == 'X':
            cmap, vmin = cmapXved(256, level)
            # cmapXved is asymmetric - should we re-center?
            if recenter_xi:
                old_center = 1.0 - level / (level - vmin)
                cmap = recenter_cmap(cmap, 0.5, old_center=old_center)
                vmin = - level
            vmax = level
        else:
            cmap = cmapSved(256)
            vmin = - level
            vmax = level

        # figure / map setup
        fig = plt.figure(**fig_config)
        ax = fig.add_axes([0,0,0.9,1.0])
        m = Basemap(ax=ax, **map_config)
        cp = m.drawmapboundary()
        m.drawcoastlines()
        print('%6.1f km : %+10.4f min / %+10.4f max' % (
            depth, x[:,ix].min(), x[:,ix].max()))

        # transform to map coords
        s = m.transform_scalar(x[:,ix].reshape(lons.shape),
                lons[0,:], lats[:,0], 1000, 500)

        # plot model, plates, hotspots
        im = m.imshow(s, vmin=vmin, vmax=vmax, cmap=cmap, clip_path=cp)
        plot_plates(m, linewidth=1, linestyle='-', color='k')
        plot_hotspots(m, linewidth=1, marker='o', linestyle='none',
                markerfacecolor=(0,1,0.05), markeredgecolor='k')

        # add a colorbar
        ## colorbar axis
        ax_cb = fig.add_axes([0.91,0.15,0.02,0.75])
        ## determine ticks
        if vmax <= 2:
            tick_step = 1.0
        else:
            tick_step = 2.0
        ticks_pos = np.arange(0, vmax + 1e-5 * tick_step,   tick_step).tolist()
        ticks_neg = np.arange(0, vmin - 1e-5 * tick_step, - tick_step).tolist()
        ticks = ticks_neg[-1:0:-1] + ticks_pos
        ## plot it and set up / format labels
        cb = plt.colorbar(im, cax=ax_cb, ticks=ticks, format='%3.0f')
        for l in cb.ax.yaxis.get_ticklabels():
            l.set_weight('bold')
            l.set_size(10)
        cb.set_label('%+.1f / %+.1f (%.0f km)' % (
            x[:,ix].min(), x[:,ix].max(), depth),
            rotation=270.0, va='bottom', weight='bold', size=14)

        # output file name / save
        saveparam = param
        if param == 'X' and recenter_xi:
            saveparam = param + '-centered'
        if rmean:
            fname_out = '%s_%s_%4.4ikm_rmean-%s.%s' % (
                    modelfile, saveparam, depth, mean_type, save_ext)
        else:
            fname_out = '%s_%s_%4.4ikm.%s' % (
                    modelfile, saveparam, depth, save_ext)
        plt.savefig(fname_out, **save_kwargs)
        #plt.show()
	plt.close()


def plot_all(modelfile, param, gridfile, rmean, recenter_xi,
        depths_and_levels, ref_model):
    # Plot all the depths of the A3d model on one plot
    ### Redo same steps as plotting each individually
    # load the A3d model
    model = ModelA3d(modelfile)
    model.load_from_file()

    # fetch the spherical spline interpolant
    lons, lats, H = get_interp(gridfile)

    # extract model coefficients
    coefs = model.get_parameter_by_name(param).get_values()

    # build the radial b-spline interpolant
    bspl = pyspl.CubicBSplines(model.get_bspl_knots())
    r = np.asarray([r_earth - depth[0] for depth in depths_and_levels])
    V = bspl.evaluate(r)

    # sample the model (relative perturbations)
    x = H * (V * coefs).T

    # load the reference model, derive reference parameters
    ref = Model1D(ref_model)
    ref.load_from_file()
    vsv = ref.get_values(1000 * r, parameter='vsv')
    vsh = ref.get_values(1000 * r, parameter='vsh')
    vs0 = 0.001 * np.sqrt((2 * vsv ** 2 + vsh ** 2) / 3)
    xi0 = vsh ** 2 / vsv ** 2

    # xi is plotted with respect to isotropy
    if param == 'X':
        x = xi0 * (1.0 + x) - 1.0

    # rms strength at each depth
    weights = np.cos(np.radians(lats))
    weights = weights.reshape((weights.size,1))
    if save_rms:
        wrms = np.sqrt((weights * x ** 2).sum(axis = 0) / weights.sum())
        fname_rms = '%s.%s.rms' % (modelfile, param)
        with open(fname_rms, 'w') as f:
            for ix in range(len(depths_and_levels)):
                f.write('%f %f\n' % (depths_and_levels[ix][0], wrms[ix]))

    # mean removal (only valid for Vs)
    if rmean:
        assert param == 'S', 'Mean removal is only supported for Vs'
        vs = (1.0 + x) * vs0
        if mean_type == 'arithmetic':
            mean = (weights * vs).sum(axis = 0) / weights.sum()
        elif mean_type == 'harmonic':
            mean = weights.sum() / (weights / vs).sum(axis = 0)
        else:
            raise ValueError('Unrecognized mean type "%s"' % (mean_type))

    # Make figure
    fig = plt.figure(figsize=(10,11))

    # loop over depths / saturation levels
    for ix in range(len(depths_and_levels)):
        # has a reference value been specified?
        if len(depths_and_levels[ix]) == 3:
            depth, level, x1 = depths_and_levels[ix]
            if param == 'S':
                vs = (x[:,ix] + 1.0) * vs0[ix]
                x[:,ix] = (vs - x1) / x1
            elif param == 'X':
                # reminder: xi shifted to isotropy
                # back to absolute
                xi = x[:,ix] + 1.0
                x[:,ix] = (xi - x1) / x1
            else:
                raise ValueError('Unrecognized parameter "%s"' % (param))
        else:
            depth, level = depths_and_levels[ix]
            # only remove mean if we have not used an alternative reference
            if rmean:
                vs = (x[:,ix] + 1.0) * vs0[ix]
                x[:,ix] = (vs - mean[ix]) / mean[ix]

        # to percent
        x[:,ix] *= 100

        # colormap / saturation
        if param == 'X':
            cmap, vmin = cmapXved(256, level)
            # cmapXved is asymmetric - should we re-center?
            if recenter_xi:
                old_center = 1.0 - level / (level - vmin)
                cmap = recenter_cmap(cmap, 0.5, old_center=old_center)
                vmin = - level
            vmax = level
        else:
            cmap = cmapSved(256)
            vmin = - level
            vmax = level

	### Plots on one figure rather than individually
       	# figure / map setup
        ax = plt.subplot(4,2,ix+1)
	ax.set_title('%+.1f / %+.1f (%.0f km)' % (x[:,ix].min(), x[:,ix].max(), depth), fontsize=16)
        m = Basemap(ax=ax, **map_config)
        cp = m.drawmapboundary()
        m.drawcoastlines()
        # transform to map coords
        s = m.transform_scalar(x[:,ix].reshape(lons.shape),lons[0,:], lats[:,0], 1000, 500)
	# plot model, plates, hotspots
        im = m.imshow(s, vmin=vmin, vmax=vmax, cmap=cmap, clip_path=cp)
        plot_plates(m, linewidth=.1, linestyle='-', color='k')
        plot_hotspots(m, linewidth=.001, marker='o', linestyle='none',
                markerfacecolor=(0,1,0.05), markeredgecolor='k')

    # add a colorbar
    # colorbar axis
    ax_cb = fig.add_axes([0.92,0.15,0.02,0.7])
    # determine ticks
    if vmax <= 2:
        tick_step = 1.0
    else:
        tick_step = 2.0
    ticks_pos = np.arange(0, vmax + 1e-5 * tick_step,   tick_step).tolist()
    ticks_neg = np.arange(0, vmin - 1e-5 * tick_step, - tick_step).tolist()
    ticks = ticks_neg[-1:0:-1] + ticks_pos
    # plot it and set up / format labels
    cb = plt.colorbar(im, cax=ax_cb, ticks=ticks, format='%3.0f')
    for l in cb.ax.yaxis.get_ticklabels():
        l.set_weight('bold')
        l.set_size(10)
    cb.set_label('dlnVs (%)')

    # output file name / save
    saveparam = param
    fname = modelfile.split("new_model.")
    plt.suptitle(fname[1], fontsize = 20)
    plt.subplots_adjust(top=0.92, bottom = 0, right=0.9, left=0.025, hspace=0.075, wspace = 0.02 )
    fname_out = '%s_%s_alldepth.pdf' % (modelfile, saveparam)
    plt.savefig(fname_out, **save_kwargs)
    plt.show()	

########################################################################
# main routine

def main():
    # setup command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str,
            help='A3d model file')
    parser.add_argument('--ref-model', required=True, type=str,
            help='reference model file')
    parser.add_argument('--grid', required=True, type=str,
            help='spherical spline grid file')
    parser.add_argument('--param', required=True, type=str,
            help='parameter name (i.e. A3d descriptor)')
    parser.add_argument('--depths', required=True, type=str,
            help='comma-separated list of depth:vmax or depth:vmax:refvalue')
    parser.add_argument('--rmean', action='store_true',
            help='remove global mean at each depth')
    parser.add_argument('--recenter-xi', action='store_true',
            help='recenter the xi colormap (default is SEMum asymmetric map')
    args = parser.parse_args()

    # parse depth / level spec
    depths_and_levels = [
        map(float,d.split(':')) for d in args.depths.split(',')]

    # plot the model
    plot_model(args.model, args.param, args.grid, args.rmean, args.recenter_xi,
        depths_and_levels, args.ref_model)

   # plot all models
    plot_all(args.model, args.param, args.grid, args.rmean, args.recenter_xi,
        depths_and_levels, args.ref_model)

if __name__ == '__main__':
    main()
