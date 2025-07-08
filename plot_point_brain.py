import numpy as np
from typing import Iterable
import matplotlib.pyplot as plt

#adapted from https://netneurotools.readthedocs.io/en/latest/generated/netneurotools.plotting.plot_point_brain.html
#to add views and change linewidth
def plot_point_brain(data, coords, views=None, views_orientation='vertical',
                     views_size=(4, 2.4), cbar=False, robust=True, size=50,
                     **kwargs):
    """
    Plot `data` as a cloud of points in 3D space based on specified `coords`.

    Parameters
    ----------
    data : (N,) array_like
        Data for an `N` node parcellation; determines color of points
    coords : (N, 3) array_like
        x, y, z coordinates for `N` node parcellation
    views : list, optional
        List specifying which views to use. Can be any of {'sagittal',
        'coronal', 'axial'}. If not specified will use 'sagittal'
        and 'axial'. Default: None
    views_orientation: str, optional
        Orientation of the views. Can be either 'vertical' or 'horizontal'.
        Default: 'vertical'.
    views_size : tuple, optional
        Figure size of each view. Default: (4, 2.4)
    cbar : bool, optional
        Whether to also show colorbar. Default: False
    robust : bool, optional
        Whether to use robust calculation of `vmin` and `vmax` for color scale.
    size : int, optional
        Size of points on plot. Default: 50
    **kwargs
        Key-value pairs passed to `matplotlib.axes.Axis.scatter`

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
        Figure object for the plot
    axes : :class:`matplotlib.axes.Axes`
        Axes object for the plot
    """
    _views = dict(sagittal=(0, 180), sag1=(0, 180), sag2=(0, 0),
                  axial=(90, 180), ax1=(90, 180), ax2=(270, 180),
                  coronal=(0, 90), cor1=(0, 90), cor2=(0, 270))

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    if views is None:
        views = [_views[f] for f in ['sagittal', 'axial']]
    else:
        if not isinstance(views, Iterable) or isinstance(views, str):
            views = [views]
        views = [_views[f] for f in views]

    if views_orientation == 'vertical':
        ncols, nrows = 1, len(views)
    elif views_orientation == 'horizontal':
        ncols, nrows = len(views), 1
    figsize = (ncols * views_size[0], nrows * views_size[1])

    # create figure and axes (3d projections)
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows,
                             figsize=figsize,
                             subplot_kw=dict(projection='3d'))

    opts = dict(linewidth=1, edgecolor='gray', cmap='viridis')
    if robust:
        vmin, vmax = np.percentile(data, [2.5, 97.5])
        opts.update(dict(vmin=vmin, vmax=vmax))
    opts.update(kwargs)

    # iterate through saggital/axial views and plot, rotating as needed
    for n, view in enumerate(views):
        # if only one view then axes is not a list!
        ax = axes[n] if len(views) > 1 else axes
        # make the actual scatterplot and update the view / aspect ratios
        col = ax.scatter(x, y, z, c=data, s=size, **opts)
        ax.view_init(*view)
        ax.axis('off')
        scaling = np.array([ax.get_xlim(),
                            ax.get_ylim(),
                            ax.get_zlim()])
        ax.set_box_aspect(tuple(scaling[:, 1] - scaling[:, 0]))

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)

    # add colorbar to axes
    if cbar:
        cbar = fig.colorbar(col, ax=axes.flatten(),
                            drawedges=False, shrink=0.7)
        cbar.outline.set_linewidth(0)

    return fig, axes