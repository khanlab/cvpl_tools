import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec
import scipy.stats


# reference: https://matplotlib.org/matplotblog/posts/create-ridgeplots-in-matplotlib/
def plot_kde(X: list[np.ndarray], xrange: tuple[float, float], ylimit: float):
    colors = ['#0000ff', '#3300cc', '#660099', '#990066', '#cc0033', '#ff0000']
    NPLOTS = len(X)

    gs = (grid_spec.GridSpec(NPLOTS, 1))
    fig = plt.figure(figsize=(8, 6))

    # creating empty list
    ax_objs = []

    for i in range(NPLOTS):
        # creating new axes object and appending to ax_objs
        ax_objs.append(fig.add_subplot(gs[i:i + 1, 0:]))

        xmin, xmax = xrange

        # grabbing x and y data from the kde plot
        x = xmin + np.arange(101) / np.array(100., dtype=np.float32) * (xmax - xmin)
        kde = scipy.stats.gaussian_kde(X[i])
        y = kde.evaluate(x)

        # filling the space beneath the distribution
        ax_objs[-1].fill_between(x, y, color=colors[i])

        # setting uniform x and y lims
        ax_objs[-1].set_xlim(xmin, xmax)
        ax_objs[-1].set_ylim(0, ylimit)

        i += 1

    plt.tight_layout()
    plt.show()
