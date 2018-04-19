import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import readline

def plot_error(file):
    xvar = 'dim'
    yvar = 'k'
    zvar = 'avg_score'

    title = 'K-Means (DIMESNSIONALITY vs. K, vs. SILHOUETTE SCORES)'

    df = pd.read_csv("data/{}".format(file))
    means_df = df.groupby([xvar, yvar], as_index=False)[zvar].mean()
    sd_df = df.groupby([xvar, yvar], as_index=False)[zvar].std()

    xx = means_df.as_matrix([xvar])
    yy = means_df.as_matrix([yvar])
    zz = means_df.as_matrix([zvar])
    z_sd = sd_df.as_matrix([zvar])

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    plt.title(title)
    ax.scatter(xx, yy, zz)

    for i in np.arange(0, len(z_sd)):
        ax.scatter([xx[i], xx[i]], [yy[i], yy[i]], [zz[i] + z_sd[i], zz[i] - z_sd[i]],
                   marker="_", c='y')

    ax.set_xlabel(xvar)
    ax.set_ylabel(yvar)
    ax.set_zlabel(zvar)

    plt.show()

if __name__ == "__main__":
    file = input("Enter a file name\n"
                 ">>>> ")
    plot_error(file)