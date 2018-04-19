from sklearn.cluster.k_means_ import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd
from multiprocessing import Pool
import re
import readline


def print_progress_bar (iteration, total, prefix='', suffix='', decimals=2, length=50, fill='█'):
    '''
    Auxillary function. Gives us a progress bar which tracks the completion status of our task. Put in loop.
    :param iteration: current iteration
    :param total: total number of iterations
    :param prefix: string
    :param suffix: string
    :param decimals: float point precision of % done number
    :param length: length of bar
    :param fill: fill of bar
    :return:
    '''
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def simulation(n, n_clusters, k_range, dim, runs=100):
    all_data = []
    k_low, k_hi = k_range
    for idx in range(runs):
        data, labels = make_blobs(n_samples=n, n_features=dim, centers=n_clusters,
                                  cluster_std=0.1, center_box=(-1.0, 1.0))

        for k in range(k_low, k_hi+1):
            # Get a model specified, fit to data, score for error, mark error as -1 if fails
            model = KMeans(n_clusters=k, random_state=0)
            labels = model.fit_predict(data)
            avg_score = silhouette_score(data, labels)
            all_data.append([n, n_clusters, k, dim, avg_score])

    df = pd.DataFrame(all_data, columns=['n', 'n_clusters', 'k', 'dim', 'avg_score'])
    return df


def run_sim(n, n_clusters, k_range, dim_range, runs, file):
    dim_min, dim_max = dim_range

    tasks = []
    total = 0
    for dim in np.arange(dim_min, dim_max + 1):
        tasks.append((n, n_clusters, k_range, dim, runs,))
        total += 1

    # Progress bar stuff
    iteration = 0
    prefix = "Simulation"
    suffix = "Complete"
    print_progress_bar(iteration, total, prefix=prefix, suffix=suffix)

    # Send our tasks to the process pool, as they complete append their results to data
    data = []
    with Pool(processes=3) as pool:
        results = [pool.apply_async(simulation, args=t) for t in tasks]
        for r in results:
            iteration += 1
            data.append(r.get())
            print_progress_bar(iteration, total, prefix=prefix, suffix=suffix)

    print("Writing data...")
    df = pd.concat(data)
    df.to_csv("data/{}".format(file), sep=',', index=False)


if __name__ == "__main__":

    userin = input("Give value for n\n"
                   ">>>> ")
    n = int(userin)

    userin = input("Give a fixed number of actual data clusters\n"
                 ">>> ")
    n_clusters = int(userin)

    userin = input("Give a range for k values (comma separated)\n"
                   ">>>> ")
    splt = re.split(",", userin)
    vals = [int(x) for x in splt]
    k_range = (vals[0], vals[1])

    userin = input("Give a range for dimensions (comma separated)\n"
                   ">>>> ")
    splt = re.split(",", userin)
    vals = [int(x) for x in splt]
    dim = (vals[0], vals[1])

    userin = input("Give numeber of runs\n"
                   ">>>> ")
    runs = int(userin)

    userin = input("Enter a file name for save\n"
                   ">>>> ")
    file = userin

    run_sim(n, n_clusters, k_range, dim, runs, file)