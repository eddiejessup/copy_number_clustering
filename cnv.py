from __future__ import print_function, division
import numpy as np
import scipy.stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
from ciabatta import ejm_rcparams

ejm_rcparams.set_pretty_plots(use_latex=False, use_pgf=True)


def get_data():
    d = np.loadtxt('exercise.txt', delimiter=',').T
    inds = np.arange(d.shape[1])
    return inds, d


def remove_single_point_spikes(d):
    d_filter = d.copy()
    for i in range(1, d.shape[0] - 1):
        if (d[i] != d[i - 1]) and (d[i - 1] == d[i + 1]):
            d_filter[i] = d[i - 1]
    return d_filter


def denoise_data(d):
    d_denoise = d.copy()
    for sample in d_denoise:
        sample[:] = remove_single_point_spikes(sample)
    return d_denoise


def reduce_data(inds, d):
    inds_reduced, d_reduced = [], []
    for i_loc_1 in range(d.shape[1]):
        for i_loc_2 in range(i_loc_1 + 1, d.shape[1]):
            if np.allclose(d[:, i_loc_1], d[:, i_loc_2]):
                break
        else:
            d_reduced.append(d[:, i_loc_1])
            inds_reduced.append(inds[i_loc_1])
    return np.array(inds_reduced), np.array(d_reduced).T


def get_cluster_number_metrics(d):
    ks, silhouettes, variances = [], [], []
    for k in range(2, 11):
        ss_k, vs_k = [], []
        for _ in range(5):
            fit = KMeans(n_clusters=k, random_state=_).fit(d)
            labels = fit.labels_

            ss_k.append(sklearn.metrics.silhouette_score(d, labels))

            variance = 0.0
            for label in labels:
                d_label = d[labels == label]
                variance += np.var(d_label)
            vs_k.append(variance)
        ks.append(k)
        variances.append(np.mean(vs_k))
        silhouettes.append(np.mean(ss_k))
    return ks, silhouettes, variances


def fit_data(d, k):
    return KMeans(n_clusters=k, random_state=1).fit(d)


def get_average_samples(d, labels):
    d_label_means = []
    for label in set(labels):
        d_label = d[labels == label]
        d_label_mean = scipy.stats.mode(d_label, axis=0)[0][0, :]
        d_label_means.append(d_label_mean)
    return np.array(d_label_means)


def signature_sample(inds, sample):
    baseline = scipy.stats.mode(sample)[0][0]
    s_norm = sample - baseline
    s_norm_diff = np.diff(s_norm)
    i_shifts_raw = np.nonzero(s_norm_diff)[0]
    shifts = [(i, s_norm_diff[i]) for i in i_shifts_raw]
    return baseline, shifts


def sort_data_by_label(d, labels):
    i_samples_sort = sorted(range(len(d)), key=lambda i: labels[i])
    return d[i_samples_sort], labels[i_samples_sort]


def plot_av_samples(inds, d_avs):
    c = ejm_rcparams.set2(range(len(d_avs)))
    fig = plt.figure()
    for i, d_av in enumerate(d_avs):
        ax = fig.add_subplot(len(d_avs), 1, i + 1)
        ejm_rcparams.prettify_axes(ax)
        ax.plot(d_av, c=c[i])
        if i < len(d_avs) - 1:
            ax.set_xticks([])
        ax.set_yticks([1, 2, 3, 4])
        ax.set_ylim(0.5, 4.5)
    plt.show()


def plot_elbow():
    fig = plt.figure()
    fig.frameon = False
    ax = fig.gca()
    ejm_rcparams.prettify_axes(ax)
    inds, d = get_data()
    d = denoise_data(d)
    inds, d = reduce_data(inds, d)
    d = denoise_data(d)
    inds, d = reduce_data(inds, d)
    ks, ss, vs = get_cluster_number_metrics(d)
    ax.plot(ks[:-1], np.diff(vs))
    ax.set_xlabel('Number of clusters', fontsize=24)
    ax.set_ylabel('Change in average in-cluster variance', fontsize=24)
    ax.tick_params(axis='x', labelsize=24)
    ax.set_yticks([])
    plt.savefig('elbow.png', bbox_inches='tight', transparency=True)


def plot():
    red_blue_zero = ejm_rcparams.shifted_cmap(ejm_rcparams.red_blue,
                                              midpoint=0.4)
    fig = plt.figure()
    fig.frameon = False
    ax = fig.gca()
    ejm_rcparams.prettify_axes(ax)
    ax.axis('off')

    inds, d = get_data()
    ax.imshow(d - 2, interpolation='nearest', cmap=red_blue_zero)
    ax.axis('off')
    plt.savefig('original.png', bbox_inches='tight', transparency=True)

    ax.cla()
    d = denoise_data(d)
    ax.imshow(d - 2, interpolation='nearest', cmap=red_blue_zero)
    ax.axis('off')
    plt.savefig('denoise.png', bbox_inches='tight', transparency=True)

    ax.cla()
    inds, d = reduce_data(inds, d)
    d = denoise_data(d)
    inds, d = reduce_data(inds, d)
    ax.imshow(d - 2, interpolation='nearest', cmap=red_blue_zero)
    ax.axis('off')
    plt.savefig('reduced.png', bbox_inches='tight', transparency=True)

    ax.cla()
    fit = fit_data(d, k=5)
    labels = fit.labels_
    d_sort, labels_sort = sort_data_by_label(d, labels)
    ax.imshow(d_sort - 2, interpolation='nearest', cmap=red_blue_zero)
    ax.hlines(np.nonzero(np.diff(labels_sort))[0], 0.0, d_sort.shape[1],
              colors=ejm_rcparams.almost_black)
    ax.axis('off')
    plt.savefig('clustered.png', bbox_inches='tight', transparency=True)

    d_avs = get_average_samples(d, labels)
    fig = plt.figure()
    for i, d_av in enumerate(d_avs):
        ax = fig.add_subplot(len(d_avs), 1, i + 1)
        ejm_rcparams.prettify_axes(ax)
        ax.plot(d_av, c=ejm_rcparams.set2[i])
        ax.set_xticks([])
        ax.set_yticks([1, 2, 3, 4])
        ax.set_ylim(0.5, 4.5)
        ax.tick_params(axis='y', labelsize=20)
        sig = signature_sample(inds, d_av)
        print(sig)
    plt.savefig('signatures.png', bbox_inches='tight')
