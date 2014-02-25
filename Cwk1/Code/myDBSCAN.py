print(__doc__)

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

import pylab as pl
pl.ion()



##############################################################################
# Generate sample data
#centers = [[10, 10], [-5, -5], [1, -1]]
#X, labels_true = make_blobs(n_samples=1000, centers=centers, cluster_std=0.4,
#                            random_state=0)
#
f = open('Datasets/Jain.txt','r')
X = []
labels_true = []
for line in f:
	x = line.split('\t')[0]
	y = line.split('\t')[1]
	label = line.split('\t')[0]
	label = label.replace('\n','')
	X.append([x,y])
	labels_true.append(label)
#print X
#print labels_true
X = StandardScaler().fit_transform(X)


fig = 0;
for preTest in range(2,6):
	test = float(preTest)/float(20)
	print("Testing with eps=%0.3f" % test)
	##############################################################################
	# Compute DBSCAN
	db = DBSCAN(test, min_samples=1).fit(X)
	core_samples = db.core_sample_indices_
	labels = db.labels_

	# Number of clusters in labels, ignoring noise if present.
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

	print('Estimated number of clusters: %d' % n_clusters_)
	print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
	print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
	print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
	print("Adjusted Rand Index: %0.3f"
	      % metrics.adjusted_rand_score(labels_true, labels))
	print("Adjusted Mutual Information: %0.3f"
	      % metrics.adjusted_mutual_info_score(labels_true, labels))
	print("Silhouette Coefficient: %0.3f"
	      % metrics.silhouette_score(X, labels))

	##############################################################################
	# Plot result
	# Black removed and is used for noise instead.
	fig = fig + 1
	pl.figure(fig)

	unique_labels = set(labels)
	colors = pl.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
	for k, col in zip(unique_labels, colors):
	    if k == -1:
	        # Black used for noise.
	        col = 'k'
	        markersize = 6
	    class_members = [index[0] for index in np.argwhere(labels == k)]
	    cluster_core_samples = [index for index in core_samples
	                            if labels[index] == k]
	    for index in class_members:
	        x = X[index]
	        if index in core_samples and k != -1:
	            markersize = 14
	        else:
	            markersize = 6
	        pl.plot(x[0], x[1], 'o', markerfacecolor=col,
	                markeredgecolor='k', markersize=markersize)
	pl.title('Estimated number of clusters: %d' % n_clusters_)
	pl.show()
raw_input("Press Enter to continue...")

