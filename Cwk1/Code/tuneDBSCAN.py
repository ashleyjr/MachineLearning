import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import pylab as pl
pl.ion()

##############################################################################
# Load data
f = open('Datasets/Aggregation.txt','r')
X = []
labels_true = []
for line in f:
	x = line.split('\t')[0]
	y = line.split('\t')[1]
	label = line.split('\t')[0]
	label = label.replace('\n','')
	X.append([x,y])
	labels_true.append(label)
X = StandardScaler().fit_transform(X)
##############################################################################
# DBSCAN Parameters
eps = 0.47
min_samples = 1
score = []
para = []
for test in range(1,480):
	eps = float(test)/float(800)
	##############################################################################
	# Compute DBSCAN
	db = DBSCAN(eps, min_samples).fit(X)
	core_samples = db.core_sample_indices_
	labels = db.labels_
	# Number of clusters in labels, ignoring noise if present.
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	score.append(metrics.adjusted_mutual_info_score(labels_true, labels))
	para.append(eps)
	if(score[test-1] < 0):
		score[test-1] = 0;
	score[test-1] = 1 - (score[test-1]*10)
	print("%0.3f: Score: %0.3f" % (eps, score[test-1]))
pl.plot(para,score, '.', markeredgecolor='k', markersize=10)
#coefs = np.lib.polyfit(para, score, 4) #4
#fit_y = np.lib.polyval(coefs, para) #5
#pl.plot(para, fit_y, 'b--') #6
pl.ylim([0,1])
pl.grid()
pl.xlabel('EPSILON')
pl.ylabel('Generalisation Error')
raw_input("Press Enter to continue...")

