from __future__ import division
import collections
import cPickle
import itertools
import json
import logging
import math
import os
import random
import scipy.stats

import processdata

base_method_names = ['DR-%s' % x for x in ['weka.classifiers.bayes.AODE', 'weka.classifiers.bayes.BayesNet', 'weka.classifiers.bayes.NaiveBayes', 'weka.classifiers.bayes.NaiveBayesSimple', 'weka.classifiers.functions.MultilayerPerceptron', 'weka.classifiers.functions.SMO', 'weka.classifiers.functions.VotedPerceptron', 'weka.classifiers.functions.Winnow', 'weka.classifiers.lazy.IB1', 'weka.classifiers.lazy.IBk', 'weka.classifiers.lazy.KStar', 'weka.classifiers.lazy.LBR', 'weka.classifiers.meta.AdaBoostM1', 'weka.classifiers.meta.LogitBoost', 'weka.classifiers.misc.HyperPipes', 'weka.classifiers.rules.JRip', 'weka.classifiers.rules.OneR', 'weka.classifiers.rules.Prism', 'weka.classifiers.rules.ZeroR', 'weka.classifiers.trees.DecisionStump', 'weka.classifiers.trees.Id3', 'weka.classifiers.trees.J48', 'weka.classifiers.trees.LMT', 'weka.classifiers.trees.RandomForest', 'weka.classifiers.trees.RandomTree']]
base_method_names.extend(['UpdateAllDelayed', 'DelayedRetrainer[C=0.05,num_methods=25]', 'DR-weka.classifiers.meta.Stacking'])
all_method_names = []
all_method_names.extend('%s-7200s' % x for x in base_method_names)
all_method_names.extend('%s-6300s' % x for x in base_method_names)
all_method_names.extend('%s-5400s' % x for x in base_method_names)
all_method_names.extend('%s-4500s' % x for x in base_method_names)
all_method_names.extend('%s-3600s' % x for x in base_method_names)
all_method_names.extend('%s-3000s' % x for x in base_method_names)
all_method_names.extend('%s-2400s' % x for x in base_method_names)
all_method_names.extend('%s-1800s' % x for x in base_method_names)
all_method_names.extend('%s-1200s' % x for x in base_method_names)
all_method_names.extend('%s-900s' % x for x in base_method_names)
all_method_names.extend('%s-600s' % x for x in base_method_names)
all_method_names.extend('%s-450s' % x for x in base_method_names)
all_method_names.extend('%s-300s' % x for x in base_method_names)
all_method_names.extend('%s-180s' % x for x in base_method_names)
all_method_names.extend('%s-60s' % x for x in base_method_names)
#all_method_names = []
#all_method_names.extend('%s-3600s' % x for x in base_method_names)

def avg(xs):
	xs = list(xs)
	return sum(xs) / len(xs)

recordall = collections.defaultdict(list)

timespentdata = {}

if __name__ == '__main__':
	for l in open("results/recorded-delayed-training.txt"):
		js = json.loads(l)
		sid = "%s#%s" % (js["algorithm"], js["dataset"])
		recordall[sid] = [js["result"]]
		timespentdata[sid] = js["time_spent"]

perdataset = collections.defaultdict(dict)
permethod = collections.defaultdict(dict)

bestperdataset = collections.defaultdict(float)

for m in all_method_names:
	for dsn in processdata.all_dataset_names:
		ds_best = []
		for seed in [0, 1, 1000, 1001]:
#			if seed > 1000 and 'Stacking' in m: continue
			p = ("%s#%s/%s" % (m, dsn, seed))
			if 'Stacking' in m and '-3600s' in m: print seed, recordall.get(p), m
			best = avg(recordall.get(p, [0]))
			ds_best.append(best)
		perdataset[dsn][m] = ds_best
		permethod[m][dsn] = ds_best
#		permethod[m][dsn] = [avg(ds_best)]

records = collections.defaultdict(list)

methods = []
for m in all_method_names:
	if '-3600s' not in m: continue
	methods.append(m)


for dsn in processdata.all_dataset_names:
	for m in methods:
		v = permethod[m][dsn]
		records[m].extend(v)

beatenscores = []

beatlookup = collections.defaultdict(list)

for m in methods:
	beaten = []
	unbeaten = []
	for t in methods:
		if t == m: continue
#		if 'UpdateAll' in t: continue
#		if 'DelayedRetrainer' in t: continue
#		if 'Stacking' in t: continue
		if scipy.stats.wilcoxon(records[m], records[t])[1] < 0.05:
			if scipy.stats.ttest_rel(records[m], records[t])[0] > 0:
				beaten.append(t)
				beatlookup[t].append(m)
				continue
		unbeaten.append(t)
	beatenscores.append((m, beaten, unbeaten))

beatenscores.sort(key=lambda (x,y,z): (-len(y), x))

for (x, y, z) in beatenscores:
	if len(z) <= 5:
		print x, len(y), z
	else:
		print x, len(y)
	if len(beatlookup[x]) < 5:
		print "  ", beatlookup[x]
		print

for (x, y, z) in beatenscores:
	win = len(y)
	lose = len(beatlookup[x])
	neither = 27 - win - lose
	print "  ", "+" * win + "0" * neither + "-" * lose, x

if False:
	print "------ ALMA vs RF -----"

	print "k-s test", scipy.stats.ks_2samp(alma, rf)

	print "t-test", scipy.stats.ttest_rel(alma, rf)
	print "t-test", scipy.stats.ttest_rel(rf, alma)

	print "wilcoxon", scipy.stats.wilcoxon(alma, rf)


#for i in range(4):
#	scores = []
#	for b in booty:
#		scores.append(b[0][i] - b[1][i])
#	print avg(scores)
