from __future__ import division
import collections
import cPickle
import itertools
import json
import logging
import math
import os
import random

import processdata

all_method_names = ['weka.classifiers.bayes.AODE', 'weka.classifiers.bayes.BayesNet', 'weka.classifiers.bayes.NaiveBayes', 'weka.classifiers.bayes.NaiveBayesSimple', 'weka.classifiers.functions.MultilayerPerceptron', 'weka.classifiers.functions.SMO', 'weka.classifiers.functions.VotedPerceptron', 'weka.classifiers.functions.Winnow', 'weka.classifiers.lazy.IB1', 'weka.classifiers.lazy.IBk', 'weka.classifiers.lazy.KStar', 'weka.classifiers.lazy.LBR', 'weka.classifiers.meta.AdaBoostM1', 'weka.classifiers.meta.LogitBoost', 'weka.classifiers.misc.HyperPipes', 'weka.classifiers.rules.JRip', 'weka.classifiers.rules.OneR', 'weka.classifiers.rules.Prism', 'weka.classifiers.rules.ZeroR', 'weka.classifiers.trees.DecisionStump', 'weka.classifiers.trees.Id3', 'weka.classifiers.trees.J48', 'weka.classifiers.trees.LMT', 'weka.classifiers.trees.RandomForest', 'weka.classifiers.trees.RandomTree']
#all_method_names += ['UpdateAll']
all_method_names += ['UCTSmartRetrainer[0.5]']
all_method_names += ['UCTSmartRetrainer[0.1]']
all_method_names += ['UCTSmartRetrainer[1.0]']
all_method_names += ['UCTSmartRetrainer[0.25]']
all_timelimits_seconds = [10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 600]

def avg(xs): return sum(xs) / len(xs)

recordall = collections.defaultdict(list)

foo = {u'time_spent': 2061.2429999999995,
u'result': 0.9977843426883308,
u'algorithm': u'UpdateAll-500s',
u'dataset': u'agaricus-lepiota.data/1'}

timespentdata = {}

if __name__ == '__main__':
	for l in open("results/recorded.txt"):
		js = json.loads(l)
		if js.get("time_spent", 1000000) > 700: continue
		if "UpdateAll" in js["algorithm"]:
			continue
			sid = "%s#%s" % (js["algorithm"], js["dataset"])
			recordall[sid].append(js["result"])
			timespentdata[sid] = js["time_spent"]
		elif '-600s' in js["algorithm"]:
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
			best = 0
			for limit in all_timelimits_seconds:
				p = ("%s-%ss#%s/%s" % (m, limit, dsn, seed))
				if p in recordall:
					best = avg(recordall[p])
			ds_best.append(best)
		score = avg(ds_best)
		bestperdataset[dsn] = max(bestperdataset[dsn], score)
		perdataset[dsn][m] = score
		permethod[m][dsn] = score


print "==== Best parameters ===="

for m in all_method_names:
	for dsn in processdata.all_dataset_names:
		ds_best = []
		ds_limit = 0
		time_overall = []
		for seed in [0, 1, 1000, 1001]:
			best = 0
			timedata = 0
			for limit in all_timelimits_seconds:
				p = ("%s-%ss#%s/%s" % (m, limit, dsn, seed))
				if p in recordall:
					best = avg(recordall[p])
					ds_limit = max(ds_limit, limit)
					timedata = timespentdata[p]
			ds_best.append(best)
			time_overall.append(timedata)
		print "%s-%s for %s: %.3f = %.3f%% in %.2fs" % (m, ds_limit, dsn, avg(ds_best), 100*avg(ds_best) / bestperdataset[dsn], avg(time_overall))
		perdataset[dsn][m] = avg(ds_best)
		permethod[m][dsn] = avg(ds_best)

best = {}

for dsn, dsnv in perdataset.iteritems():
	best[dsn] = max(dsnv.values())

print best

print "==== Overall scores ===="

for m, mv in permethod.iteritems():
	values = [v / best[k] for k, v in mv.iteritems()]
	print m, avg(values)
