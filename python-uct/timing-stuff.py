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

Classifier = processdata.Classifier

all_method_names = ['weka.classifiers.bayes.AODE', 'weka.classifiers.bayes.BayesNet', 'weka.classifiers.bayes.NaiveBayes', 'weka.classifiers.bayes.NaiveBayesSimple', 'weka.classifiers.functions.MultilayerPerceptron', 'weka.classifiers.functions.SMO', 'weka.classifiers.functions.VotedPerceptron', 'weka.classifiers.functions.Winnow', 'weka.classifiers.lazy.IB1', 'weka.classifiers.lazy.IBk', 'weka.classifiers.lazy.KStar', 'weka.classifiers.lazy.LBR', 'weka.classifiers.meta.AdaBoostM1', 'weka.classifiers.meta.LogitBoost', 'weka.classifiers.misc.HyperPipes', 'weka.classifiers.rules.JRip', 'weka.classifiers.rules.OneR', 'weka.classifiers.rules.Prism', 'weka.classifiers.rules.ZeroR', 'weka.classifiers.trees.DecisionStump', 'weka.classifiers.trees.Id3', 'weka.classifiers.trees.J48', 'weka.classifiers.trees.LMT', 'weka.classifiers.trees.RandomForest', 'weka.classifiers.trees.RandomTree']
all_dataset_names = ['agaricus-lepiota.data', 'balance-scale.data', 'car.data', 'connect-4.data', 'covtype-subset.data', 'house-votes-84.data', 'kddcup-subset.data', 'kr-vs-kp.data', 'monks-1.test', 'monks-2.test', 'monks-3.test', 'nursery-casebase', 'phishing-websites.data', 'poker-hand-subset.data', 'secom_processed.data', 'Skin_NonSkin.txt', 'soybean-large.data', 'SPECTF.test', 'ThoraricSurgery.data', 'tic-tac-toe.data', 'travel-casebase']

allresults = collections.defaultdict(list)

def avg(xs):
	xs = list(xs)
	if not xs: return "empty"
	return sum(xs) / len(xs)

def median(xs):
	s = sorted(xs)
	while len(s) > 2:
		s = s[1:-1]
	return (s[0] + s[-1]) / 2

def do_algo(name, ds):
	last_n = 0
	elapsed = 0
	for n_exp in xrange(100):
		n = 2 ** n_exp
		try:
			current = processdata.readdatasetfile(ds, "method", name, "seen_%d_examples.pickle" % n)
			which = "%d->%d" % (last_n, n)
			timed = (elapsed, current.timespent_seconds)
			allresults[which].append(timed)
			print which, timed
			elapsed = current.timespent_seconds
			last_n = n
		except IOError: break

def parse_allresults():
	for ds in processdata.get_datasets():
		print ds.description()
		for m in all_method_names:
			print m
			do_algo(m, ds)
		break

def save_allresults():
	print allresults
	with open('results/timing-stuff.pickle', 'w') as f:
		cPickle.dump(allresults, f)

def load_allresults():
	global allresults
	with open('results/timing-stuff.pickle') as f:
		allresults = cPickle.load(f)

if __name__ == '__main__':
	load_allresults()
	for k in sorted(allresults.keys(), key=lambda x:(len(x), x)):
		filtered = [x for x in allresults[k] if x[1] >= 2]
		avg=median
		old = avg(old for old, new in filtered)
		new = avg(new for old, new in filtered)
		increase = avg(new-old for old, new in filtered)
		p_increase = avg(new/old*100-100 if old else 0 for old, new in filtered)
		p_new_fraction = avg((new-old)/new*100 for old, new in filtered)
		print k, "%.3f ==> %.3f  | %+.3f |  %+.2f%%  |  %.2f%%" % (old, new, increase, p_increase, p_new_fraction)


#	print
#	print
#	print
#	for old, new in sorted(allresults["256->512"]):
#		print old, new
