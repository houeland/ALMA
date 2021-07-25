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
import numpy as np

import processdata

best_precomputed = {'car.data': 0.9401041666666667, 'monks-1.test': 0.8940972222222223, 'phishing-websites.data': 0.9564676616915422, 'connect-4.data': 0.7912947585002293, 'secom_processed.data': 0.9329929802169751, 'Skin_NonSkin.txt': 0.9720320170409333, 'house-votes-84.data': 0.9419540229885057, 'monks-2.test': 0.7586805555555556, 'travel-casebase': 0.3662109375, 'tic-tac-toe.data': 0.9444154488517745, 'agaricus-lepiota.data': 0.9977535696701132, 'SPECTF.test': 0.9144385026737968, 'kr-vs-kp.data': 0.9750469336670838, 'nursery-casebase': 0.9699845679012346, 'balance-scale.data': 0.8968, 'covtype-subset.data': 0.63703, 'soybean-large.data': 0.7728013029315961, 'poker-hand-subset.data': 0.7034900000000001, 'kddcup-subset.data': 0.9976149999999999, 'monks-3.test': 0.962962962962963, 'ThoraricSurgery.data': 0.8489361702127659}

almaname = 'DelayedRetrainer[C=0.05,num_methods=25]-3600s'
rfname = 'DR-weka.classifiers.trees.RandomForest-3600s'

recordall = {}

def mksid(algoname, dsn):
	return "%s#%s" % (algoname, dsn)

def score(results):
	return sum(1 if x is True else 0 for x in results)

def summarize(dsn):
	almarecord = recordall[mksid(almaname, dsn)]
	rfrecord = recordall[mksid(rfname, dsn)]
	print dsn, len(almarecord), len(rfrecord)
	print "  alma", score(almarecord)
	print "  rf", score(rfrecord)
	AR = 0
	Ar = 0
	aR = 0
	ar = 0
	for i in xrange(len(almarecord)):
		a = almarecord[i] is True
		r = rfrecord[i] is True
		if a and r: AR += 1
		elif a and not r: Ar += 1
		elif not a and r: aR += 1
		elif not a and not r: ar += 1
	print AR, Ar, aR, ar
	print "%.20f" % scipy.stats.binom_test([Ar, aR], 0.5)
	print

def unsumma(s):
	a = [1] * s[0] + [1] * s[1] + [0] * s[2] + [0] * s[3]
	b = [1] * s[0] + [0] * s[1] + [1] * s[2] + [0] * s[3]
	return a, b

def agrees(xs):
	pos = 0
	neg = 0
	for x in xs:
		if x >= 0: pos += 1
		else: neg += 1
	return pos == 0 or neg == 0

def summarize2(dsbase):
	ps = []
	for seed in [0, 1, 1000, 1001]:
		dsn = "%s/%s" % (dsbase, seed)
		almarecord = recordall[mksid(almaname, dsn)]
		rfrecord = recordall[mksid(rfname, dsn)]
#		print dsn, len(almarecord), len(rfrecord)
#		print "  alma", score(almarecord)
#		print "  rf", score(rfrecord)
		AR = 0
		Ar = 0
		aR = 0
		ar = 0
		for i in xrange(len(almarecord)):
			a = almarecord[i] is True
			r = rfrecord[i] is True
			if a and r: AR += 1
			elif a and not r: Ar += 1
			elif not a and r: aR += 1
			elif not a and not r: ar += 1
#		print AR, Ar, aR, ar
#		print "%.20f" % scipy.stats.binom_test([Ar, aR], 0.5)
#		print
		ps.append((scipy.stats.binom_test([Ar, aR], 0.5), (AR, Ar, aR, ar)))
#		ps.append((scipy.stats.fisher_exact([[AR, Ar], [aR, ar]]), (AR, Ar, aR, ar)))
#	print dsbase
	t_ts = []
	t_ps = []
	collects = [0, 0, 0, 0]
	for p in ps:
		for i, v in enumerate(p[1]):
			collects[i] += v
		a__, b__ = unsumma(p[1])
		t_p = scipy.stats.ttest_rel(a__, b__)
#		print "  ", p, t_p
		t_ts.append(t_p[0])
		t_ps.append(t_p[1])
#	if agrees(t_ts):
#		print "OK", t_ps[0], collects, scipy.stats.binom_test([collects[1], collects[2]], 0.5)
#	else:
#		print "--", t_ps[0], collects, scipy.stats.binom_test([collects[1], collects[2]], 0.5)
	print "%s %s %.0e" % (dsbase, "OK" if agrees(t_ts) else "--", scipy.stats.binom_test([collects[1], collects[2]], 0.5))
	print "  ", collects
	print "  X >= %d, n = %d, p = 0.5" % (max(collects[1], collects[2]), collects[1] + collects[2])

if __name__ == '__main__':
	for l in open("results/stats-test.txt"):
		js = json.loads(l)
		sid = mksid(js["algorithm"], js["dataset"])
		recordall[sid] = js["result"]

	for dsbase in processdata.all_dataset_names:
		summarize2(dsbase)
