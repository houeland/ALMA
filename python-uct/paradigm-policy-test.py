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

def avg(xs): return sum(xs) / len(xs)

def doalgo(ds, algoname, n_limit, time_limit):
	results = []
	elapsed = 0
	highest_n_trained_on = 0
	for n_exp in itertools.count():
		n = 2 ** n_exp
		if n > n_limit: break
		try:
			current = processdata.readdatasetfile(ds, "method", algoname, "seen_%d_examples.pickle" % n)
			elapsed += current.timespent_seconds
#			if elapsed > time_limit:
			if current.timespent_seconds > time_limit:
				break
			if current.results == "failed":
				continue
			highest_n_trained_on = n
			while len(results) < n: results.append(False)
			results[n:] = current.results
		except IOError: break
	return results

def descacc(node):
	return node.wins / node.visits if node.visits else 0

def dodataset(ds):
	scores = {}
	for m in all_method_names:
		result = doalgo(ds, m, ds.length / 2, 600)
		score = sum(result[:ds.length // 2])
#		print m, score
		scores[m] = score
	best = max(scores.values())
#	print scores
	return [k for k, v in scores.items() if v == best]

def doall():
	bigone = {}
	for ds in processdata.get_datasets():
		sel = dodataset(ds)
		print ds, sel
		bigone[ds.description()] = sel
		processdata.clear_cache()
	print bigone
	f = open("paradigm-policy-test___ds-algo-selections__noncumul_timelimit.pickle", "w")
	cPickle.dump(bigone, f)
	f.close()

def allresults():
	bigone = cPickle.load(open("paradigm-policy-test___ds-algo-selections.pickle"))
	bigresult = []
	for ds in processdata.get_datasets():
		sel = bigone[ds.description()]
		print ds, sel
		score = 0
		total = 0
		for algoname in sel:
			result = doalgo(ds, algoname, ds.length / 2, 600)
			r = result[ds.length // 2:]
			score += sum(r)
			total += len(r)
			print algoname, sum(r)
		print ds, score/total
		bigresult.append(score/total)
	print "total avg score", avg(bigresult)

if __name__ == '__main__':
	doall()
#	allresults()
