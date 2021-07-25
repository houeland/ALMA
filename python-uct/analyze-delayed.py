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
			if seed > 1000 and 'Stacking' in m: continue
			p = ("%s#%s/%s" % (m, dsn, seed))
			best = avg(recordall.get(p, [0]))
			ds_best.append(best)
		score = avg(ds_best)
		bestperdataset[dsn] = max(bestperdataset[dsn], score)
		perdataset[dsn][m] = score
		permethod[m][dsn] = score


print "==== Best parameters ===="

timely = collections.defaultdict(dict)

for m in all_method_names:
	for dsn in processdata.all_dataset_names:
		ds_best = []
		time_overall = []
		for seed in [0, 1, 1000, 1001]:
			best = 0
			timedata = 0
			p = ("%s#%s/%s" % (m, dsn, seed))
			if p in recordall:
				best = avg(recordall[p])
				timedata = timespentdata[p]
			ds_best.append(best)
			time_overall.append(timedata)
#		print "%s for %s: %.3f = %.3f%% in %.2fs" % (m, dsn, avg(ds_best), 100*avg(ds_best) / bestperdataset[dsn], avg(time_overall))
		perdataset[dsn][m] = avg(ds_best)
		permethod[m][dsn] = avg(ds_best)
		timely[m][dsn] = avg(time_overall)

best = {}

best_precomputed = {'car.data': 0.9401041666666667, 'monks-1.test': 0.8940972222222223, 'phishing-websites.data': 0.9564676616915422, 'connect-4.data': 0.7912947585002293, 'secom_processed.data': 0.9329929802169751, 'Skin_NonSkin.txt': 0.9720320170409333, 'house-votes-84.data': 0.9419540229885057, 'monks-2.test': 0.7586805555555556, 'travel-casebase': 0.3662109375, 'tic-tac-toe.data': 0.9444154488517745, 'agaricus-lepiota.data': 0.9977535696701132, 'SPECTF.test': 0.9144385026737968, 'kr-vs-kp.data': 0.9750469336670838, 'nursery-casebase': 0.9699845679012346, 'balance-scale.data': 0.8968, 'covtype-subset.data': 0.63703, 'soybean-large.data': 0.7728013029315961, 'poker-hand-subset.data': 0.7034900000000001, 'kddcup-subset.data': 0.9976149999999999, 'monks-3.test': 0.962962962962963, 'ThoraricSurgery.data': 0.8489361702127659}

for dsn, dsnv in perdataset.iteritems():
	best[dsn] = max(best_precomputed[dsn], *dsnv.values())
#	if best[dsn] > best_precomputed[dsn]:
#		print "holy shit", dsn, best[dsn], best_precomputed[dsn]
#	best[dsn] = 1

for m in all_method_names:
	for dsn in processdata.all_dataset_names:
		if '-3600s' not in m: continue
		print "%s for %s: %.3f = %.3f%% in %.2fs" % (m, dsn, perdataset[dsn][m], 100*perdataset[dsn][m] / best[dsn], timely[m][dsn])


print "==== Per dataset ===="

for dsn in processdata.all_dataset_names:
	for m in all_method_names:
		if '-3600s' not in m: continue
		print "%s | %60s: %.3f = %.3f%% in %.2fs" % (dsn, m, perdataset[dsn][m], 100*perdataset[dsn][m] / best[dsn], timely[m][dsn])

print "==== Overall scores ===="

overall_scores = {}
all_scores = {}

for m, mv in permethod.iteritems():
	values = [v / best[k] for k, v in mv.iteritems()]
	overall_scores[m] = avg(values)
	all_scores[m] = values

for m, score in sorted(overall_scores.iteritems(), key=lambda (k,v): (v, k), reverse=True):
	if '-3600s' not in m: continue
	print '%s %.2f%%, %.1fs' % (m, score, sum(timely[m].values()))

print "==== Table ===="
def outputter():
	alma_score_list = []
	rf_score_list = []
	for dsn in processdata.all_dataset_names:
		alma = perdataset[dsn]["DelayedRetrainer[C=0.05,num_methods=25]-3600s"]
		rf = perdataset[dsn]["DR-weka.classifiers.trees.RandomForest-3600s"]
		target = best[dsn]
		alma_score = 100*alma / target
		rf_score = 100*rf / target
		alma_score_list.append(alma_score)
		rf_score_list.append(rf_score)
		yield (dsn, target, alma_score, rf_score)
	yield ("\\hline", None, None, None)
	yield ("Average (mean score)", "", avg(alma_score_list), avg(rf_score_list))

for (dsn, target, alma_score, rf_score) in outputter():
	if alma_score is None:
		print dsn
		continue
	alma_print = "%.2f\\%%" % alma_score
	rf_print = "%.2f\\%%" % rf_score
	diff_print = "%+.2f\\%%" % (alma_score - rf_score)
	if alma_score - rf_score >= 5.0:
		alma_print = "\\textbf{%s}" % alma_print
		diff_print = "\\textbf{%s}" % diff_print
	elif rf_score - alma_score >= 5.0:
		rf_print = "\\textbf{%s}" % rf_print
		diff_print = "\\textbf{%s}" % diff_print
	try:
		target = "%.4f" % target
	except:
		pass
	print "%s & %s & %s & %s & %s \\\\" % (dsn.replace("_", "\\_"), target, alma_print, rf_print, diff_print)

print "==== Table ===="

for m, score in sorted(overall_scores.iteritems(), key=lambda (k,v): (v, k), reverse=True):
	if '-3600s' in m:
		m60 = m.replace('-3600s', '-60s')
		print "%s & %.1f%% & %.1f%%" % (m, overall_scores[m] * 100, overall_scores[m60] * 100)

print "==== Graph ===="

graph_methods = {
	'DelayedRetrainer': 'DelayedRetrainer[C=0.05,num_methods=25]',
	'UpdateAllDelayed': 'UpdateAllDelayed',
	'RandomForest': 'DR-weka.classifiers.trees.RandomForest',
	'SMO': 'DR-weka.classifiers.functions.SMO',
	'LMT': 'DR-weka.classifiers.trees.LMT',
	'MultilayerPerceptron': 'DR-weka.classifiers.functions.MultilayerPerceptron',
	'LogitBoost': 'DR-weka.classifiers.meta.LogitBoost',
}

for name, lookup in sorted(graph_methods.iteritems()):
	break
	fout = open(os.path.join('gnuplot-table', '%s.plot' % name), 'w')
	print >> fout, '# %s / %s' % (name, lookup)
	for v in [0,0,60,180,300,450,600,900,1200,1800,2400,3000,3600,4500,5400,6300,7200]:
		print >> fout, '%s;%s' % (v / 60.0, overall_scores.get("%s-%ds" % (lookup, v), 0) * 100)
	fout.close()

print "==== Stuff ===="

def magnitude(xs):
	xs = list(xs)
	return (sum(x**2 for x in xs)/len(xs))**0.5

def superscore(xs):
	return 1 - magnitude(1-x for x in xs)
	

print "--- LOSS anti-magnitude ---"
for m in sorted(all_method_names, key=lambda m: superscore(all_scores[m]), reverse=True):
	if '-3600s' not in m: continue
	scores = []
	for dsn in processdata.all_dataset_names:
		scores.append(perdataset[dsn][m] / best[dsn])
	scores.sort()
	print '%50s' % m.replace('DR-','').split('-')[0], ' '.join('%x' % int('%.0f' % (s*10)) for s in scores), 1-superscore(all_scores[m])

print
print

print "--- MAGNITUDE ---"
for m in sorted(all_method_names, key=lambda m: magnitude(all_scores[m]), reverse=True):
	if '-3600s' not in m: continue
	scores = []
	for dsn in processdata.all_dataset_names:
		scores.append(perdataset[dsn][m] / best[dsn])
	scores.sort()
	print '%50s' % m.replace('DR-','').split('-')[0], ' '.join('%x' % int('%.0f' % (s*10)) for s in scores), magnitude(all_scores[m])


print
print

print "--- AVGSCORE ---"
for m in sorted(all_method_names, key=lambda m: avg(all_scores[m]), reverse=True):
	if '-3600s' not in m: continue
	scores = []
	for dsn in processdata.all_dataset_names:
		scores.append(perdataset[dsn][m] / best[dsn])
	print '%50s' % m.replace('DR-','').split('-')[0], ' '.join('%x' % int('%.0f' % (s*10)) for s in scores), avg(all_scores[m])
