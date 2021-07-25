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
#all_method_names = ['weka.classifiers.bayes.AODE', 'weka.classifiers.trees.RandomForest', 'weka.classifiers.rules.ZeroR']
all_method_names = ['weka.classifiers.trees.RandomForest']

def avg(xs): return sum(xs) / len(xs)

class Node(object):
	def __init__(self, parent, algo):
		self.visits = 0
		self.wins = 0
		self.parent = parent
		self.algo = algo
		self.timespent = 0
		self.last_print = 0

def score(results):
	return sum(1 if r is True else 0 for r in results) / len(results)

def _desc(self, name):
	if not self.time_limit_seconds:
		return name
	else:
		return "%s-%ss" % (name, self.time_limit_seconds)


class BaseAlgorithm(object):
	def __init__(self, name, ds, time_limit_seconds=None, n_limit=1000000000):
		self.name = name
		self.time_limit_seconds = time_limit_seconds
		results = []
		self.elapsed = 0
		self.n_limit = n_limit
		self.highest_n_trained_on = 0
		self.timed_out = False
		for n_exp in xrange(100):
			n = 2 ** n_exp
			if n > self.n_limit: break
			try:
				current = processdata.readdatasetfile(ds, "method", self.name, "seen_%d_examples.pickle" % n)
				self.elapsed += current.timespent_seconds
				if self.exceeded_time_limit():
					self.timed_out = True
					break
				if current.results == "failed": continue
				while len(results) < n: results.append(False)
#				print n, current.results
#				print n, score(current.results)
#				print n, current
				results[n:] = current.results
				self.highest_n_trained_on = n
			except IOError: break
		self.results = results

	def exceeded_time_limit(self):
		return self.time_limit_seconds is not None and self.elapsed > self.time_limit_seconds

	def time_spent(self):
		if self.time_limit_seconds is not None:
			return min(self.elapsed, self.time_limit_seconds)
		else:
			return self.elapsed

	def description(self):
		return _desc(self, self.name)

	def predict(self, i):
		try:
			return self.results[i]
		except LookupError:
			return "failed"

def score_node(n, exploration_multiplier=1):
	if n.visits == 0:
		return 1000000 + random.random()
	exploit = n.wins / n.visits
	parent = n.parent
	variance_estimate = 1000000
	if n.visits > 1:
		variance_estimate = n.wins / (n.visits - 1) * (1 - n.wins / n.visits)
	V = variance_estimate + math.sqrt(2 * math.log(parent.visits) / n.visits)
	explore_tuned = math.sqrt(math.log(parent.visits) / (n.visits) * min(1/4, V))
	timespent_modifier = 1 # math.log(math.exp(1) + n.timespent / n.visits)
	return exploit + explore_tuned * exploration_multiplier / timespent_modifier

def update_node(n, result):
	n.visits += 1
	n.wins += result
	if n.parent: update_node(n.parent, result)

def do_node(node, i):
	try:
		prediction = node.algo.predict(i)
	except LookupError:
		prediction = "failed"
		raise
	update_node(node, prediction is True)
	return prediction

def add_node_timespent(node, timespent):
	node.timespent += timespent
	if node.parent: add_node_timespent(node.parent, timespent)


def uct_eval_tuned(n, parent, score):
	if n == 0: return 1000000
	variance_estimate = 1000000
	if n > 1:
		variance_estimate = score / (n - 1) * (1 - score / n)
	V = variance_estimate + math.sqrt(2 * math.log(parent) / n)
	explore_tuned = math.sqrt(math.log(parent) / (n) * min(1/4, V))
	return (score / n) + explore_tuned

def descacc(node):
	return node.wins / node.visits if node.visits else 0


class UCTSmartRetrainer(object):
	def __init__(self, ds, time_limit_seconds=None, C_retrain=0.5):
		self.ds = ds
		self.rootnode = Node(None, None)
		self.nodes = {m: Node(self.rootnode, BaseAlgorithm(m, ds, time_limit_seconds=1.0, n_limit=0)) for m in all_method_names}
		self.elapsed = 0
		self.C_retrain = C_retrain
		self.time_limit_seconds = time_limit_seconds

	def exceeded_time_limit(self):
		return self.time_limit_seconds and self.elapsed >= self.time_limit_seconds

	def time_remaining(self):
		return self.time_limit_seconds - self.elapsed if self.time_limit_seconds else None

	def _score_node_for_example_num(self, node, i, totalish_time, debugprint=False):
		if node.algo.n_limit == 0:
			return 1000000, 0
		elif node.visits == 0:
			accuracy = 0
		else:
			accuracy = node.wins / node.visits
#		exploration_value = math.sqrt(math.log(i) / node.algo.n_limit)
#		reuse_score = accuracy + exploration_value * math.sqrt(self.C_reuse)
#		retrain_score = accuracy + exploration_value * math.sqrt(self.C_retrain) * (i-node.algo.n_limit)/node.algo.n_limit
#		print "    %s w[%s] v[%s]" % (node.algo.description(), node.wins, node.visits)
		exploration_value = uct_eval_tuned(node.algo.n_limit, i, accuracy)
		reuse_score = accuracy + exploration_value
#		n_mod = math.sqrt((i-node.algo.n_limit)/node.algo.n_limit)
		if node.algo.timed_out and node.algo.highest_n_trained_on:
			n_mod = (i-node.algo.highest_n_trained_on)/node.algo.highest_n_trained_on
		else:
			n_mod = (i-node.algo.n_limit)/node.algo.n_limit*5
		time_mod = totalish_time / node.timespent if node.timespent else 1
#		time_mod = 1
		retrain_score = accuracy + exploration_value * self.C_retrain * n_mod * time_mod
		debugprint = False
		if debugprint:
			print "    %s score: %.2f, %.2f" % (node.algo.name, retrain_score, reuse_score)
			print "        retrain[%.2f] = accuracy[%.2f] + explore[%.2f] * C[%.2f] * n_mod[%.2f]" % (retrain_score, accuracy, exploration_value, self.C_retrain, n_mod)
#			print "        retrain[%.2f] = accuracy[%.2f] + explore[%.2f] * C[%.2f] * n_mod[%.2f] * time_mod[%.2f]" % (retrain_score, accuracy, exploration_value, self.C_retrain, n_mod, time_mod)
#			print "        time_mod[%.2f] = totalish[%.2fs] / node[%.2fs]" % (time_mod, totalish_time, node.timespent)
		return retrain_score, reuse_score

	def _do_update_node(self, node, time_limit_seconds, n_limit):
		old_timespent = node.timespent
		node.algo = BaseAlgorithm(node.algo.name, self.ds, time_limit_seconds=time_limit_seconds, n_limit=n_limit)

		# TODO: retroactively recompute score
		print "BEFORE", node.wins, node.visits
		print "RECOMP", sum(node.algo.predict(x) is True for x in range(n_limit-1)), n_limit-1
		node.wins = sum(node.algo.predict(x) is True for x in range(n_limit-1))

		new_timespent = node.algo.time_spent()
		timediff = new_timespent - old_timespent
		add_node_timespent(node, timediff)
		self.elapsed += timediff
		if self.exceeded_time_limit() or True:
			chosen_k = max(self.nodes, key=lambda k:score_node(self.nodes[k], 0))
			chosen_node = self.nodes[chosen_k]
			print "do_update_node(%s, %.1f, %d) => %+.3fs for N=%d at %.3f  |  Best is %s at %.3f" % (node.algo.name, time_limit_seconds, n_limit, timediff, node.algo.highest_n_trained_on, descacc(node), chosen_node.algo.name, descacc(chosen_node))

	def _will_retrain(self, node, i):
		if self.exceeded_time_limit():
			return False
		elif node.algo.time_limit_seconds > self.time_remaining():
			return False
		else:
			return True

	def _retrain(self, node, i):
		if not self._will_retrain(node, i):
			return
		self._do_update_node(node, min(self.time_remaining(), max(1, node.timespent * 2)), i)

	def time_spent(self):
		return self.elapsed

	def description(self):
		return _desc(self, 'UCTSmartRetrainer[%s]' % (self.C_retrain))

	def predict(self, i):
		if i == 0:
			return "failed"
		chosen_k = max(self.nodes, key=lambda k:score_node(self.nodes[k], 0))
		prediction = self.nodes[chosen_k].algo.predict(i)
		retrain_scores = {}
		all_scores = set()
		totalish_time = 0
		for m, node in self.nodes.iteritems():
			accuracy = node.wins / node.visits if node.visits else 0.5
			accuracy = 0.25  # hack, disable for now
			totalish_time += node.timespent * accuracy * accuracy
		for m, node in self.nodes.iteritems():
			retrain, reuse = self._score_node_for_example_num(node, i, totalish_time)
			retrain_scores[m] = retrain
			all_scores.add(retrain)
			all_scores.add(reuse)
#			print "  %s score = %.3f | %.3f" % (node.algo.description(), retrain, reuse)
		best_score = max(all_scores)
		to_retrain = []
		for m, node in self.nodes.iteritems():
			if retrain_scores[m] == best_score:
				to_retrain.append(m)

		chosen_m = random.choice(to_retrain) if to_retrain else None
		for m, node in self.nodes.iteritems():
			if chosen_m == m:
#				print "retraining %s at n=%d (from %d)" % (node.algo.description(), i, node.algo.n_limit)
				if self._will_retrain(node, i):
					self._score_node_for_example_num(node, i, totalish_time, True)
				self._retrain(node, i)
			do_node(node, i)
		return prediction


updateall_data = """
UpdateAll-30 agaricus-lepiota.data 0.996984244215
UpdateAll-600 balance-scale.data 0.8872
UpdateAll-600 car.data 0.937210648148
UpdateAll-20 connect-4.data 0.547071361961
UpdateAll-20 covtype-subset.data 0.49772
UpdateAll-600 house-votes-84.data 0.933333333333
UpdateAll-20 kddcup-subset.data 0.84337
UpdateAll-75 kr-vs-kp.data 0.972309136421
UpdateAll-400 monks-1.test 0.877314814815
UpdateAll-300 monks-2.test 0.849537037037
UpdateAll-400 monks-3.test 0.956018518519
UpdateAll-30 nursery-casebase 0.931790123457
UpdateAll-20 phishing-websites.data 0.927905924921
UpdateAll-20 poker-hand-subset.data 0.50571
UpdateAll-20 secom_processed.data 0.932354818124
UpdateAll-20 Skin_NonSkin.txt 0.792440126175
UpdateAll-600 soybean-large.data 0.763843648208
UpdateAll-400 SPECTF.test 0.911764705882
UpdateAll-600 ThoraricSurgery.data 0.843617021277
UpdateAll-600 tic-tac-toe.data 0.935803757829
UpdateAll-200 travel-casebase 0.34765625
"""

def get_updateall_timelimit_for(dsname, time_limit):
	if not time_limit:
		return None
	elif time_limit == 600:
		for l in updateall_data.splitlines():
			if dsname in l:
				m = l.split(' ')[0]
				return int(m.split('-')[1])
	else:
		raise ValueError('untrained time limit for updateall')
			

class UpdateAll(object):
	def __init__(self, ds, time_limit_seconds=None):
		self.ds = ds
		self.rootnode = Node(None, None)
		sub_time_limit = get_updateall_timelimit_for(ds.name, time_limit_seconds)
		self.nodes = {m: Node(self.rootnode, BaseAlgorithm(m, ds, sub_time_limit)) for m in all_method_names}
		self.time_limit_seconds = time_limit_seconds
		self.elapsed = sum(v.algo.time_spent() for v in self.nodes.values())

	def time_spent(self):
		return self.elapsed

	def description(self):
		return _desc(self, 'UpdateAll')

	def predict(self, i):
		if self.time_limit_seconds and self.elapsed > self.time_limit_seconds: return 'failed'
		chosen_k = max(self.nodes, key=lambda k:score_node(self.nodes[k], 0))
		for k in self.nodes:
			v = do_node(self.nodes[k], i)
			if k == chosen_k:
				prediction = v
		return prediction

def evaluate(ds, algo):
#	classifications = readdatasetfile(ds, "classifications.pickle")
	result = [algo.predict(i) for i in xrange(ds.length)]
	result_score = score(result)
	return result_score

def evaluate_and_print(ds, algo):
	result = [algo.predict(i) for i in xrange(ds.length)]
	result_score = score(result)
	print algo.description(), "%.3f" % result_score, ds.description(), "%.2fs" % algo.time_spent()
	return result_score

#scan_datasets()

#scan_output()

def do_N(ds, algo, N):
	res = []
	for x in xrange(N):
		res.append(evaluate(ds, algo))
	print "average:", avg(res)


def record(algo, ds, result):
	with open("results/recorded.txt", "a") as f:
		json.dump(dict(algorithm=algo.description(), dataset=ds.description(), result=result, time_spent=algo.time_spent()), f)
		print >> f, ""


def doalgo(algorithm):
	for ds in processdata.get_datasets():
		algo = algorithm(ds)
		result = evaluate_and_print(ds, algo)
		record(algo, ds, result)

def manual_test():
#	ds = processdata.Dataset("poker-hand-subset.data", 0)
#	for C in [0.01, 0.05, 0.1, 0.5]:
#		for i in [0,1,1000,1001]:
#		for (dsname, i) in [("connect-4.data", 0), ("covtype-subset.data", 0), ("kddcup-subset.data", 1000), ("monks-2.test", 0), ("poker-hand-subset.data", 0), ("Skin_NonSkin.txt", 0)]:
	for (dsname, i) in [("poker-hand-subset.data", 0)]:
		ds = processdata.Dataset(dsname, i)
		evaluate_and_print(ds, UCTSmartRetrainer(ds, 6000))

	return
	evaluate(ds, BaseAlgorithm("weka.classifiers.functions.MultilayerPerceptron", ds))
	evaluate(ds, BaseAlgorithm("weka.classifiers.functions.MultilayerPerceptron", ds, 600))
	evaluate(ds, BaseAlgorithm("weka.classifiers.functions.MultilayerPerceptron", ds, 200))
	evaluate(ds, BaseAlgorithm("weka.classifiers.functions.MultilayerPerceptron", ds, 10))
	evaluate(ds, UpdateAll(ds))
	evaluate(ds, UpdateAll(ds, 600))
	evaluate(ds, UpdateAll(ds, 200))
	evaluate(ds, UpdateAll(ds, 100))
	evaluate(ds, UpdateAll(ds, 50))
	evaluate(ds, UpdateAll(ds, 20))
	evaluate(ds, UpdateAll(ds, 10))

def doalgos(timespec):
	doalgo(lambda ds: UCTSmartRetrainer(ds, timespec, 0.5))
	doalgo(lambda ds: UCTSmartRetrainer(ds, timespec, 0.1))
	doalgo(lambda ds: UCTSmartRetrainer(ds, timespec, 1.0))
	doalgo(lambda ds: UCTSmartRetrainer(ds, timespec, 0.25))
#	doalgo(lambda ds: UCTRetrainer(ds, timespec))
#	doalgo(lambda ds: UpdateAll(ds, timespec))

#	for m in all_method_names:
#		doalgo(lambda ds: BaseAlgorithm(m, ds, timespec))

def doallalgos():
	doalgos(600)

	doalgos(None)
	doalgos(200)
	doalgos(100)
	doalgos(50)
	doalgos(20)

	doalgos(500)
	doalgos(400)
	doalgos(300)
	doalgos(150)
	doalgos(75)
	doalgos(30)
	doalgos(10)


if __name__ == '__main__':
	manual_test()
#	doallalgos()
#	ds = processdata.Dataset("poker-hand-subset.data", 0)



