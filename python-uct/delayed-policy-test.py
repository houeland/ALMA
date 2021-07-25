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
#all_method_names = ['weka.classifiers.trees.RandomForest']
#selected_method_names = ['weka.classifiers.rules.JRip', 'weka.classifiers.trees.LMT']
#selected_method_names = ['weka.classifiers.trees.RandomForest']
#selected_method_names = ['weka.classifiers.meta.Stacking']

def avg(xs): return sum(xs) / len(xs)

def next_train_target(node, i):
	if node.algo.just_failed:
#		print "%s next_trainA: %d" % (node.algo.name, node.algo.just_failed * 2)
		return node.algo.just_failed * 2
	else:
#		print "%s next_trainB: %d" % (node.algo.name, min(i, max(1, node.algo.highest_n_trained_on * 2)))
		return max(1, node.algo.highest_n_trained_on * 2)

class Node(object):
	def __init__(self, parent, algo):
		self.visits = 0
		self.wins = 0
		self.parent = parent
		self.algo = algo
		self.timespent = 0
		self.last_print = 0
		self.highest_attempted_without_timeout = 0

def score(results):
	return sum(1 if r is True else 0 for r in results) / len(results)

def _desc(self, name):
	return name


class BaseAlgorithm(object):
	def __init__(self, name, ds, time_limit_seconds=None, n_limit=1000000000):
		self.name = name
		self.time_limit_seconds = time_limit_seconds
		results = []
		self.elapsed = 0
		self.n_limit = n_limit
		self.highest_n_trained_on = 0
		self.timed_out = False
		self.next_interesting_n = 0
		self.next_interesting_time = 0
		self.just_failed = 0
		for n_exp in itertools.count():
			n = 2 ** n_exp
			self.next_interesting_n = n
			if n > self.n_limit: break
			self.just_failed = 0
			try:
				current = processdata.readdatasetfile(ds, "method", self.name, "seen_%d_examples.pickle" % n)
				self.elapsed += current.timespent_seconds
				if self.exceeded_time_limit():
					self.timed_out = True
					self.next_interesting_time = self.elapsed
					break
				self.highest_n_trained_on = n
				if current.results == "failed":
					self.just_failed = n
					continue
				while len(results) < n: results.append(False)
				results[n:] = current.results
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


class DelayedRetrainer(object):
	def __init__(self, ds, method_list=None, C_retrain=0.05, total_time_budget_seconds=60*60, time_per_example_milliseconds=None, method_mod=None):
		if method_list is None:
			method_list = all_method_names
		if method_mod:
			always_keep = set(['weka.classifiers.functions.MultilayerPerceptron', 'weka.classifiers.trees.RandomForest', 'weka.classifiers.meta.LogitBoost', 'weka.classifiers.functions.SMO'])
#			always_keep.add(method_mod)
			method_list = [m for m in method_list if m in always_keep]
		self.method_list = method_list
		self.method_mod = method_mod
		self.ds = ds
		self.rootnode = Node(None, None)
		self.nodes = {m: Node(self.rootnode, BaseAlgorithm(m, ds, time_limit_seconds=0.0, n_limit=0)) for m in self.method_list}
		self.C_retrain = C_retrain
		self.total_available_time = 0
		self.elapsed = 0
		self.total_time_budget_seconds = total_time_budget_seconds
		if total_time_budget_seconds:
			self.time_per_example_seconds = self.total_time_budget_seconds / ds.length
		else:
			self.time_per_example_seconds = time_per_example_milliseconds * 1000
		self.training_in_progress_f = None

	def _score_node_for_example_num(self, node, i, totalish_time, debugprint=False):
		if node.algo.n_limit == 0:
			return 1000000
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
#		if node.algo.timed_out and node.algo.highest_n_trained_on:
#			n_mod = (i-node.algo.highest_n_trained_on)/node.algo.highest_n_trained_on
#		else:
#			n_mod = (i-node.algo.n_limit)/node.algo.n_limit*5
		n_mod_basis = (i-node.algo.n_limit)/node.algo.n_limit
		n_mod = math.sqrt(math.log(n_mod_basis+1))
		time_mod = totalish_time / node.timespent if node.timespent else 1
#		time_mod = 1
		retrain_score = accuracy + (accuracy + exploration_value) * self.C_retrain * n_mod * time_mod
		debugprint = False
		if debugprint:
			print "    %s score: %.2f" % (node.algo.name, retrain_score)
			print "        retrain[%.2f] = accuracy[%.2f] + explore[%.2f] * C[%.2f] * n_mod[%.2f]" % (retrain_score, accuracy, exploration_value, self.C_retrain, n_mod)
			print "        n_mod[%.2f] from i[%d] highest_n_trained_on[%d] n_limit[%d] timed_out[%s]" % (n_mod, i, node.algo.highest_n_trained_on, node.algo.n_limit, node.algo.timed_out)
#			print "        time_mod[%.2f] = totalish[%.2fs] / node[%.2fs]" % (time_mod, totalish_time, node.timespent)
		return retrain_score * (i >= node.algo.highest_n_trained_on * 2)

	def time_spent(self):
		return self.elapsed

	def _do_update_node(self, node, time_limit_seconds, n_limit_train, n_limit_predict, printscore):
		old_timespent = node.timespent
		algo = BaseAlgorithm(node.algo.name, self.ds, time_limit_seconds=time_limit_seconds, n_limit=n_limit_train)
		new_timespent = algo.time_spent()
		timediff = new_timespent - old_timespent
		def f():
			self.elapsed += timediff
			node.algo = algo

		# TODO: retroactively recompute score
#		print "BEFORE", node.wins, node.visits
#		print "RECOMP", sum(node.algo.predict(x) is True for x in range(n_limit-1)), n_limit-1
			node.wins = sum(node.algo.predict(x) is True for x in range(n_limit_predict))
			node.highest_attempted_without_timeout = 0
			if new_timespent < time_limit_seconds:
				node.highest_attempted_without_timeout = n_limit_train

			add_node_timespent(node, timediff)
			if False:
				chosen_k = max(self.nodes, key=lambda k:score_node(self.nodes[k], 0))
				chosen_node = self.nodes[chosen_k]
				print "%.3f: do_update_node(%s, %.1f, %d) => %+.3fs for N=%d at %.3f  |  Best is %s at %.3f of %d" % (printscore, node.algo.name, time_limit_seconds, node.algo.highest_n_trained_on, timediff, node.algo.highest_n_trained_on, descacc(node), chosen_node.algo.name, descacc(chosen_node), n_limit_predict)
#		print "  ...starting to train %s (t=%.2fs, n=%d) %+.1fs" % (node.algo.name, time_limit_seconds, n_limit_train, timediff)
		self.train_f_for_n_seconds(f, timediff)

	def train_f_for_n_seconds(self, f, timediff):
		if self.training_in_progress_f:
			raise RuntimeError
		self.training_in_progress_f = f
		if timediff >= 0:
			self.total_available_time -= timediff

	def _will_retrain(self, node, i):
#		print "next train", next_train_target(node, i), i
		if next_train_target(node, i) <= node.highest_attempted_without_timeout:
#			print "no retrain because it's already done"
			return False
		if next_train_target(node, i) > i:
#			print "no retrain because not enough examples"
			return False
		return True

	def _retrain(self, node, i, printscore):
		if not self._will_retrain(node, i):
			return
		self._do_update_node(node, max(1, node.timespent * 2), next_train_target(node, i), i-1, printscore)

	def description(self):
		if self.total_time_budget_seconds:
			timespec = '%ds' % self.total_time_budget_seconds
		else:
			timespec = '%dms' % self.time_per_example_seconds * 1000

		if len(self.method_list) == 1:
			return _desc(self, 'DR-%s-%s' % (self.method_list[0], timespec))
		elif self.method_mod:
			return _desc(self, 'DRminus[C=%s,M=MLP,RF,LB,SMO]-%s' % (self.C_retrain, timespec))
		else:
			return _desc(self, 'DelayedRetrainer[C=%s,num_methods=%d]-%s' % (self.C_retrain, len(self.method_list), timespec))

	def predict(self, i):
#		print "predict %d" % i
		self.total_available_time += self.time_per_example_seconds
		if i == 0:
			return "failed"
		while self.total_available_time >= 0:
#			print "...eval"
			if self.training_in_progress_f:
				self.training_in_progress_f()
				self.training_in_progress_f = None
			retrain_scores = {}
			all_scores = set()
			all_scores.add(-1000)
			totalish_time = 0
			for m, node in self.nodes.iteritems():
				accuracy = node.wins / node.visits if node.visits else 0.5
				accuracy = 0.25  # hack, disable for now
				totalish_time += node.timespent * accuracy * accuracy
			for m, node in self.nodes.iteritems():
				retrain = self._score_node_for_example_num(node, i, totalish_time)
				retrain_scores[m] = retrain
				if self._will_retrain(node, i):
					all_scores.add(retrain)
#				if m == chosen_k and self.total_available_time >= 0:
#					self._score_node_for_example_num(node, i, totalish_time, True)
			best_score = max(all_scores)
			to_retrain = []
			for m, node in self.nodes.iteritems():
				if retrain_scores[m] == best_score:
					to_retrain.append(m)
			if not to_retrain:
				break
			chosen_m = random.choice(to_retrain)
#			print "chosen:", chosen_m, "time:", self.total_available_time, "from:", to_retrain
			for m, node in self.nodes.iteritems():
				if chosen_m == m:
	#				print "  %s score = %.3f" % (node.algo.description(), best_score)
	#				print "retraining %s at n=%d (from %d)" % (node.algo.description(), i, node.algo.n_limit)
					self._score_node_for_example_num(node, i, totalish_time, True)
					self._retrain(node, i, retrain_scores[m])



		chosen_k = max(self.nodes, key=lambda k:score_node(self.nodes[k], 0))
		chosen_node = self.nodes[chosen_k]
		prediction = chosen_node.algo.predict(i)
		for m, node in self.nodes.iteritems():
			do_node(node, i)
		return prediction


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
			

class UpdateAllDelayed(object):
	def __init__(self, ds, total_time_budget_seconds=60*60, time_per_example_milliseconds=None):
		self.ds = ds
		self.rootnode = Node(None, None)
		self.nodes = {m: Node(self.rootnode, BaseAlgorithm(m, ds, 0)) for m in all_method_names}
		self.total_available_time = 0
		self.elapsed = 0
		self.total_time_budget_seconds = total_time_budget_seconds
		if total_time_budget_seconds:
			self.time_per_example_seconds = self.total_time_budget_seconds / ds.length
		else:
			self.time_per_example_seconds = time_per_example_milliseconds * 1000
		self.training_in_progress_f = None
		def schedule():
			nodes = self.nodes.values()
			for c in itertools.count():
				for n in random.sample(nodes, len(nodes)):
					yield n, c
		self.schedule = schedule()

	def time_spent(self):
		return self.elapsed

	def description(self):
		if self.total_time_budget_seconds:
			return _desc(self, 'UpdateAllDelayed-%ds' % self.total_time_budget_seconds)
		else:
			return _desc(self, 'UpdateAllDelayed-%dms' % self.time_per_example_seconds * 1000)

	def predict(self, i):
		self.total_available_time += self.time_per_example_seconds
#		print "step", i, self.total_available_time
		while self.total_available_time >= 0:
			if self.training_in_progress_f:
				self.training_in_progress_f()
				self.training_in_progress_f = None
			node, time_limit_seconds = self.schedule.next()
#			print "training %s: %ds, n=%d" % (node.algo.name, time_limit_seconds, i-1)
			if node.algo.next_interesting_n <= i-1 and node.algo.next_interesting_time <= time_limit_seconds:
				old_timespent = node.timespent
				algo = BaseAlgorithm(node.algo.name, self.ds, time_limit_seconds=time_limit_seconds, n_limit=i-1)
				new_timespent = algo.time_spent()
				timediff = new_timespent - old_timespent
				def f():
#					print "...trained %s: %d (%d/%d) in %.1fs/%ds (+%.1fs)" % (node.algo.name, new_timespent, node.algo.highest_n_trained_on, i-1, new_timespent, time_limit_seconds, timediff)
					self.elapsed += timediff
					node.algo = algo
					node.wins = sum(node.algo.predict(x) is True for x in range(i-1))
					add_node_timespent(node, timediff)
				self.training_in_progress_f = f
				self.total_available_time -= timediff
			elif node.algo.next_interesting_time > time_limit_seconds:
				def f():
					self.elapsed += 1
					add_node_timespent(node, 1)
				self.training_in_progress_f = f
				self.total_available_time -= 1
			else:
				break
		chosen_k = max(self.nodes, key=lambda k:(score_node(self.nodes[k], 0), random.random()))
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
	with open("results/recorded-delayed-training.txt", "a") as f:
		json.dump(dict(algorithm=algo.description(), dataset=ds.description(), result=result, time_spent=algo.time_spent()), f)
		print >> f, ""


def doalgo(algorithm):
	for ds in processdata.get_datasets():
		processdata.clear_cache()
		algo = algorithm(ds)
		result = evaluate_and_print(ds, algo)
		record(algo, ds, result)

def manual_test():
#	ds = processdata.Dataset("poker-hand-subset.data", 0)
#	for C in [0.01, 0.05, 0.1, 0.5]:
#		for i in [0,1,1000,1001]:
#	for (dsname, i) in [("connect-4.data", 0), ("covtype-subset.data", 0), ("kddcup-subset.data", 1000), ("monks-2.test", 0), ("poker-hand-subset.data", 0), ("Skin_NonSkin.txt", 0)]:
	for (dsname, i) in [("connect-4.data", 0), ("connect-4.data", 1), ("connect-4.data", 1000), ("connect-4.data", 1001)]:
#	for (dsname, i) in [("poker-hand-subset.data", 0)]:
		if "connect" not in dsname: continue
		processdata.clear_cache()
		ds = processdata.Dataset(dsname, i)
#		evaluate_and_print(ds, UpdateAllDelayed(ds))
		evaluate_and_print(ds, DelayedRetrainer(ds))
#		evaluate_and_print(ds, DelayedRetrainer(ds, method_list=selected_method_names))
	return

def doalgos(timespec):
#	doalgo(lambda ds: UCTSmartRetrainer(ds, timespec, 0.5))
#	doalgo(lambda ds: UCTSmartRetrainer(ds, timespec, 0.1))
#	doalgo(lambda ds: UCTSmartRetrainer(ds, timespec, 1.0))
#	doalgo(lambda ds: UCTSmartRetrainer(ds, timespec, 0.25))
#	doalgo(lambda ds: UpdateAllDelayed(ds, total_time_budget_seconds=timespec))
#	doalgo(lambda ds: DelayedRetrainer(ds, total_time_budget_seconds=timespec))

#	for m in all_method_names:
#	for m in selected_method_names:
#		doalgo(lambda ds: DelayedRetrainer(ds, method_list=[m], total_time_budget_seconds=timespec))
	for m in all_method_names:
		doalgo(lambda ds: DelayedRetrainer(ds, total_time_budget_seconds=timespec, method_mod=m))

def doallalgos():
#	doalgos(7200)
	doalgos(3600)
#	doalgos(60)
#	doalgos(900)
#	doalgos(5400)
#	doalgos(7200)
#	doalgos(4500)
#	doalgos(6300)


if __name__ == '__main__':
	doallalgos()
#	manual_test()
#	doallalgos()
#	ds = processdata.Dataset("poker-hand-subset.data", 0)



