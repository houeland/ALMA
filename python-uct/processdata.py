from __future__ import division
import collections
import cPickle
import json
import logging
import math
import os
import random

Classifier = collections.namedtuple("Classifier", "results timespent_seconds")
#Dataset = collections.namedtuple("Dataset", "name whichtime_seed")
all_dataset_names = [
	'agaricus-lepiota.data',
	'balance-scale.data',
	'car.data',
	'connect-4.data',
	'covtype-subset.data',
	'house-votes-84.data',
	'kddcup-subset.data',
	'kr-vs-kp.data',
	'monks-1.test',
	'monks-2.test',
	'monks-3.test',
	'nursery-casebase',
	'phishing-websites.data',
	'poker-hand-subset.data',
	'secom_processed.data',
	'Skin_NonSkin.txt',
	'soybean-large.data',
	'SPECTF.test',
	'ThoraricSurgery.data',
	'tic-tac-toe.data',
	'travel-casebase',
]

class Dataset(object):
	def __init__(self, name, whichtime_seed):
		self.name = name
		self.whichtime_seed = whichtime_seed
		try:
			self.length = len(readdatasetfile(self, "classifications.pickle"))
		except BaseException as e:
			print("meow", dir(e))
			pass

	def description(self):
		return "%s/%s" % (self.name, self.whichtime_seed)

	def __str__(self):
		return "Dataset[%s]" % self.description()

def jstods(js):
	return Dataset(name=js["dataset"], whichtime_seed=js["whichtime_seed"])

read_cache = {}

def clear_cache():
	read_cache.clear()

def readdatasetfile(ds, *args):
	filename = os.path.join("processed", ds.name, str(ds.whichtime_seed), *args)
	v = read_cache.get(filename)
	if not v:
		with open(filename) as f:
			v = cPickle.load(f)
			read_cache[filename] = v
	return v

def writedatasetfile(output, ds, *args):
	filename = os.path.join("processed", ds.name, str(ds.whichtime_seed), *args)
	try: os.makedirs(os.path.dirname(filename))
	except: pass
	with open(filename, 'w') as f:
		cPickle.dump(output, f)

def scan_datasets():
	total = 0
	yo = 0
	for line in open("dataset_orders.txt"):
		js = json.loads(line)
		ds = jstods(js)
		yo += 1
		total += len(line)
		if yo % 10 == 0: print total, yo
		classifications = [x["classification"] for x in js["data"]]
		writedatasetfile(classifications, ds, "classifications.pickle")
	print total, yo

def compute_results(correct, predicted):
	if len(predicted) == 0: return "failed"
	else: return [x == y for x, y in zip(correct, predicted)]

def compute_accuracy(results):
	if results == "failed": return "failed"
	else: return sum(results) / len(results)

def scan_output():
	total = 0
	yo = 0
	for line in open("model_outputs.txt"):
		js = json.loads(line)
		ds = jstods(js)
		if ds.whichtime_seed not in set([0, 1, 1000, 1001]): continue
		dataset = js["dataset"]
		classifier_name = js["classifier"]
		if 'weka.classifiers.meta.Stacking' in classifier_name: classifier_name = 'weka.classifiers.meta.Stacking'
		predictions = js["predictions"]
		seen_examples_count = js["seen_examples_count"]
		precomputed_accuracy = js["accuracy"]
		timespent_seconds = js["timespent_seconds"]
		classifications = readdatasetfile(ds, "classifications.pickle")
		results = compute_results(classifications[seen_examples_count:], predictions)
		recomputed_accuracy = compute_accuracy(results)
		print seen_examples_count, len(predictions), precomputed_accuracy, recomputed_accuracy, timespent_seconds
		c = Classifier(results=results, timespent_seconds=timespent_seconds)
		writedatasetfile(c, ds, "method", classifier_name, "seen_%d_examples.pickle" % seen_examples_count)
		yo += 1
		total += len(line)
	print total, yo

def get_datasets():
	for dsn in all_dataset_names:
		yield Dataset(dsn, 0)
		yield Dataset(dsn, 1)
		yield Dataset(dsn, 1000)
		yield Dataset(dsn, 1001)

if __name__ == '__main__':
	scan_datasets()
	scan_output()
