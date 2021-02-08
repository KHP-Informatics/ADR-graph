###
# Edge prediction algorithm - objective functions
# Developer: Dan Bean, King's College London
# Contact: daniel.bean@kcl.ac.uk
###

import numpy as np

class Objective:
	"""Calculate several standard metrics that can be used as an objective function.

	For a given set of scores and known source nodes from the graph, this method calculates several
	standard metrics: F0.5, F1, F2, J, accuracy, precision, recall, false discovery rate, false positive rate,
	contingency table. These metrics and the hits from the scores according to the threshold are returned,
	with the hits also broken down in to new hits (not known in the graph) and known hits (edges already
	present in the graph).

	The intended use of this class is via the Objective.evaluate method, which will calculate and return the
	above properties for a given threshold. This function is used to evaluate possible thresholds for a sef of
	scores and find the optimium threshold to maximise the objective function in the EdgePrediction library.

	Parameters
	----------
	score : numpy float array
		The score for every source node in the graph. Must be in the same order as the 'known' parameter.

	known : numpy bool array
		For every source node, value is True if there is an edge of the type being predicted to the target node, False
		otherwise. Must be in the same order as the 'score' parameter.
		

	Attributes
	----------
	n_known : float
		The number of known source nodes in the graph, calculated from the known parameter

	n_total : float
		The total number of source nodes in the graph, calculated from the known parameter

	"""
	def __init__(self, score, known):
		self.score = score
		self.known = known
		self.n_known = float(np.sum(known))
		self.n_total = float(len(known))

	def evaluate(self, threshold):
		"""Calculate metrics and hits for a given threshold

		Parameters
		----------
		threshold : float
			The threshold applied to the source node scores to determine which nodes are hits. Any nodes with 
			score >= threshold are considered hits.

		Returns
		-------
		result : dict
			Contains all the calculated metrics, contingency table and lists of hits.
		"""
		is_hit = self.score >= threshold
		n_hits = float(np.sum(is_hit))
		tp, fp, tn, fn = self.contingency(threshold)
		n_new_predictions = n_hits - tp
		prec = self.precision(tp,fp,tn,fn)
		rec = self.recall(tp,fp,tn,fn)
		#print("tp: %s ,fp: %s ,tn: %s ,fn: %s" % (tp,fp,tn,fn))

		f1 =  self.f_beta(prec, rec)
		f05 = self.f_beta(prec, rec, 0.5)
		f2 = self.f_beta(prec, rec, 2)
		acc =  self.accuracy(tp,fp,tn,fn)
		j = self.youden_j(tp,fp,tn,fn)
		fdr = self.falseDiscoveryRate(tp,fp,tn,fn)
		falsePosRate = self.falsePositiveRate(tp,fp,tn,fn)

		contingency = {'tp':tp,'fp':fp,'tn':tn,'fn':fn}

		result = {'F1': f1,'F05':f05,'F2':f2, 'ACC': acc, 'J': j, 'PREC':prec, 'REC':rec, 'FDR':fdr, 'FPR': falsePosRate, 'hits_total':n_hits,'hits_new':n_new_predictions, 'hits_known':tp, 'is_hit': is_hit}
		result['contingency'] = contingency
		return result

	def contingency(self, threshold):
		"""Generate a contingency table

		Parameters
		----------
		threshold : float
			The threshold applied to the source node scores to determine which nodes are hits. Any nodes with 
			score >= threshold are considered hits.

		Returns
		-------
		tp, fp, tn, fn : int
			tuple of true positives, false positives, true negatives, false negatives
		"""
		is_hit = self.score >= threshold
		n_hits = float(np.sum(is_hit))
		
		tp = np.sum(self.known[is_hit])
		fp = n_hits - tp
		fn = np.sum(self.known[self.score < threshold])
		tn = self.n_total - tp - fp - fn
		return tp, fp, tn, fn


	def f_beta(self,prec, rec, beta = 1):
		"""F-beta statistic for any beta.

		'The effectiveness of retrieval with respect to a user who places beta times as much importance to recall as
		precision' - Van Rijsbergen, C. J. (1979). Information Retrieval (2nd ed.). Used to calculate F0.5, F1, F2.

		Parameters
		----------
		prec : float
			precision of the model

		rec : float
			recall of the model

		beta : int ; float
			beta parameter of F statistic, relative importance of recall over precision

		Returns
		-------
		f : float
			The F-beta statistic
		"""
		if prec == 0 or rec == 0:
			f = 0
		else:
			pxr = prec * rec
			paddr = (beta**2 * prec) + rec
			f = (1 + beta**2) * (pxr/paddr)
		return f


	def accuracy(self,tp,fp,tn,fn):
		"""Calculate model accuracy
		
		The proportion of all predictions (for the positive or negative class) from the model that are correct

		Parameters
		----------
		tp : int ; float
			True positives of model

		fp : int ; float
			False positives of model

		tn : int ; float
			True negatives of model

		fn : int ; float
			False negatives of model

		Returns
		-------
		acc : float
			Accuacy of model
		"""
		acc = (float(tp) + tn)/(tp+fp+tn+fn)
		return acc

	def youden_j(self,tp,fp,tn,fn):
		"""Youden's J statistic
		
		J = sensitivity + specificity - 1 = TP/(TP+FN) + TN/(TN+FP) - 1

		Parameters
		----------
		tp : int ; float
			True positives of model

		fp : int ; float
			False positives of model

		tn : int ; float
			True negatives of model

		fn : int ; float
			False negatives of model

		Returns
		-------
		j : float
			Youden's J statistic
		"""
		a = float(tp)/(tp + fn)
		b = float(tn)/(tn + fp)
		j = a + b - 1
		return j

	def precision(self,tp,fp,tn,fn):
		"""Calculate precision of model

		The proportion of all positives from the model that are true positives.

		Parameters
		----------
		tp : int ; float
			True positives of model

		fp : int ; float
			False positives of model

		tn : int ; float
			True negatives of model

		fn : int ; float
			False negatives of model

		Returns
		-------
		prec : float
			Precision of the model
		"""
		prec = float(tp)/(tp+fp)
		return prec

	def recall(self,tp,fp,tn,fn):
		"""Calculate recall of model

		The proportion of all positives in the population that are predicted positive by the model.

		Parameters
		----------
		tp : int ; float
			True positives of model

		fp : int ; float
			False positives of model

		tn : int ; float
			True negatives of model

		fn : int ; float
			False negatives of model

		Returns
		-------
		rec : float
			Recall of model
		"""
		rec = float(tp)/(tp+fn)
		return rec

	def falseDiscoveryRate(self,tp,fp,tn,fn):
		"""Calculate False Discovery Rate (FDR)

		The proportion of all positive predictions that are false positives.

		Parameters
		----------
		tp : int ; float
			True positives of model

		fp : int ; float
			False positives of model

		tn : int ; float
			True negatives of model

		fn : int ; float
			False negatives of model

		Returns
		-------
		rate : float
			The false discovery rate
		"""
		rate = float(fp)/(fp+tp)
		return rate

	def falsePositiveRate(self,tp,fp,tn,fn):
		"""Calculate the False Positive Rate (FPR)

		Proportion of of actual negatives that are predicted positive

		Parameters
		----------
		tp : int ; float
			True positives of model

		fp : int ; float
			False positives of model

		tn : int ; float
			True negatives of model

		fn : int ; float
			False negatives of model

		Returns
		-------
		rate : float
			The false positive rate
		"""
		
		rate = float(fp)/(tn+fp)
		return rate


