###
# Edge prediction algorithm
# Developer: Dan Bean, King's College London
# Contact: daniel.bean@kcl.ac.uk
###

import sys, csv, json, igraph, itertools
import numpy as np
import scipy.stats as stats #for stats.fisher_exact
from .Objective import Objective
from .errors import InputError
from scipy import optimize
from io import open #for py2 compatibility

#for multiple testing correction
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector


class EdgePrediction:
	"""Edge Prediction class

	Implements the aglorithm described in Bean et al. 2017

	All parameters have sensible defaults except to_predict, which must be specified by the user at some point
	before the algorithm can run.

	Parameters
	----------
	min_weight : float
		minimum allowed feature weight, default 0.0

	max_weight : float
		maximum allowed feature weight is max_weight - step. Default 1.1 gives a  max feature weight is 1.0 with the 
		default step size 0.1
	
	step : float
		step size used in parameter grid search for feature weights. Default 0.1
	
	to_predict : str
		Type of edge to predict. Must exactly match an edge type in the graph and must be set by user
		before the prediction algorithm will run. default None.
	
	pval_significance_threshold : float
		threshold applied to select features from the enrichment test. Features with p-value for enrichment (after
		multiple testing correction if used) < threshold are considered enriched. Default 0.05
	
	require_all_predictors : bool
		Optional. If true, a model will only be trained for a given target if at least one predictor for each type of edge 
		in the graph is found with the enrichment test (after applying multiple testing correction if used). Default True
	
	objective_function : str
		Optional. This is parameterised to allow convenient extension. Only the J objective is used in Bean et al 2017.
		J means Youden's J statistic, J = sensitivity + specificity - 1. Default is "J", options are {'J', 'F1', 'F2', 'F05', 'ACC'}.
	
	ties : str
		Optional. Method used to select which set of weights to use where two sets give identical performance of the objective 
		function. 'first' uses the set found first, 'minL2norm' uses L2 normalisation to prefer balanced weights. If
		two sets of weights give idential performance and have the same L2 normalisation, the weights found first are
		kept. This means the network_order is important for both methods. Default 'minL2norm', options are {'first', 'minL2norm'}

	network_order : list
		Optional. The order in which the features are iterated over. This can be important as ultimately any ties are broken by keeping
		the parameters found first in the search. The best order may not ultimately matter, and will be context-specific. In
		general it is recommended to specify the order so it will remain consistent. The default order is determined by the 
		keys of the internal dict object, which is not guaranteed. 
	
	randomise_folds : bool
		Optional. Whether to randomise the order of items in the full dataset before splitting all items into train-test folds.
		If False, the folds will always be identical between runs. Default True

	correct_pval : str
		Optional. Correct p-values from enrichment test for multiple testing. Currently setting anything other than "BH" will
		result in no correction being applied. 'BH' or None, default 'BH'.


	Attributes
	----------
	graphs : dict 
		Internal representation of the input graph. There is one element per edge type. {'graph': igraph.Graph, 'sourcenodes': list, 
		'sourcetype': str, 'targetnodes': list, 'targettype': str}
	
	can_analyse : bool
		Flag used to keep track of any conditions that mean the network cannot be analysed
	
	optimisation_method : str
		Currently only "graph" is available, which implements the method of Bean et al. 2017. 

	"""
	def __init__(self, to_predict = None):
		
		self.min_weight = 0.0 
		self.max_weight = 1.1
		self.step = 0.1
		self.to_predict = None 
		
		self.pval_significance_threshold = 0.05
		self.require_all_predictors = True #only build a model if predictors are found in all input graphs 
		self.objective_function = "J"
		 
		self.ties = "minL2norm" #method to break tied weights
		self.randomise_folds = True #put known source nodes in random order before generating folds
		self.network_order = None

		#internal parameters
		self.graphs = {}
		self.can_analyse = True
		self.correct_pval = "BH"
		self.optimisation_method = "graph"

	def CSV_to_graph(self,fname, srcNameCol = "Source node name", 
                  srcTypeCol = "Source node type", tgtNameCol = "Target node name", 
                  tgtTypeCol = "Target node type", edgeTypeCol = "Relationship type", encoding='utf-8'):
		"""Parse csv file to internal graph representation
		
		The parsed graph is stored internally in self.graphs and is not returned. 

		Parameters
		----------
		fname : str
			Input file name or path. Must be a csv file with a header.

		srcNameCol : str
			Column in input file containing source node names.

		srcTypeCol : str
			Column in input file containing source node type. 

		tgtNameCol : str
			Column in input file containing target node names. 

		tgtTypeCol : str
			Column in input file containing target node types. 

		edgeTypeCol : str
			Column in input file containing edge types.
		
		encoding : str
			encoding type for input file e.g. utf-8, utf-8-sig

		Returns
		-------
		bool : bool 
			True for success, False otherwise.
		"""
		
		edge_types = {}
		with open(fname, 'r', encoding=encoding) as f:
			reader = csv.reader(f)
			header = next(reader, None)
			for line_raw in reader:
				line = dict(zip(header, line_raw))
				etype = line[edgeTypeCol]
				if not etype in edge_types:
					edge_types[etype] = set()
				edge_types[etype].add((line[srcNameCol], line[srcTypeCol], line[tgtNameCol], line[tgtTypeCol]))
		
		for etype in edge_types:
			sourcenodes = set()
			targetnodes = set()
			sourcetypes = set()
			targettypes = set()
			edges = set()
			for edge in edge_types[etype]:
				if edge[0] != "" and edge[2] != "":
					edges.add((edge[0], edge[2]))
					sourcenodes.add(edge[0])
					targetnodes.add(edge[2])
					sourcetypes.add(edge[1])
					targettypes.add(edge[3])
			if len(sourcetypes) > 1:
				raise InputError("too many source node types in input data")
			elif len(targettypes) > 1:
				raise InputError("too many target node types in input data")
			else:
				sourcetype = list(sourcetypes)[0]
				targettype = list(targettypes)[0]
			g = igraph.Graph(directed = True)
			nodes = list(sourcenodes) + list(targetnodes)
			g.add_vertices(nodes)
			nodeProp = [sourcetype]*len(sourcenodes) + [targettype]*len(targetnodes)
			g.vs['type'] = nodeProp
			g.add_edges(list(edges))
			data = {'graph': g, 'sourcenodes': list(sourcenodes), 'sourcetype': sourcetype, 'targetnodes': list(targetnodes), 'targettype': targettype}
			self.graphs[etype] = data
		return True

	def preprocess(self):
		"""Automates the first two steps to prepare data for training loop

		Does not need to be called manually.

		Parameters
		----------
		self : object

		Returns
		-------
		None : 
			Internal network representation is updated
		"""
		self.filterNetworksByCommonSource()
		self.sparseAdjacency()


	## filter all graphs, keeping only source nodes common to all graphs (with all of their edges)
	## filtering is not necessary for the algorithm, but the current implementation expects filtering and won't work otherwise
	def filterNetworksByCommonSource(self):
		"""Delete source nodes that don't have at least one edge of every type.

		Parameters
		----------
		self : object

		Returns
		-------
		None :
			Internal network representation is updated. 
		"""

		graph_names = list(self.graphs.keys())

		#get the common source nodes
		common_source_nodes = set(self.graphs[graph_names[0]]['sourcenodes'])
		if len(graph_names) > 1:
			for nw in graph_names[1:]:
				common_source_nodes = common_source_nodes.intersection(set(self.graphs[nw]['sourcenodes']))
		if len(common_source_nodes) == 0:
			self.can_analyse = False
			raise InputError("Input graph cannot be analysed, no source nodes are common to all input graphs")
			return False
		
		print("%s source nodes are common to all %s input graphs" % (len(common_source_nodes), len(graph_names)))
		self.n_source_nodes = len(common_source_nodes)

		for network_name in self.graphs:
			networkA = self.graphs[network_name]
			A_source_nodes = set(networkA['sourcenodes'])
			#get a list of the drug nodes in A that are not in the common set
			delete_from_A = A_source_nodes.difference(common_source_nodes)

			#get the indices of these nodes in this graph
			A_graph_del_idx = [v.index for v in networkA['graph'].vs() if v['name'] in delete_from_A]

			#delete these nodes
			print("deleting %s nodes that don't overlap between networks" % len(A_graph_del_idx))
			networkA['graph'].delete_vertices(A_graph_del_idx)

			#check for nodes that now have degree zero
			networkA['graph'].vs['degree'] = networkA['graph'].degree()
			A_graph_del_idx = [v.index for v in networkA['graph'].vs() if v['degree'] == 0]
			#delete these nodes
			print("deleting another %s nodes that now have degree zero" % len(A_graph_del_idx))
			networkA['graph'].delete_vertices(A_graph_del_idx)

			#reset sourcenodes and targetnodes
			networkA['sourcenodes'] = networkA['graph'].vs(type_eq=networkA['sourcetype'])['name']
			networkA['targetnodes'] = networkA['graph'].vs(type_eq=networkA['targettype'])['name']

		#the network objects are mutable, so they'll be modified in the calling scope
		#return None so that this is clear
		return None

	def sparseAdjacency(self):
		"""Efficient representation of sparse adjacency matrix.

		Updates self.graphs generated from csv input with a sparse adjacency matrix. Edges are stored in both directions:
		source to target (ST) and target to source (TS). The representation is a dict where keys are node names and values are
		sets of other nodes connected with an edge of each type. There is one dict per edge type in the input data.

		Parameters
		----------
		self : object

		Returns
		-------
		None : 
			Internal network representation is updated.
		"""
		for network_name in self.graphs:

			#map graph ids to node names
			graph_names = self.graphs[network_name]['graph'].vs()['name']
			graph_idToName = dict(zip(range(0,len(graph_names)), graph_names))
			
			adj_TS = {}
			for t in self.graphs[network_name]['targetnodes']:
				adj = self.graphs[network_name]['graph'].neighbors(t, "IN")
				adj_TS[t] = set([graph_idToName[x] for x in adj])


			adj_ST = {}
			for s in self.graphs[network_name]['sourcenodes']:
					adj = self.graphs[network_name]['graph'].neighbors(s, "OUT")
					adj_ST[s] = set([graph_idToName[x] for x in adj])

			self.graphs[network_name]['ST'] = adj_ST
			self.graphs[network_name]['TS'] = adj_TS

	def groupSparseAdjacency(self, target):
		"""Adjacency for all nodes with known edges to the target vs all others.

		Parameters
		----------
		target : str
			The name of the target node that we're predicting edges to.

		Returns
		-------
		grouped : dict
			The grouped adjacency matrix. Each element of the dict is one type of edge in the network.
			The output is the full (sparse) matrix.
		"""
		known = self.graphs[self.to_predict]['TS'][target]
		grouped = {}
		for network_name in self.graphs:
			sparseAdj = self.graphs[network_name]['TS']
			nrows = len(sparseAdj.keys())
			output_matrix = np.zeros(shape = (nrows, 2), dtype=int)
			rownames = sparseAdj.keys()
			known_adj = [len(known.intersection(sparseAdj[x])) for x in rownames]
			other_adj = [len(sparseAdj[x]) - len(known.intersection(sparseAdj[x])) for x in rownames]
			output_matrix[:,0] = known_adj
			output_matrix[:,1] = other_adj
			grouped[network_name] = {'matrix':output_matrix, 'rownames':rownames}

		return grouped

	def filterSparseAdjacency(self, pvals, ignore = None):
		"""Filter a sparse adjacency matrix, keeping only the target nodes that are significantly enriched

		Parameters
		----------
		pvals : dict
			Output from from self.enrichment, see return value for self.enrichment

		ignore : bool
			name of the target node that that edges are predicted for, so it should removed from the enrichment calculation

		Returns
		-------
		all_filtered : dict
			keys are edge types, values are {'overlap':list,'colnames':list, 'predictors': list}
			'overlap' : adjacency of each source node with all predictors
			'colnames' : source nodes in the graph
			'predictors' : all enriched predictor names
		"""
		all_filtered = {}
		for network_name in self.graphs:
			predictors = pvals[network_name][:,0] < self.pval_significance_threshold
			rownames = np.array(list(self.graphs[network_name]['TS'].keys()))
			if ignore != None and network_name == self.to_predict:
				if ignore in rownames:
					predictors[rownames == ignore] = False
					print("ignoring %s as a predictor in network %s" % (ignore, network_name))
				else:
					print("WARNING %s not found in row names for network %s, continuing" % (ignore, network_name))
			predictors = set(rownames[predictors])
			colnames = list(self.graphs[network_name]['ST'].keys())
			overlaps = [len(predictors.intersection(self.graphs[network_name]['ST'][x])) for x in colnames]

			all_filtered[network_name] = {'overlap':overlaps,'colnames':colnames, 'predictors': predictors}
		return all_filtered

	def enrichment(self, grouped, n_known, n_other):
		"""Fisher's exact test for enrichment to identify features (predictors)

		Parameters
		----------
		grouped : dict 
			output from from self.groupSparseAdjacency

		n_known : int
			number of source nodes with an edge to the target node

		n_other : int
			number of source nodes without an edge to the target node

		Returns
		-------
		all_pvals : dict
			Keys are edge types, values are numpy arrays. Array columns are [p, known_present, other_present, known_absent, other_absent ]
		"""
		all_pvals = {}
		for network_name in grouped:
			nrows = grouped[network_name]['matrix'].shape[0]
			pvals = np.zeros(shape=(nrows,5))
			for i in range(nrows):
				known_present = grouped[network_name]['matrix'][i,0]
				other_present = grouped[network_name]['matrix'][i,1]
				known_absent = n_known - known_present
				other_absent = n_other - other_present
				#                  known          other
				# edge present [known_present, other_present],
				# edge absent  [known_absent,  other_absent ]
				#
				odds, p = stats.fisher_exact([ [known_present, other_present], [known_absent, other_absent] ], alternative = "greater")
				pvals[i,:] = [p, known_present, other_present, known_absent, other_absent ]
			if self.correct_pval == "BH":
				r_stats = importr('stats')
				p_list = pvals[:,0].tolist()
				p_adjust = r_stats.p_adjust(FloatVector(p_list), method = self.correct_pval)
				pvals[:,0] = list(p_adjust)
			else:
				print("WARNING - NOT correcting p-values for multiple comparisons")
			all_pvals[network_name] = pvals
		return all_pvals

	def createWeightsGenerator(self, min_weight = None, max_weight = None, step = None):
		"""Generate weights for parameter grid search.

		All parameters are required. They must be set on self for some omtimisation methods (min_weight, max_weight, step). 

		Parameters
		----------
		min_weight : float
			lower bound of search space
		max_weight : float
			upper bound of search space, should be intended bound + step
		step : float
			granularity of search space

		Returns
		-------
		weights_generator : generator
			Instance of a generator that returns all combinations of parameters in the specified range
		"""
			
		weights_generator = itertools.product(np.arange(min_weight, max_weight, step), repeat = len(self.graphs))
		return weights_generator
		
	
	def getKnown(self, target):
		"""Convenience function to list all nodes with an edge to the target.

		self.to_predict must be set to a valid edge type.

		Parameters
		----------
		target: str
			Node name.

		Returns
		-------
		list : list
			all nodes with an edge to the target
			returns empty list if the target is found but has no edges or is not found
		"""
		if target in self.graphs[self.to_predict]['TS']:
			return list(self.graphs[self.to_predict]['TS'][target])
		return []

	def normalisePredictorOverlap(self, filtered):
		"""Perform feature normalisation to range 0-1.

		The raw adjacencies for each feature are divided by the max value for that feature.

		Parameters
		----------
		filtered: dict 
			output of self.filterSparseAdjacency

		Returns
		-------
		all_normalised : dict
			Keys are edge types, values are dicts. The nexted dict is keyed by source node name and values are normalised
			adjacencies.

		all_overlap_max : dict
			Keys are edge types, values are the max adjacency in for that edge type.
		"""
		all_normalised = {}
		all_overlap_max = {}
		for network_name in filtered:
			norm = {}
			overlap_max = float(max(filtered[network_name]['overlap']))
			all_overlap_max[network_name] = overlap_max
			if overlap_max == 0:
				overlap_max = 1 #all values must = 0, divide by 1 so unchanged but still keyed by node name
			for i in range(len(filtered[network_name]['colnames'])):
				norm[filtered[network_name]['colnames'][i]] = filtered[network_name]['overlap'][i]/overlap_max
			all_normalised[network_name] = norm
		return all_normalised, all_overlap_max

	def weightPredictorOverlap(self, overlaps, weights):
		"""Multiply each feature by a weight.

		Parameters
		----------
		overlaps : dict 
			output of normalisePredictorOverlap

		weights : dict
			Keys are edge types, values are weights

		Returns
		-------
		weighted : dict
			dict with same structure as input overlaps, but with all values multiplied by their respective weights
		"""
		weighted = {}
		for network_name in overlaps:
			weighted[network_name] = {}
			for node_name in overlaps[network_name]:
				weighted[network_name][node_name] = overlaps[network_name][node_name] * weights[network_name]
		return weighted

	def score(self, overlaps, breakdown=False):
		"""Calculate the final score for each source node from weighted features.

		Parameters
		----------
		overlaps : dict 
			output from self.weightPredictorOverlap, normalised and weighted features for each source node

		breakdown : bool 
			if True, also return the breakdown of score contribution by network for each node

		Returns
		-------
		result : dict
			scores - Keys are edge types, values are dicts keyed by source node name and values are scores.
			breakdown - if breakdown=True, a dict of {node: {normalised contribution of each network type to score}}, otherwise an empty dict. 
		"""
		networks = list(overlaps.keys())
		nodes = overlaps[networks[0]].keys()
		scores = {}
		brkd = {}
		for n in nodes:
			scores[n] = 0
			per_nw = {}
			for nw in networks:
				scores[n] += overlaps[nw][n]
				per_nw[nw] = overlaps[nw][n] #cache in case breakdown needed
			if breakdown:
				brkd[n] = {nw: per_nw[nw]/scores[n] for nw in networks}
		result = {'scores': scores, 'breakdown': brkd}
		return result

	def findOptimumThreshold(self, score, known, calculate_auc = False):
		"""Set the prediction threshold according to the objective function

		The objective function is set by self.objective_function

		Parameters
		----------
		score : dict 
			output from from self.score, keys are edge types, values are dicts keyed by source node name and values are scores.

		known : list
			source nodes with an edge to the target of type self.to_predict

		calculate_auc : bool
			Whether or not to calculare and return the AUC. Default True.

		Returns
		-------
		best : dict
			Contains many standard metrics for the model, e.g. F1 score, AUC, precision, recall, which have predictable names.
			Important proporties of the output are:
			'threshold' : cutoff value that maximises the objective function
			'unique_threshold' : bool, true if the same performance can be achieved with at least one different threshold
			'hits_known' : hits from the model that are already known in the input graph
			'hits_new' : hits from the model that are not already known in the input graph
			'is_hit' : bool list, hit status for every source node.

		"""
		node_names = list(score.keys())
		score = np.array([score[x] for x in score])
		known = np.in1d(node_names, known, assume_unique=True)
		thresholds = np.unique(score)
		placeholder = {}
		method = self.objective_function
		placeholder[method] = -1 #all current objective functions are in range 0-1 so the first result always replaces the placeholder
		best_performance = [placeholder]
		obj = Objective(score, known)
		if calculate_auc:
			x = [] #FPR = FP/(population N)
			y = [] #TPR TP/(population P)
			pop_pos = float(np.sum(known))
			pop_neg = float(len(known) - pop_pos)
		for t in thresholds:
			result = obj.evaluate(t)
			if result[method] > best_performance[0][method]:
				result['unique_threshold'] = True
				result['threshold'] = t
				best_performance = [result] #if there's a new best, reset to an array of one element
			elif result[method] == best_performance[0][method]: 
				#keep track of different thresholds that give equivalent results
				result['threshold'] = t
				result['unique_threshold'] = False
				best_performance.append(result)
				best_performance[0]['unique_threshold'] = False 
			#auc
			if calculate_auc:
				x.append(result['contingency']['fp']/ pop_neg)
				y.append(result['contingency']['tp'] / pop_pos)
		best = best_performance[0] #only return one result even if there are ties
		best['all_hits'] = set(itertools.compress(node_names, best['is_hit']))
		del best['is_hit']
		best['auc'] = "NA"
		if calculate_auc:
			x = np.array(x)
			y = np.array(y)
			best['auc'] = self.auc(x, y, True)
		return best

	def L2norm(self, weights):
		"""Regluarisation of weights

		Parameters
		----------
		weights : list
			Model parameters, weights of each feature.

		Returns
		-------
		Float : Float
			L2 regularisation of the weights
		"""
		return sum([x**2 for x in weights])

	def predict(self, target, calculate_auc = False, return_scores=False):
		"""Train a predictive model for a given target.

		Optimum parameters are found using a grid search.

		Parameters
		----------
		target : str
			target node name to predict edges of type self.to_predict for

		calculate_auc: bool
			If True, the AUC is calculated and included in the output. Default False.

		return_scores: bool
			If True, also returns the score and normalised contribution of each edge type to the total score for each node
		
		Returns
		-------
		optimisation_result : dict
			Predictions from the trained model and various standard metrics such as precision, recall, F1, etc. 
			Output contains the model target and objective function so the results are self-describing. The most
			important proporties are:
			'all_hits' : all hit source nodes from the model
			'new_hits' : all hits from the model that are not known in the input graph
			'known_hits' : all hits from the model that are known in the input graph
			'weights' : dict of parameters in the trained model, keys are edge types
			'threshold' : threshold of trained model
			'scores' : if called with return_scores=True, a dict with score and breakdown for every node (output of self.getScores with optimised weights)
		"""
		if self.to_predict == None:
			raise InputError("Cannot run prediction, no target set for EdgePrediction.to_predict")
		if self.can_analyse == False:
			raise InputError("Cannot run prediction, input graph cannot be analysed.")
		known = self.getKnown(target)
		if len(known) == 0:
			msg = "Cannot run prediction: no edges for target or target not in network: {}".format(target)
			raise InputError(msg)
			# return {'error': "no edges or not in graph"}
		known_set = set(known)
		n_known = len(known)
		n_other = self.n_source_nodes - n_known
		grouped = self.groupSparseAdjacency(target)
		enrichment_pvals = self.enrichment(grouped, n_known, n_other)
		filtered = self.filterSparseAdjacency(enrichment_pvals, target)
		normalised, overlap_max = self.normalisePredictorOverlap(filtered)

		if self.require_all_predictors:
			no_predictor_overlap = [x for x in overlap_max if overlap_max[x] == 0]
			if len(no_predictor_overlap) > 0:
				print("self.require_all_predictors is %s and %s/%s networks have 0 predictor overlap" % (self.require_all_predictors, len(no_predictor_overlap), len(normalised)))
				print("not optimising a model for %s" % (target))
				optimisation_result = {}
				optimisation_result['model_target'] = target
				optimisation_result['model_built'] = False
				return optimisation_result

		if self.optimisation_method == "graph":
			weights_generator = self.createWeightsGenerator(self.min_weight, self.max_weight, self.step)
			if self.network_order == None:
				network_names = list(normalised.keys())
			else:
				network_names = self.network_order
			optimisation_result = {}
			optimisation_result[self.objective_function] = -1 #all current objectives are in range 0-1 so the first result always replaces this
			for weights in weights_generator:
				weights = dict(zip(network_names, weights))
				weighted = self.weightPredictorOverlap(normalised, weights)
				scores = self.score(weighted)['scores']
				best_threshold_for_weights = self.findOptimumThreshold(scores, known, calculate_auc)
				if best_threshold_for_weights[self.objective_function] > optimisation_result[self.objective_function]:
					optimisation_result = best_threshold_for_weights
					optimisation_result['weights'] = weights
					optimisation_result['count_equivalent_weights'] = 1
				elif best_threshold_for_weights[self.objective_function] == optimisation_result[self.objective_function]:
					optimisation_result['count_equivalent_weights'] += 1 #equivalent in terms of objective function score, not necessarily predictions made
					if self.ties == "minL2norm":
						lnorm_best = self.L2norm(optimisation_result['weights'].values())
						lnorm_now = self.L2norm(weights.values())
						if lnorm_now < lnorm_best:
							optimisation_result = best_threshold_for_weights
							optimisation_result['weights'] = weights
							optimisation_result['count_equivalent_weights'] = 1
			optimisation_result['known_hits'] = known_set.intersection(optimisation_result['all_hits'])
			optimisation_result['new_hits'] = optimisation_result['all_hits'].difference(known_set)
			optimisation_result['all_hits'] = list(optimisation_result['all_hits'])
			optimisation_result['new_hits'] = list(optimisation_result['new_hits'])
			optimisation_result['known_hits'] = list(optimisation_result['known_hits'])
		
		elif self.optimisation_method == "graph_sparse":
			#first search at half the density
			#print("coarse search")
			weights_generator = self.createWeightsGenerator(self.min_weight, self.max_weight, self.step * 2)
			optimisation_result = self.evaluate_weights(weights_generator, normalised, known, known_set, calculate_auc)
			#starting from the current best, test each weight +/- fine grain step
			start_weights = optimisation_result['weights']
			fine_weights = []
			
			if self.network_order == None:
				network_names = list(normalised.keys())
			else:
				network_names = self.network_order
			
			for x in network_names:
				mid = start_weights[x]
				out = [mid]
				low = mid - self.step
				high = mid + self.step
				if low >= self.min_weight:
					out.append(low)
				if high <= self.max_weight:
					out.append(high)
				out.sort()
				fine_weights.append(out)
			#print(fine_weights)
			weights_generator = itertools.product(*fine_weights)
			#print("fine search")
			optimisation_result = self.evaluate_weights(weights_generator, normalised, known, known_set, calculate_auc)
			
			
		else:
			print("No method definied to handle the optimisation method %s" % self.optimisation_method)
			raise NameError(self.optimisation_method)
		
		optimisation_result['optimisation_method'] = self.optimisation_method
		optimisation_result['objective'] = self.objective_function
		optimisation_result['model_target'] = target
		optimisation_result['model_built'] = True
		optimisation_result['model_edge_type'] = self.to_predict
		optimisation_result['predictors'] = {}
		for network_name in filtered:
			optimisation_result['predictors'][network_name] = list(filtered[network_name]['predictors'])
		if return_scores:
			optimisation_result['scores'] = self.getScores(target, optimisation_result['weights'], breakdown=True)

		return optimisation_result
	
	def evaluate_weights(self, weights_generator, normalised, known, known_set, calculate_auc):
		if self.network_order == None:
				network_names = list(normalised.keys())
		else:
			network_names = self.network_order
		optimisation_result = {}
		optimisation_result[self.objective_function] = -1 #all current objectives are in range 0-1 so the first result always replaces this
		for weights in weights_generator:
			weights = dict(zip(network_names, weights))
			#print(weights)
			weighted = self.weightPredictorOverlap(normalised, weights)
			scores = self.score(weighted)['scores']
			best_threshold_for_weights = self.findOptimumThreshold(scores, known, calculate_auc)
			if best_threshold_for_weights[self.objective_function] > optimisation_result[self.objective_function]:
				optimisation_result = best_threshold_for_weights
				optimisation_result['weights'] = weights
				optimisation_result['count_equivalent_weights'] = 1
			elif best_threshold_for_weights[self.objective_function] == optimisation_result[self.objective_function]:
				optimisation_result['count_equivalent_weights'] += 1 #equivalent in terms of objective function score, not necessarily predictions made
				if self.ties == "minL2norm":
					lnorm_best = self.L2norm(optimisation_result['weights'].values())
					lnorm_now = self.L2norm(weights.values())
					if lnorm_now < lnorm_best:
						optimisation_result = best_threshold_for_weights
						optimisation_result['weights'] = weights
						optimisation_result['count_equivalent_weights'] = 1
		optimisation_result['known_hits'] = known_set.intersection(optimisation_result['all_hits'])
		optimisation_result['new_hits'] = optimisation_result['all_hits'].difference(known_set)
		optimisation_result['all_hits'] = list(optimisation_result['all_hits'])
		optimisation_result['new_hits'] = list(optimisation_result['new_hits'])
		optimisation_result['known_hits'] = list(optimisation_result['known_hits'])
		return optimisation_result

	def predictAll(self, calculate_auc=False):
		"""Train predictive models for all target nodes.

		Train predictive model for all target nodes of edges with the type self.to_predict. Not all targets
		will necessarily results in models depending on whether any enriched features are identified, and
		on self.require_all_predictors. The results is the same as manually calling self.predict on each 
		target, this function is for convenience.

		Parameters
		----------
		calculate_auc : bool
			If true, the AUC is calculated and returned for each model. Default False.

		Returns
		-------
		all_results : dict
			Keys are model target node names, values are the output of self.predict()
		"""
		if self.to_predict == None:
			raise InputError("Cannot run prediction, no target set for EdgePrediction.to_predict")
		if self.can_analyse == False:
			raise InputError("Cannot run prediction, input graph cannot be analysed.")
		all_results = {}
		all_targets = list(self.graphs[self.to_predict]['TS'].keys())
		n_targets = len(all_targets)
		n = 1
		for target in all_targets:
			print("%s (%s/%s)" % (target, n, n_targets))
			n += 1
			all_results[target] = self.predict(target, calculate_auc)
		return all_results

	def loo(self, target, calculate_auc = False):
		"""Leave-one-out cross validation

		In each iteration, a single edge from a source node to the target node is deleted. A predictive model
		is trained on this modified data to determine whether the model predicts the missing (deleted) edge.

		Parameters
		----------
		target : str
			Target node name to predict edges of type self.to_predict for

		calculate_auc : bool
			If true, the AUC is calculated and returned for each model. Default False.

		Returns
		-------
		loo_results : dict
			Keys are names of known source nodes in the graph. Values are the objective function performance and 
			whether the deleted edge was predicted.
		"""
		if self.to_predict == None:
			raise InputError("Cannot run prediction, no target set for EdgePrediction.to_predict")
		if self.can_analyse == False:
			raise InputError("Cannot run prediction, input graph cannot be analysed.")
		known = self.getKnown(target)
		target_node_id = self.graphs[self.to_predict]['graph'].vs.select(name_eq=target)
		loo_results = {}
		for k in known:
			#find the edge from this known source node to the target and delete it
			source_node_id = self.graphs[self.to_predict]['graph'].vs.select(name_eq=k)
			edge_to_delete = self.graphs[self.to_predict]['graph'].es.select(_between=(source_node_id, target_node_id))
			self.graphs[self.to_predict]['graph'].delete_edges(edge_to_delete)
			#update the master adjacency matrix
			self.sparseAdjacency()
			#run the prediction
			res = self.predict(target, calculate_auc)
			loo_results[k] = {'target':target, 'left_out_name':k, 'model_built': res['model_built']}
			if res['model_built']:
				ignored_is_hit = k in res['all_hits']
				loo_results[k]['was_predicted'] = ignored_is_hit
				loo_results[k]['objective_performance'] = res[self.objective_function]				

			#put the edge back
			self.graphs[self.to_predict]['graph'].add_edges([(source_node_id[0], target_node_id[0])])

		#update the master adjacency matrix so make sure it contains all edges again
		self.sparseAdjacency()
		return loo_results

	def k_fold(self, target, k, calculate_auc = False):
		"""Modified k-fold cross validation.

		This is a modidication of a standard k-fold cross validation. In this implementation, edges are deleted from 
		the graph and a predictive model is then trained on this modified data. Therefore the test set is not entirely 
		held out during training, instead it is included as true negative examples. The ability of the trained model 
		to predict the deleted edges is determined in every fold. 

		Parameters
		----------
		target : str
			Target node name to predict edges of type self.to_predict for.

		k : int
			The number of folds.

		calculate_auc : bool
			If true, the AUC is calculated and returned for each model. Default False.

		Returns
		-------
		all_folds : list
			Each item in the list is a dict. The result is the output of self.predict with additional properties.
			'left_out_predicted' : which of the deleted edges was predicted
			'proportion_predicted' : proportion of all deleted edges that was predicted
		"""
		#generate folds
		known = self.getKnown(target)
		if self.randomise_folds:
			#get known source nodes into random order
			np.random.shuffle(known)

		#number of edges to delete per fold
		edges_per_fold = int(len(known)/k)
		remainder = len(known) % k

		if edges_per_fold == 0:
			msg = "specified fold size %s is too large for %s with %s known sources" .format(k, target, len(known))
			if self.to_predict == None:
				raise InputError(msg)
			return False

		start = 0
		stop = edges_per_fold
		all_folds = []
		target_node_id = self.graphs[self.to_predict]['graph'].vs.select(name_eq=target)
		for fold in range(k):
			if fold < remainder:
				stop += 1

			#find and delete the edges between these source nodes and the target
			delete_this_fold = known[start:stop]
			deleted_source_ids = []
			for source_name in delete_this_fold:
				source_node_id = self.graphs[self.to_predict]['graph'].vs.select(name_eq=source_name)
				edge_to_delete = self.graphs[self.to_predict]['graph'].es.select(_between=(source_node_id, target_node_id))
				self.graphs[self.to_predict]['graph'].delete_edges(edge_to_delete)
				deleted_source_ids.append(source_node_id)
			#update the master adjacency matrix
			self.sparseAdjacency()
			#run the prediction
			res = self.predict(target, calculate_auc)
			fold_result = {'target':target, 'left_out': delete_this_fold, 'model_built': res['model_built']}
			fold_result['n_known_train'] = len(known) - len(delete_this_fold)
			fold_result['n_known_test'] = len(delete_this_fold)
			if res['model_built']:
				ignored_is_hit = []
				for source_name in delete_this_fold:
					ignored_is_hit.append(source_name in res['new_hits'])
				fold_result['left_out_predicted'] = ignored_is_hit
				fold_result['proportion_predicted'] = float(len([x for x in ignored_is_hit if x]))/len(ignored_is_hit)
				fold_result['objective_performance'] = res[self.objective_function]	
				fold_result['contingency'] = res['contingency']
			all_folds.append(fold_result)			

			#put the edges back
			for source_node_id in deleted_source_ids:
				self.graphs[self.to_predict]['graph'].add_edges([(source_node_id[0], target_node_id[0])])

			start = stop
			stop += edges_per_fold

		#update the master adjacency matrix so make sure it contains all edges again
		self.sparseAdjacency()
		return all_folds

	def auc(self, x, y, reorder=False):
		"""Calculate AUC

		Credit to scipy.metrics 
		
		Parameters
		----------
		x : list

		y : list

		reorder : bool
			reorder the data points according to the x axis and using y to break ties.
			Default False.

		Returns
		-------
		area : float
			The area under the curve
		"""
		direction = 1
		if reorder:
			
			order = np.lexsort((y, x))
			x, y = x[order], y[order]
		else:
			dx = np.diff(x)
			if np.any(dx < 0):
				if np.all(dx <= 0):
					direction = -1
				else:
					raise ValueError("Reordering is not turned on, and the x array is not increasing: %s" % x)

		area = direction * np.trapz(y, x)
		if isinstance(area, np.memmap):
			# Reductions such as .sum used internally in np.trapz do not return a
			# scalar by default for numpy.memmap instances contrary to
			# regular numpy.ndarray instances.
			area = area.dtype.type(area)
		return area

	def getScores(self,target,weights, breakdown = False):
		"""Calculate the score for all source nodes for a given set of weights.

		Not used internally, but a convenient way to calculate the score distribution for an arbitrary set of weights
		to manually explore how the distribution varies with weight, or to visualise the score distributino with the 
		trained model weights.

		Parameters
		----------
		target : str
			Target node name to predict edges of type self.to_predict for.

		weights : dict
			Keys are edge types, values are weights
		
		breakdown : bool
			If true, return a breakdown of the normalised contribution to the score of each node by network name

		Returns
		-------
		scores : dict
			scores - Keys are edge types, values are dicts keyed by source node name and values are scores.
			breakdown - if breakdown is true, keys are node names, values are dicts of {network name : contribution}, otherwise an empty dict.
		"""
		if self.to_predict == None or self.can_analyse == False:
			if self.to_predict == None:
				raise InputError("Cannot run prediction, no target set for EdgePrediction.to_predict")
			if self.can_analyse == False:
				raise InputError("Cannot run prediction, input graph cannot be analysed.")
		known = self.getKnown(target)
		n_known = len(known)
		n_other = self.n_source_nodes - n_known
		grouped = self.groupSparseAdjacency(target)
		enrichment_pvals = self.enrichment(grouped, n_known, n_other)
		filtered = self.filterSparseAdjacency(enrichment_pvals, target)
		normalised, overlap_max = self.normalisePredictorOverlap(filtered)

		if self.require_all_predictors:
			no_predictor_overlap = [x for x in overlap_max if overlap_max[x] == 0]
			if len(no_predictor_overlap) > 0:
				print("self.require_all_predictors is %s and %s/%s networks have 0 predictor overlap" % (self.require_all_predictors, len(no_predictor_overlap), len(normalised)))
				print("not optimising a model for %s" % (target))
				optimisation_result = {}
				optimisation_result['model_target'] = target
				optimisation_result['model_built'] = False
				return optimisation_result

		weighted = self.weightPredictorOverlap(normalised, weights)
		scores = self.score(weighted, breakdown)
		return scores

	def min_func(self, params, normalised, known, obj_method, network_names):
		#in development, do not use this
		threshold = params[0]
		weights_list = params[1:]

		weights = dict(zip(network_names, weights_list))
		weighted = self.weightPredictorOverlap(normalised, weights)
		scores = self.score(weighted)['scores']

		node_names = list(scores.keys())
		score = np.array([scores[x] for x in scores])
		known = np.in1d(node_names, known, assume_unique=True)

		obj = Objective(score, known)
		result = obj.evaluate(threshold)
		#print("step done with score", result[obj_method])
		return 1.0 - result[obj_method]
	
	def new_opt(self, target, method='nelder-mead', init_method='mid'):
		#in development, do not use this
		known = self.getKnown(target)
		known_set = set(known)
		n_known = len(known)
		n_other = self.n_source_nodes - n_known
		grouped = self.groupSparseAdjacency(target)
		enrichment_pvals = self.enrichment(grouped, n_known, n_other)
		filtered = self.filterSparseAdjacency(enrichment_pvals, target)
		normalised, overlap_max = self.normalisePredictorOverlap(filtered)
		if self.network_order == None:
				network_names = list(normalised.keys())
		else:
			network_names = self.network_order

		n_params = len(network_names)+1
		if init_method == 'mid':
			x0 = [0.5] * n_params
		if init_method == 'random':
			x0 = [np.random.rand() for x in range(n_params)]

		op = optimize.minimize(self.min_func, x0, args=(normalised, known, self.objective_function, network_names), method=method)
		return op