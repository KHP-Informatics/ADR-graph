EdgePrediction class documentation
**********************************

class EdgePrediction.EdgePrediction(to_predict=None)

   Edge Prediction class

   Implements the aglorithm described in Bean et al. 2017

   All parameters have sensible defaults except to_predict, which must
   be specified by the user at some point before the algorithm can
   run.

   Parameters:
      * **min_weight** (*float*) – minimum allowed feature weight,
        default 0.0

      * **max_weight** (*float*) – maximum allowed feature weight is
        max_weight - step. Default 1.1 gives a  max feature weight is
        1.0 with the default step size 0.1

      * **step** (*float*) – step size used in parameter grid search
        for feature weights. Default 0.1

      * **to_predict** (*str*) – Type of edge to predict. Must
        exactly match an edge type in the graph and must be set by
        user before the prediction algorithm will run. default None.

      * **pval_significance_threshold** (*float*) – threshold
        applied to select features from the enrichment test. Features
        with p-value for enrichment (after multiple testing correction
        if used) < threshold are considered enriched. Default 0.05

      * **require_all_predictors** (*bool*) – Optional. If true, a
        model will only be trained for a given target if at least one
        predictor for each type of edge in the graph is found with the
        enrichment test (after applying multiple testing correction if
        used). Default True

      * **objective_function** (*str*) – Optional. This is
        parameterised to allow convenient extension. Only the J
        objective is used in Bean et al 2017. J means Youden’s J
        statistic, J = sensitivity + specificity - 1. Default is “J”,
        options are {‘J’, ‘F1’, ‘F2’, ‘F05’, ‘ACC’}.

      * **ties** (*str*) – Optional. Method used to select which set
        of weights to use where two sets give identical performance of
        the objective function. ‘first’ uses the set found first,
        ‘minL2norm’ uses L2 normalisation to prefer balanced weights.
        If two sets of weights give idential performance and have the
        same L2 normalisation, the weights found first are kept. This
        means the network_order is important for both methods. Default
        ‘minL2norm’, options are {‘first’, ‘minL2norm’}

      * **network_order** (*list*) – Optional. The order in which
        the features are iterated over. This can be important as
        ultimately any ties are broken by keeping the parameters found
        first in the search. The best order may not ultimately matter,
        and will be context-specific. In general it is recommended to
        specify the order so it will remain consistent. The default
        order is determined by the keys of the internal dict object,
        which is not guaranteed.

      * **randomise_folds** (*bool*) – Optional. Whether to
        randomise the order of items in the full dataset before
        splitting all items into train-test folds. If False, the folds
        will always be identical between runs. Default True

      * **correct_pval** (*str*) – Optional. Correct p-values from
        enrichment test for multiple testing. Currently setting
        anything other than “BH” will result in no correction being
        applied. ‘BH’ or None, default ‘BH’.

   Variables:
      * **graphs** (*dict*) – Internal representation of the input
        graph. There is one element per edge type. {‘graph’:
        igraph.Graph, ‘sourcenodes’: list, ‘sourcetype’: str,
        ‘targetnodes’: list, ‘targettype’: str}

      * **can_analyse** (*bool*) – Flag used to keep track of any
        conditions that mean the network cannot be analysed

      * **optimisation_method** (*str*) – Currently only “simple” is
        available, which implements the method of Bean et al. 2017.

   CSV_to_graph(fname, header, srcNameCol, srcTypeCol, tgtNameCol, tgtTypeCol, edgeTypeCol)

      Parse csv file to internal graph representation

      The parsed graph is stored internally in self.graphs and is not
      returned.

      Parameters:
         * **fname** (*str*) – Input file name or path

         * **header** (*bool*) – Does the input file have a header
           row?

         * **srcNameCol** (*str*) – Column in input file containing
           source node names. Zero-indexed.

         * **srcTypeCol** (*str*) – Column in input file containing
           source node type. Zero-indexed.

         * **tgtNameCol** (*str*) – Column in input file containing
           target node names. Zero-indexed.

         * **tgtTypeCol** (*str*) – Column in input file containing
           target node types. Zero-indexed.

         * **edgeTypeCol** (*str*) – Column in input file containing
           edge types. Zero-indexed.

      Returns:
         **bool** – True for success, False otherwise.

      Return type:
         bool

   L2norm(weights)

      Regluarisation of weights

      Parameters:
         **weights** (*list*) – Model parameters, weights of each
         feature.

      Returns:
         **Float** – L2 regularisation of the weights

      Return type:
         Float

   auc(x, y, reorder=False)

      Calculate AUC

      Credit to scipy.metrics

      Parameters:
         * **x** (*list*) –

         * **y** (*list*) –

         * **reorder** (*bool*) – reorder the data points according
           to the x axis and using y to break ties. Default False.

      Returns:
         **area** – The area under the curve

      Return type:
         float

   createWeightsGenerator()

      Generate weights for parameter grid search.

      Configured by the EdgePrediction object instance properties
      (min_weight, max_weight, step).

      Parameters:
         **None.** –

      Returns:
         **weights_generator** – Instance of a generator that returns
         all combinations of parameters in the specified range

      Return type:
         generator

   enrichment(grouped, n_known, n_other)

      Fisher’s exact test for enrichment to identify features
      (predictors)

      Parameters:
         * **grouped** (*dict*) – output from from
           self.groupSparseAdjacency

         * **n_known** (*int*) – number of source nodes with an edge
           to the target node

         * **n_other** (*int*) – number of source nodes without an
           edge to the target node

      Returns:
         **all_pvals** – Keys are edge types, values are numpy arrays.
         Array columns are [p, known_present, other_present,
         known_absent, other_absent ]

      Return type:
         dict

   filterNetworksByCommonSource()

      Delete source nodes that don’t have at least one edge of every
      type.

      Parameters:
         **self** (*object*) –

      Returns:
         Internal network representation is updated.

      Return type:
         None

   filterSparseAdjacency(pvals, ignore=None)

      Filter a sparse adjacency matrix, keeping only the target nodes
      that are significantly enriched

      Parameters:
         * **pvals** (*dict*) – Output from from self.enrichment,
           see return value for self.enrichment

         * **ignore** (*bool*) – name of the target node that that
           edges are predicted for, so it should removed from the
           enrichment calculation

      Returns:
         **all_filtered** – keys are edge types, values are
         {‘overlap’:list,’colnames’:list, ‘predictors’: list}
         ‘overlap’ : adjacency of each source node with all predictors
         ‘colnames’ : source nodes in the graph ‘predictors’ : all
         enriched predictor names

      Return type:
         dict

   findOptimumThreshold(score, known, calculate_auc=False)

      Set the prediction threshold according to the objective function

      The objective function is set by self.objective_function

      Parameters:
         * **score** (*dict*) – output from from self.score, keys
           are edge types, values are dicts keyed by source node name
           and values are scores.

         * **known** (*list*) – source nodes with an edge to the
           target of type self.to_predict

         * **calculate_auc** (*bool*) – Whether or not to calculare
           and return the AUC. Default True.

      Returns:
         **best** – Contains many standard metrics for the model, e.g.
         F1 score, AUC, precision, recall, which have predictable
         names. Important proporties of the output are: ‘threshold’ :
         cutoff value that maximises the objective function
         ‘unique_threshold’ : bool, true if the same performance can
         be achieved with at least one different threshold
         ‘hits_known’ : hits from the model that are already known in
         the input graph ‘hits_new’ : hits from the model that are not
         already known in the input graph ‘is_hit’ : bool list, hit
         status for every source node.

      Return type:
         dict

   getKnown(target)

      Convenience function to list all nodes with an edge to the
      target.

      self.to_predict must be set to a valid edge type.

      Parameters:
         **target** (*str*) – Node name.

      Returns:
         **list** – all nodes with an edge to the target

      Return type:
         list

   getScores(target, weights)

      Calculate the score for all source nodes for a given set of
      weights.

      Not used internally, but a convenient way to calculate the score
      distribution for an arbitrary set of weights to manually explore
      how the distribution varies with weight, or to visualise the
      score distributino with the trained model weights.

      Parameters:
         * **target** (*str*) – Target node name to predict edges of
           type self.to_predict for.

         * **weights** (*dict*) – Keys are edge types, values are
           weights

      Returns:
         **scores** – Keys are edge types, values are dicts keyed by
         source node name and values are scores.

      Return type:
         dict

   groupSparseAdjacency(target)

      Adjacency for all nodes with known edges to the target vs all
      others.

      Parameters:
         **target** (*str*) – The name of the target node that we’re
         predicting edges to.

      Returns:
         **grouped** – The grouped adjacency matrix. Each element of
         the dict is one type of edge in the network. The output is
         the full (sparse) matrix.

      Return type:
         dict

   k_fold(target, k, calculate_auc=False)

      Modified k-fold cross validation.

      This is a modidication of a standard k-fold cross validation. In
      this implementation, edges are deleted from the graph and a
      predictive model is then trained on this modified data.
      Therefore the test set is not entirely held out during training,
      instead it is included as true negative examples. The ability of
      the trained model to predict the deleted edges is determined in
      every fold.

      Parameters:
         * **target** (*str*) – Target node name to predict edges of
           type self.to_predict for.

         * **k** (*int*) – The number of folds.

         * **calculate_auc** (*bool*) – If true, the AUC is
           calculated and returned for each model. Default False.

      Returns:
         **all_folds** – Each item in the list is a dict. The result
         is the output of self.predict with additional properties.
         ‘left_out_predicted’ : which of the deleted edges was
         predicted ‘proportion_predicted’ : proportion of all deleted
         edges that was predicted

      Return type:
         list

   loo(target, calculate_auc=False)

      Leave-one-out cross validation

      In each iteration, a single edge from a source node to the
      target node is deleted. A predictive model is trained on this
      modified data to determine whether the model predicts the
      missing (deleted) edge.

      Parameters:
         * **target** (*str*) – Target node name to predict edges of
           type self.to_predict for

         * **calculate_auc** (*bool*) – If true, the AUC is
           calculated and returned for each model. Default False.

      Returns:
         **loo_results** – Keys are names of known source nodes in the
         graph. Values are the objective function performance and
         whether the deleted edge was predicted.

      Return type:
         dict

   normalisePredictorOverlap(filtered)

      Perform feature normalisation to range 0-1.

      The raw adjacencies for each feature are divided by the max
      value for that feature.

      Parameters:
         **filtered** (*dict*) – output of self.filterSparseAdjacency

      Returns:
         * **all_normalised** (*dict*) – Keys are edge types, values
           are dicts. The nexted dict is keyed by source node name and
           values are normalised adjacencies.

         * **all_overlap_max** (*dict*) – Keys are edge types,
           values are the max adjacency in for that edge type.

   predict(target, calculate_auc=False)

      Train a predictive model for a given target.

      Optimum parameters are found using a grid search.

      Parameters:
         * **target** (*str*) – target node name to predict edges of
           type self.to_predict for

         * **calculate_auc** (*bool*) – If True, the AUC is
           calculated and included in the output. Default False.

      Returns:
         **optimisation_result** – Predictions from the trained model
         and various standard metrics such as precision, recall, F1,
         etc. Output contains the model target and objective function
         so the results are self-describing. The most important
         proporties are: ‘all_hits’ : all hit source nodes from the
         model ‘new_hits’ : all hits from the model that are not known
         in the input graph ‘known_hits’ : all hits from the model
         that are known in the input graph ‘weights’ : dict of
         parameters in the trained model, keys are edge types
         ‘threshold’ : threshold of trained model

      Return type:
         dict

   predictAll(calculate_auc=False)

      Train predictive models for all target nodes.

      Train predictive model for all target nodes of edges with the
      type self.to_predict. Not all targets will necessarily results
      in models depending on whether any enriched features are
      identified, and on self.require_all_predictors. The results is
      the same as manually calling self.predict on each target, this
      function is for convenience.

      Parameters:
         **calculate_auc** (*bool*) – If true, the AUC is calculated
         and returned for each model. Default False.

      Returns:
         **all_results** – Keys are model target node names, values
         are the output of self.predict()

      Return type:
         dict

   preprocess()

      Automates the first two steps to prepare data for training loop

      Does not need to be called manually.

      Parameters:
         **self** (*object*) –

      Returns:
         Internal network representation is updated

      Return type:
         None

   score(overlaps)

      Calculate the final score for each source node from weighted
      features.

      Parameters:
         **overlaps** (*dict*) – output from
         self.weightPredictorOverlap, normalised and weighted features
         for each source node

      Returns:
         **scores** – Keys are edge types, values are dicts keyed by
         source node name and values are scores.

      Return type:
         dict

   sparseAdjacency()

      Efficient representation of sparse adjacency matrix.

      Updates self.graphs generated from csv input with a sparse
      adjacency matrix. Edges are stored in both directions: source to
      target (ST) and target to source (TS). The representation is a
      dict where keys are node names and values are sets of other
      nodes connected with an edge of each type. There is one dict per
      edge type in the input data.

      Parameters:
         **self** (*object*) –

      Returns:
         Internal network representation is updated.

      Return type:
         None

   weightPredictorOverlap(overlaps, weights)

      Multiply each feature by a weight.

      Parameters:
         * **overlaps** (*dict*) – output of
           normalisePredictorOverlap

         * **weights** (*dict*) – Keys are edge types, values are
           weights

      Returns:
         **weighted** – dict with same structure as input overlaps,
         but with all values multiplied by their respective weights

      Return type:
         dict
