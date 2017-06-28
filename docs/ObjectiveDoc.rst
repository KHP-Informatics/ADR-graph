Objective class documentation
*****************************

class Objective.Objective(score, known)

   Calculate several standard metrics that can be used as an objective
   function.

   For a given set of scores and known source nodes from the graph,
   this method calculates several standard metrics: F0.5, F1, F2, J,
   accuracy, precision, recall, false discovery rate, false positive
   rate, contingency table. These metrics and the hits from the scores
   according to the threshold are returned, with the hits also broken
   down in to new hits (not known in the graph) and known hits (edges
   already present in the graph).

   The intended use of this class is via the Objective.evaluate
   method, which will calculate and return the above properties for a
   given threshold. This function is used to evaluate possible
   thresholds for a sef of scores and find the optimium threshold to
   maximise the objective function in the EdgePrediction library.

   Parameters:
      * **score** (*numpy float array*) – The score for every source
        node in the graph. Must be in the same order as the ‘known’
        parameter.

      * **known** (*numpy bool array*) – For every source node,
        value is True if there is an edge of the type being predicted
        to the target node, False otherwise. Must be in the same order
        as the ‘score’ parameter.

   Variables:
      * **n_known** (*float*) – The number of known source nodes in
        the graph, calculated from the known parameter

      * **n_total** (*float*) – The total number of source nodes in
        the graph, calculated from the known parameter

   accuracy(tp, fp, tn, fn)

      Calculate model accuracy

      The proportion of all predictions (for the positive or negative
      class) from the model that are correct

      Parameters:
         * **tp** (*int ; float*) – True positives of model

         * **fp** (*int ; float*) – False positives of model

         * **tn** (*int ; float*) – True negatives of model

         * **fn** (*int ; float*) – False negatives of model

      Returns:
         **acc** – Accuacy of model

      Return type:
         float

   contingency(threshold)

      Generate a contingency table

      Parameters:
         **threshold** (*float*) – The threshold applied to the source
         node scores to determine which nodes are hits. Any nodes with
         score >= threshold are considered hits.

      Returns:
         **tp, fp, tn, fn** – tuple of true positives, false
         positives, true negatives, false negatives

      Return type:
         int

   evaluate(threshold)

      Calculate metrics and hits for a given threshold

      Parameters:
         **threshold** (*float*) – The threshold applied to the source
         node scores to determine which nodes are hits. Any nodes with
         score >= threshold are considered hits.

      Returns:
         **result** – Contains all the calculated metrics, contingency
         table and lists of hits.

      Return type:
         dict

   f_beta(prec, rec, beta=1)

      F-beta statistic for any beta.

      ‘The effectiveness of retrieval with respect to a user who
      places beta times as much importance to recall as precision’ -
      Van Rijsbergen, C. J. (1979). Information Retrieval (2nd ed.).
      Used to calculate F0.5, F1, F2.

      Parameters:
         * **prec** (*float*) – precision of the model

         * **rec** (*float*) – recall of the model

         * **beta** (*int ; float*) – beta parameter of F statistic,
           relative importance of recall over precision

      Returns:
         **f** – The F-beta statistic

      Return type:
         float

   falseDiscoveryRate(tp, fp, tn, fn)

      Calculate False Discovery Rate (FDR)

      The proportion of all positive predictions that are false
      positives.

      Parameters:
         * **tp** (*int ; float*) – True positives of model

         * **fp** (*int ; float*) – False positives of model

         * **tn** (*int ; float*) – True negatives of model

         * **fn** (*int ; float*) – False negatives of model

      Returns:
         **rate** – The false discovery rate

      Return type:
         float

   falsePositiveRate(tp, fp, tn, fn)

      Calculate the False Positive Rate (FPR)

      Proportion of of actual negatives that are predicted positive

      Parameters:
         * **tp** (*int ; float*) – True positives of model

         * **fp** (*int ; float*) – False positives of model

         * **tn** (*int ; float*) – True negatives of model

         * **fn** (*int ; float*) – False negatives of model

      Returns:
         **rate** – The false positive rate

      Return type:
         float

   precision(tp, fp, tn, fn)

      Calculate precision of model

      The proportion of all positives from the model that are true
      positives.

      Parameters:
         * **tp** (*int ; float*) – True positives of model

         * **fp** (*int ; float*) – False positives of model

         * **tn** (*int ; float*) – True negatives of model

         * **fn** (*int ; float*) – False negatives of model

      Returns:
         **prec** – Precision of the model

      Return type:
         float

   recall(tp, fp, tn, fn)

      Calculate recall of model

      The proportion of all positives in the population that are
      predicted positive by the model.

      Parameters:
         * **tp** (*int ; float*) – True positives of model

         * **fp** (*int ; float*) – False positives of model

         * **tn** (*int ; float*) – True negatives of model

         * **fn** (*int ; float*) – False negatives of model

      Returns:
         **rec** – Recall of model

      Return type:
         float

   youden_j(tp, fp, tn, fn)

      Youden’s J statistic

      J = sensitivity + specificity - 1 = TP/(TP+FN) + TN/(TN+FP) - 1

      Parameters:
         * **tp** (*int ; float*) – True positives of model

         * **fp** (*int ; float*) – False positives of model

         * **tn** (*int ; float*) – True negatives of model

         * **fn** (*int ; float*) – False negatives of model

      Returns:
         **j** – Youden’s J statistic

      Return type:
         float
