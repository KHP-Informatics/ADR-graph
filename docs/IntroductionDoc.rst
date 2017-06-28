Introduction
************

The EdgePrediction library is a Python machine learning library for
knowledge graph completion, described and applied to the prediction of
adverse drug reactions (ADRs) in Bean et al. 2017. The algorithm is
based on using Fisher’s exact test for enrichment to identify
properties of positive examples in the training data that are likely
to be predictive. Using these features, a grid search of parameters
determines feature weights that maximise the objective function
(Louden’s J statistic). This trained model predicts edges that are
missing from the graph.
