# ADR-graph
Predicting adverse drug reactions from a knowledge graph

# Description
Adverse drug reactions (ADRs) are a significant health risk, and a challenge to accurate cost-benefit considerations for drug treatment. This project uses a graph publicly available drug knowledge to predict possible additional (currently unknown) ADRs for marketed drugs. Predictions are made by a model trained on the known causes of each ADR. These models are binary classifiers used to place new edges between Drugs and ADRs in the graph based on predictive nodes identified in the network and optimised by an iterative weighting procedure.

# Acknowledgements
This work is funded by the National Institute for Health Research (NIHR) Biomedical Research Centre at South London and Maudsley NHS Foundation Trust and King’s College London.
