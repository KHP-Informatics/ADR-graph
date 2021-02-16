# Documentation for the EdgePrediction library

This repository contains a Python implementation of the knowledge graph edge prediction algorithm described in Bean et al. 2017, and the input drug knowledge graph used in that paper. The algorithm is a general binary classifier that leans a model to predict new members of a given class within the training data. 

# Install

The package is available through pip:

```
pip install edgeprediction
```

# Contents:

* [Introduction](docs/markdown/IntroductionDoc.md)

* [Dependencies](docs/markdown/DependenciesDoc.md)

  * Dependencies list


* [Usage example](docs/markdown/ExampleUseDoc.md)

  * Initial setup

  * Input data format

  * Load data

  * Prepare to run prediction algorithm

  * Run prediction algorithm

* [EdgePrediction class documentation](docs/markdown/EdgePredictionDoc.md)

* [Objective class documentation](docs/markdown/ObjectiveDoc.md)

* [Utils class documentation](docs/markdown/UtilsDoc.md)

* [Contributing](docs/markdown/ContributingDoc.md)

# Acknowledgements
This work is funded by the National Institute for Health Research (NIHR) Biomedical Research Centre at South London and Maudsley NHS Foundation Trust and King’s College London.

The publicly available drug data as used in Bean et al. 2017 was collected from DrugBank (www.drugbank.ca) and SIDER (http://sideeffects.embl.de).

# Documentation and testing
Documentation is built with sphinx from docs_templates with sphinx>=v3.4.3
```
sphinx-build -M markdown ./ ../docs
```

Testing with pytest
```
python -m pytest tests
```