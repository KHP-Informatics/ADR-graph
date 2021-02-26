###
# Test for prediction library
###

#import the module
import EdgePrediction, json
#to test for significant enrichment
from scipy.stats import hypergeom

#if testing for enrichment, this function uses the hypergeometric distribution
#to test whether the observed number of successful predictions is likely by chance
def hyper_prob_at_least(pop_n, pop_true, draw_n, draw_true):
    #prob of at least h hits is i - cdf(h-1)
    prb = hypergeom.cdf(draw_true-1, pop_n, pop_true, draw_n)
    prb_h_plus = 1 - prb
    return prb_h_plus

#create a new instance
ep = EdgePrediction.EdgePrediction()

#load data from edge list
ep.CSV_to_graph(fname = '../data/data.csv')

#filter the source nodes in all graphs so that all the source nodes are common to all input graphs
#keep those nodes and all their edges
#create adjacency matrix
ep.preprocess()

#type of edge to predict
ep.to_predict = 'HAS_SIDE_EFFECT'

#the order can be important as if there are ties, the result found first is returned
#ties may make different predictions
ep.network_order = ['HAS_SIDE_EFFECT', 'DRUG_TARGETS', 'INDICATED_FOR']

#run 5-fold cross validation
target_name = "C0027849"
xv = ep.k_fold(target_name, 5)

#it may not be possible to train a model for every fold
built = [x for x in xv if x['model_built'] == True]

#test for significant enrichment vs random guessing
for result in built:
    result['n_deleted_predicted'] = sum(result['left_out_predicted'])
    pop_n = result['contingency']['tn'] + result['contingency']['fp']
    pop_true = result['n_known_test']
    draw_n = result['contingency']['fp']
    draw_true = result['n_deleted_predicted']
    prob = hyper_prob_at_least(pop_n, pop_true, draw_n, draw_true)
    result['prob'] = prob
    result['signif'] = prob < 0.05

is_significant = [x['signif'] for x in built]
print("{} folds could be trained".format(len(built)))
print("of these {} were significant at p < 0.05".format(sum(is_significant)))


