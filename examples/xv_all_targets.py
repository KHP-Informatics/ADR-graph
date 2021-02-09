# =============================================================================
# Example script run cross validation for all targets in the database
# =============================================================================

import json
import xv_parallel as xv
import EdgePrediction

# =============================================================================
# project-specific configs
# =============================================================================

#number of CPU cores to use. On your laptop probably 1 or 2
#Don't set to more than number of actual cores.
cores = 4

#name ofthe formatted risk data folder
path = 'my_data_formatted'

#name of each file you want to load from that folder.
#each additional dataset will make the training slower.
files = ["edges1.csv",
	"edges2.csv",
	"edges3.csv"
	]

#the name of the relatonship type you want to predict 
to_predict = 'my_interesting_edge'

#how many targets to process. Set to anything higher than the actual total
#to process all targets, e.g. 999999999
#any number less than the max possible means process targets up to this number
#and is useful for testing
stop = 99999999 


# =============================================================================
# this part is likely to stay the same between projects
# =============================================================================

#start index in targets. 
#generally will be 0 when processing all but may need to change if you want to
#work on smaller chunks 
start = 0
  
ep_conf = {'path': path, 'files': files, 
		   'to_predict': to_predict, 
		   'optimisation_method': "graph_sparse",
		   'require_all_predictors': True
		   }

ep = EdgePrediction.EdgePrediction()

#load data from edge list
for to_load in ep_conf['files']:
	fname = ep_conf['path'] + "/" + to_load
	ep.CSV_to_graph(fname)


ep.preprocess()

ep.to_predict = ep_conf['to_predict']
ep.objective_function = "J"

nw_order = ep.graphs.keys().sort()
ep.network_order = nw_order

ep.ties = "minL2norm"

targets = ep.graphs[ep.to_predict]['TS'].keys()
targets.sort() #otherwise keys won't necessarily be in the same order when another chunk is started

stop_max = len(targets)
stop = min(stop, stop_max) #so we definitely don't exceed possible size

xv_results = xv.run_parallel(start, stop, cores, targets, ep_conf)
output = {'xv': xv_results, 'meta': { 'ep_conf': ep_conf, 'files': files, 'file_path': path}}

out_fname = 'xv_%s_%s_%s.txt' % ("all_targets", start, stop)
with open(out_fname,'w') as f:
	json.dump(output, f)	