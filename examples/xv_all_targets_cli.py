# =============================================================================
# Example utility script around the EdgePrediction library
# makes a basic CLI for testing
# =============================================================================

## run cross validation for all targets in the graph

import json, sys
import xv_parallel as xv
import EdgePrediction

#usage all_targets_cli [cores] [start] [stop] [require all predictors (1 or 0)]

cores = int(sys.argv[1])
start = int(sys.argv[2])
stop = int(sys.argv[3])
require_all = bool(int(sys.argv[4]))

# =============================================================================
# this part will need to be updated to load project-specific data
# =============================================================================

#number of CPU cores to use. On your laptop probably 1 or 2
#on a rosalind instance use all cores. Don't set to more than number of actual cores.
#set in command line

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


# =============================================================================
# this part is likely to stay the same between projects
# =============================================================================

#start index in targets. 
#generally will be 0 when processing all but may need to change if you want to
#work on smaller chunks 
  
ep_conf = {'path': path, 'files': files, 
		   'to_predict': to_predict, 
		   'optimisation_method': "graph_sparse",
		   'require_all_predictors': require_all
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
print "[LOG] processing from %s to %s with %s targets" % (start, stop, stop_max)

xv_results = xv.run_parallel(start, stop, cores, targets, ep_conf)
output = {'xv': xv_results, 'meta': { 'ep_conf': ep_conf, 'files': files, 'file_path': path}}

out_fname = 'xv_%s_%s_%s.txt' % ("all_targets", start, stop)
with open(out_fname,'w') as f:
	json.dump(output, f)	

