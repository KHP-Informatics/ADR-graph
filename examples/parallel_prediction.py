import xv_parallel as parallel
import json
start = 0
stop = 6
cores = 2

path = '../data'
files = [ "data.csv" ]
to_predict = 'HAS_SIDE_EFFECT'

ep_conf = {'path': path, 'files': files, 'to_predict': to_predict}
ep_conf['require_all_predictors'] = False
ep_conf['optimisation_method'] = "graph_sparse"
ep_conf['job'] = 'predict'
results = parallel.run_parallel(start, stop, cores, None, ep_conf)
if results:
    out_fname = 'predict_parallel_test_%s_%s.txt' % (start, stop)

    with open(out_fname,'w') as f:
        json.dump(results, f) 
else:
    print("results were saved per thread automatically")