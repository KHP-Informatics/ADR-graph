## run risk prediction cross validation on multiple cores in parallel
## thanks https://sebastianraschka.com/Articles/2014_multiprocessing.html
import multiprocessing as mp
import EdgePrediction, json, time
import numpy as np

#ep_conf:
#     files []
#     path 'path/to/files'
#     to_predict 'edge type'
#     folds (optional, default 5)
#     network_order (optional, default is alphabetical)
#     

#https://github.com/hmallen/numpyencoder
class NumpyEncoder(json.JSONEncoder):
     """ Custom encoder for numpy data types """
     def default(self, obj):
          if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                         np.int16, np.int32, np.int64, np.uint8,
                         np.uint16, np.uint32, np.uint64)):

               return int(obj)

          elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
               return float(obj)

          elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
               return {'real': obj.real, 'imag': obj.imag}

          elif isinstance(obj, (np.ndarray,)):
               return obj.tolist()

          elif isinstance(obj, (np.bool_)):
               return bool(obj)

          elif isinstance(obj, (np.void)): 
               return None

          return json.JSONEncoder.default(self, obj)

def xv_chunk(start, stop, parallel = True, target_list = None, output=None, ep_conf = {}, out_fname='result'):
     chunk_t_start = time.time()
     #if target_list == None, use all targets. Else use the list provided.
     
     ep = EdgePrediction.EdgePrediction()

     #load data from edge list
     for to_load in ep_conf['files']:
          fname = ep_conf['path'] + "/" + to_load
          ep.CSV_to_graph(fname)
     
     #filter the source nodes in all graphs so that all the source nodes are common to all input graphs
     #keep those nodes and all their edges
     #create adjacency matrix
     ep.preprocess()
     
     #type of edge to predict
     ep.to_predict = ep_conf['to_predict']
     ep.objective_function = "J"
      
     #optimisation method
     ep.optimisation_method = ep_conf["optimisation_method"]
      
     #require all predictors
     ep.require_all_predictors = ep_conf["require_all_predictors"]
     
     #the order can be important as if there are ties, the result found first is returned
     #ties may make different predictions
     #ep.network_order = ['risk_gene', 'hpo_annotation', 'go_annotation', 'gene_ppi']
     if 'network_order' in ep_conf:
          ep.network_order = ep_conf['network_order']
     else:
          nw_order = list(ep.graphs.keys()).sort()
          ep.network_order = nw_order
     
     
     ep.ties = "minL2norm"

     
     if target_list == None:
          targets = list(ep.graphs[ep.to_predict]['TS'].keys())
          targets.sort() #otherwise keys won't necessarily be in the same order when another chunk is started
          targets = targets[start:stop]
     else: 
          targets = target_list[start:stop]
     
     if 'folds' in ep_conf:
          folds = ep_conf['folds']
     else:
          folds = 5
     
     xv_results = []
     for umls in targets:
          print(umls)
          xv = ep.k_fold(umls, folds)
          if xv:
               for fold in xv:
                    fold['objective'] = "J"
               xv_results.append(xv)
     chunk_t_stop = time.time()
     print("[LOG] chunk %s-%s done at %s, took %0.3f" % (start, stop, chunk_t_stop, chunk_t_stop - chunk_t_start))
     if parallel:
          #output.put(xv_results)
          output = {'xv': xv_results, 'meta': { 'ep_conf': ep_conf}}

          out_fname = 'xv_%s_chunk_%s_%s.txt' % (out_fname, start, stop)
          with open(out_fname,'w') as f:
               json.dump(output, f, cls=NumpyEncoder)

     else:
          return xv_results

def predict_chunk(start, stop, parallel = True, target_list = None, output=None, ep_conf = {}, out_fname='result'):
     chunk_t_start = time.time()
     #if target_list == None, use all targets. Else use the list provided.
     
     ep = EdgePrediction.EdgePrediction()

     #load data from edge list
     for to_load in ep_conf['files']:
          fname = ep_conf['path'] + "/" + to_load
          ep.CSV_to_graph(fname)
     
     #filter the source nodes in all graphs so that all the source nodes are common to all input graphs
     #keep those nodes and all their edges
     #create adjacency matrix
     ep.preprocess()
     
     #type of edge to predict
     ep.to_predict = ep_conf['to_predict']
     ep.objective_function = "J"
      
     #optimisation method
     ep.optimisation_method = ep_conf["optimisation_method"]
      
     #require all predictors
     ep.require_all_predictors = ep_conf["require_all_predictors"]
     
     #the order can be important as if there are ties, the result found first is returned
     #ties may make different predictions
     #ep.network_order = ['risk_gene', 'hpo_annotation', 'go_annotation', 'gene_ppi']
     if 'network_order' in ep_conf:
          ep.network_order = ep_conf['network_order']
     else:
          nw_order = list(ep.graphs.keys()).sort()
          ep.network_order = nw_order
     
     
     ep.ties = "minL2norm"

     
     if target_list == None:
          targets = list(ep.graphs[ep.to_predict]['TS'].keys())
          targets.sort() #otherwise keys won't necessarily be in the same order when another chunk is started
          targets = targets[start:stop]
     else: 
          targets = target_list[start:stop]
     
     results = []
     for umls in targets:
          print(umls)
          try:
               result = ep.predict(umls)
          except Exception as err:
               result = {'model_target': umls, 'model_built': False}
          results.append(result)
          
     chunk_t_stop = time.time()
     print("[LOG] chunk %s-%s done at %s, took %0.3f" % (start, stop, chunk_t_stop, chunk_t_stop - chunk_t_start))
     if parallel:
          #output.put(xv_results)
          output = {'results': results, 'meta': { 'ep_conf': ep_conf}}

          out_fname = 'predict_%s_chunk_%s_%s.txt' % (out_fname, start, stop)
          with open(out_fname,'w') as f:
               json.dump(output, f, cls=NumpyEncoder)

     else:
          return results


def run_parallel(start, stop, cores = 2, target_list = None, ep_conf = {}, fname = 'result'):
     if cores == 1:
          chunks = [(start, stop)]
     else: 
          diff = stop - start
          width = int(diff / cores)
          cuts = list(range(start, stop, width))
          cuts[-1] = stop
          chunks = [(cuts[x], cuts[x+1]) for x in range(len(cuts) - 1)]
       
     # Define an output queue
     output = mp.Queue()
     
     # Setup a list of processes that we want to run
     processes = []

     if ep_conf['job'] == 'xv':
          target = xv_chunk
     elif ep_conf['job'] == 'predict':
          target = predict_chunk
     else:
          print("job not implemented", ep_conf[job])
          return
     
     for c in chunks:
          processes.append(mp.Process(target=target, args=(c[0], c[1], True, target_list, output, ep_conf, fname)))
     
     # Run processes
     start = time.time()
     for p in processes:
         p.start()
     
     # Exit the completed processes
     p_num = 0
     for p in processes:
          print("[LOG] exit process %s" % (p_num))
          p_num += 1
          p.join()
     
     # Get process results from the output queue
     #results = [output.get() for p in processes]
     end = time.time()
     print("[LOG] all processes done, parallel (n=%s) %0.3f" % (cores, end-start))
    # return results


if __name__ == '__main__':
     start = 0
     stop = 6
     cores = 2

     path = '../data'
     files = [ "data.csv" ]
     to_predict = 'HAS_SIDE_EFFECT'
       
     ep_conf = {'path': path, 'files': files, 'to_predict': to_predict}
     ep_conf['require_all_predictors'] = False
     ep_conf['optimisation_method'] = "graph_sparse"
     ep_conf['job'] = 'xv'
     xv_results = run_parallel(start, stop, cores, None, ep_conf)
     out_fname = 'xv_test_%s_%s.txt' % (start, stop)
     
     with open(out_fname,'w') as f:
          json.dump(xv_results, f)     
