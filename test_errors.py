#import the module
import EdgePrediction, json, time

#create a new instance
ep = EdgePrediction.EdgePrediction()

#load data from edge list
print("file has target node error")
try:
    ep.CSV_to_graph(fname = 'data/data_tgt_error.csv', encoding='utf-8-sig')
except Exception as err:
    print(err)

print("file has src node error")
ep = EdgePrediction.EdgePrediction()
try:
    ep.CSV_to_graph(fname = 'data/data_src_error.csv', encoding='utf-8-sig')
except Exception as err:
    print(err)


ep = EdgePrediction.EdgePrediction()
ep.CSV_to_graph(fname = 'data/data_no_common_src.csv', encoding='utf-8-sig')
print("file has no common source nodes")
try:
    ep.preprocess()
except Exception as err:
    print(err)

#load a network with no common source nodes, catch the error but try to run anyway
ep = EdgePrediction.EdgePrediction()
ep.CSV_to_graph(fname = 'data/data_no_common_src.csv', encoding='utf-8-sig')
print("file has no common source nodes")
try:
    ep.preprocess()
except Exception as err:
    print(err)
target_name = "C0027849"
ep.to_predict = 'HAS_SIDE_EFFECT'
print("try to predict anyway")
try:
    result = ep.predict(target = target_name)
except Exception as err:
    print(err)

# load an ok file but don't set to_predict
#load data from edge list
ep = EdgePrediction.EdgePrediction()
ep.CSV_to_graph(fname = 'data/data.csv')
ep.preprocess()
target_name = "C0027849"
print("did not set to_predict")
try:
    result = ep.predict(target = target_name)
except Exception as err:
    print(err)

#try to use an invalid optimisation method
ep.to_predict = 'HAS_SIDE_EFFECT'
ep.optimisation_method = "something not implemented"
print("set an invalid optimisation method")
try:
    result = ep.predict(target = target_name)
except Exception as err:
    print(err)