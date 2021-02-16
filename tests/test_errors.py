import EdgePrediction, pytest

def test_target_node_error():
    ep = EdgePrediction.EdgePrediction()
    with pytest.raises(Exception):
        ep.CSV_to_graph(fname = 'data/data_tgt_error.csv', encoding='utf-8-sig')


def test_source_node_error():
    ep = EdgePrediction.EdgePrediction()
    with pytest.raises(Exception):
        ep.CSV_to_graph(fname = 'data/data_src_error.csv', encoding='utf-8-sig')

def test_no_common_source_nodes():
    ep = EdgePrediction.EdgePrediction()
    ep.CSV_to_graph(fname = 'data/data_no_common_src.csv', encoding='utf-8-sig')
    with pytest.raises(Exception):
        ep.preprocess()

def test_no_common_source_nodes_run_anyway():
    #load a network with no common source nodes, catch the error but try to run anyway
    ep = EdgePrediction.EdgePrediction()
    ep.CSV_to_graph(fname = 'data/data_no_common_src.csv', encoding='utf-8-sig')
    try:
        ep.preprocess()
    except Exception as err:
        pass
    target_name = "C0027849"
    ep.to_predict = 'HAS_SIDE_EFFECT'
    with pytest.raises(Exception):
        result = ep.predict(target = target_name)


def test_no_target_set():
    # load an ok file but don't set to_predict
    #load data from edge list
    ep = EdgePrediction.EdgePrediction()
    ep.CSV_to_graph(fname = 'data/data.csv')
    ep.preprocess()
    target_name = "C0027849"
    with pytest.raises(Exception):
        result = ep.predict(target = target_name)

def test_invalid_optimisation():
    #try to use an invalid optimisation method
    ep = EdgePrediction.EdgePrediction()
    ep.CSV_to_graph(fname = 'data/data.csv')
    ep.preprocess()
    target_name = "C0027849"
    ep.to_predict = 'HAS_SIDE_EFFECT'
    ep.optimisation_method = "something not implemented"
    with pytest.raises(Exception):
        result = ep.predict(target = target_name)
