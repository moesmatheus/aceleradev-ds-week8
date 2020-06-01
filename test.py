import preprocess
import train
import predict
import numpy as np

def test_preprocess():
    out = preprocess.preProcess(train = False)

    assert len(out) == 2
    assert type(out[0]) == type(np.array([]))