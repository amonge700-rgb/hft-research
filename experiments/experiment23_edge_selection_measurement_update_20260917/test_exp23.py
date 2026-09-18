import numpy as np
from run_exp23 import rebuild_c,rebuild_l,PAIRS

def test_physical_rebuild():
    c=np.eye(8)*8e-12-np.ones((8,8))*1e-12; np.fill_diagonal(c,7e-12)
    l=np.eye(8)*90e-9+np.ones((8,8))*20e-9
    d=np.zeros(len(PAIRS)); d[3]=.1
    cc=rebuild_c(c,d); ll=rebuild_l(l,d)
    assert np.max(abs(cc-cc.T))<1e-20
    assert np.min(np.linalg.eigvalsh(cc))>-1e-20
    assert np.max(abs(ll-ll.T))<1e-20
    assert np.min(np.linalg.eigvalsh(ll))>0
