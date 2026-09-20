from pathlib import Path
import sys, numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_exp27 import load_teachers,rebuild_physical,response_feature,jacobian,FREQ

t=load_teachers()[0]; c,l=rebuild_physical(t,np.zeros(3))
assert np.allclose(c,t['C']) and np.allclose(l,t['L'],rtol=2e-6,atol=1e-12)
assert np.linalg.eigvalsh(c).min()>-1e-12 and np.linalg.eigvalsh(l).min()>0
assert response_feature(t,np.zeros(3)).shape==(8*len(FREQ),) and jacobian(t,np.zeros(3)).shape==(8*len(FREQ),3)
print({'C_min_eig':float(np.linalg.eigvalsh(c).min()),'L_min_eig':float(np.linalg.eigvalsh(l).min()),'physical_coordinates':3})
