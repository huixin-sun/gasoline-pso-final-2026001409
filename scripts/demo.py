"""Small runnable example using the supplied fitted models (no fitting needed)."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import sys,csv,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.model import Regressor
from src.pso import GasolineObjective,optimize
root=Path(__file__).resolve().parents[1]
with (root/'data/samples.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
r=next(r for r in rows if r['sample_id']=='133');keys=list(r)[2:29]
x=np.array([float(r[k]) for k in keys]);bounds=json.loads((root/'results/bounds.json').read_text())
obj=GasolineObjective(Regressor.load(root/'models/ron.npz'),Regressor.load(root/'models/sulfur.npz'),x[:4])
res=optimize(obj,bounds['lower'],bounds['upper'],population=32,iterations=30,initial=x[4:],seed=133)
print('Sample: 133; evaluation budget:',res.evaluations)
print('Predicted [RON, sulfur]:',obj.predict(res.x[None,:])[0])
print('Feasible in surrogate:',res.value<1e6)
print('This is a computational example, not a validated plant recommendation.')
