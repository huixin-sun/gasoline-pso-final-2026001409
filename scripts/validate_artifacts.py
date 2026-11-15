"""Independent consistency checks for final recorded results and deliverables."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.model import Regressor,metrics
from src.pso import GasolineObjective
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1]
def read(name):
    with (ROOT/'results'/name).open(encoding='utf-8') as f:return list(csv.DictReader(f))
def main():
    with (ROOT/'data/samples.csv').open(encoding='utf-8') as f:data=list(csv.DictReader(f))
    by_id={r['sample_id']:r for r in data};keys=list(data[0])[2:29]
    split=json.loads((ROOT/'results/split.json').read_text())
    assert len(split['training_ids'])==260 and len(split['holdout_ids'])==65
    assert not set(split['training_ids'])&set(split['holdout_ids'])
    assert len(set(split['training_ids']+split['holdout_ids']))==325
    m=[Regressor.load(ROOT/f'models/{t}.npz') for t in ['ron','sulfur']]
    bounds=json.loads((ROOT/'results/bounds.json').read_text());lo=np.array(bounds['lower']);hi=np.array(bounds['upper'])
    controls=read('optimized_controls.csv');app={r['sample_id']:r for r in read('application.csv')}
    assert len(controls)==325 and set(r['sample_id'] for r in controls)==set(by_id)
    max_err=0
    for row in controls:
        raw=by_id[row['sample_id']];x=np.array([float(raw[k]) for k in keys]);u=np.array([float(row[k]) for k in bounds['tags']])
        assert np.all(u>=lo-1e-10) and np.all(u<=hi+1e-10)
        obj=GasolineObjective(*m,x[:4]);p=obj.predict(u[None,:])[0];record=app[row['sample_id']]
        max_err=max(max_err,abs(p[0]-float(record['optimized_ron'])),abs(p[1]-float(record['optimized_sulfur'])))
        assert int(obj(u[None,:])[0]<1e6)==int(record['feasible'])
    assert max_err<1e-10
    pred=[r for r in read('predictions.csv') if r['split']=='holdout'];modelmetrics=read('model_metrics.csv')
    for target in ['ron','sulfur']:
        y=np.array([float(r[target+'_actual']) for r in pred]);p=np.array([float(r[target+'_pred']) for r in pred])
        actual=metrics(y,p);record=next(r for r in modelmetrics if r['target']==target and r['method']=='MLP')
        for k,v in actual.items():assert abs(v-float(record[k]))<1e-12
    assert max(float(r['max_history_abs_error']) for r in read('equivalence.csv'))<1e-10
    q=read('quality.csv');assert len(q)==200
    groups={}
    for r in q:groups.setdefault((r['sample_id'],r['seed']),[]).append(r)
    assert len(groups)==100
    assert all(len(v)==2 and {int(r['budget']) for r in v}=={7776} for v in groups.values())
    assert all(float(r['max_abs_error'])<1e-10 for r in read('parallel.csv'))
    pages={name:len(PdfReader(ROOT/name).pages) for name in ['report.pdf','slides.pdf']}
    assert pages=={'report.pdf':8,'slides.pdf':11},pages
    result={'passed':True,'application_rows':325,'holdout_rows':65,'paired_quality_runs':100,
            'max_recomputed_prediction_error':max_err,'pdf_pages':pages,'core_tests':'11 passed; see test_run.txt'}
    (ROOT/'results/final_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
