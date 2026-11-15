"""Compute every reported number directly from raw experiment CSVs."""
import csv,json,statistics,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(name):
    with (ROOT/'results'/name).open(encoding='utf-8') as f:return list(csv.DictReader(f))
def describe(values):
    values=list(map(float,values));return dict(mean=statistics.mean(values),median=statistics.median(values),sd=statistics.stdev(values) if len(values)>1 else 0.,min=min(values),max=max(values))
def main():
    times=read('timings.csv');q=read('quality.csv');app=read('application.csv');sens=read('bounds_sensitivity.csv');p=read('parallel.csv')
    summary={'timings':[],'models':read('model_metrics.csv'),'quality':[],'application':{},'parallel':[],'sensitivity':[],'analytic':[]}
    for pop in sorted(set(int(r['population']) for r in times)):
        modes={m:describe(r['seconds'] for r in times if r['mode']==m and int(r['population'])==pop) for m in ['loop','vector']}
        summary['timings'].append({'population':pop,**modes,'speedup_median':modes['loop']['median']/modes['vector']['median']})
    for m in ['PSO','Random']:
        rows=[r for r in q if r['method']==m];feasible=[r for r in rows if r['feasible']=='1']
        summary['quality'].append({'method':m,'runs':len(rows),'feasible':len(feasible),'loss':describe(r['predicted_loss'] for r in feasible),
                                   'seconds':describe(r['seconds'] for r in rows),'budget':int(rows[0]['budget'])})
    for split in ['all','train','holdout']:
        rows=[r for r in app if split=='all' or r['split']==split]
        f=[r for r in rows if r['feasible']=='1']
        summary['application'][split]={'n':len(rows),'feasible':len(f),'predicted_loss':describe(r['optimized_loss'] for r in f),
            'sulfur':describe(r['optimized_sulfur'] for r in f),'seconds_sum':sum(float(r['seconds']) for r in rows),
            'distance':describe(r['nearest_training_z_distance'] for r in f),
            'clipped_cases':sum(int(r['original_out_of_box_dims'])>0 for r in rows),
            'model_reduction_over30':sum(float(r['loss_reduction_vs_model'])>.3 for r in f if r['loss_reduction_vs_model'])}
    for workers in [1,2,4]:
        summary['parallel'].append({'workers':workers,**describe(r['seconds'] for r in p if int(r['workers'])==workers)})
    for box in ['training_box','physical_box']:
        rows=[r for r in sens if r['box']==box]
        summary['sensitivity'].append({'box':box,'runs':len(rows),'feasible':sum(r['feasible']=='1' for r in rows),
             'distance':describe(r['nearest_training_z_distance'] for r in rows),'loss':describe(r['predicted_loss'] for r in rows)})
    a=read('analytic.csv')
    for fun in ['sphere','rastrigin','rosenbrock']:
        for dim in [2,10]:
            rows=[r for r in a if r['function']==fun and int(r['dimension'])==dim]
            summary['analytic'].append({'function':fun,'dimension':dim,**describe(r['best_value'] for r in rows)})
    summary['max_equivalence_error']=max(float(r['max_history_abs_error']) for r in read('equivalence.csv'))
    summary['split']=json.loads((ROOT/'results/split.json').read_text())
    # Baseline distances for held-out original observations make the OOD comparison interpretable.
    import sys
    sys.path.insert(0,str(ROOT))
    import numpy as np
    from src.model import Regressor
    from scripts.run_experiments import read_data
    _,x,_,ids=read_data();model=Regressor.load(ROOT/'models/ron.npz')
    z=(x-model.xmean)/model.xstd
    distances=np.sqrt(np.min(np.sum((z[260:,None,:]-z[None,:260,:])**2,axis=2),axis=1))
    summary['holdout_original_nn_distance']=describe(distances)
    summary['environment']=json.loads((ROOT/'results/environment.json').read_text())
    (ROOT/'results/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['environment','split']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
