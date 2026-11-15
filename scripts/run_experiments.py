"""All numerical experiments. No plot/report generation in performance timings."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import argparse,csv,json,platform,sys,time,contextlib,io
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.model import Regressor,ridge_predict,metrics
from src.pso import optimize,GasolineObjective,random_search

ROOT=Path(__file__).resolve().parents[1]

def write_json(name,obj):
    (ROOT/'results'/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def write_csv(name,rows):
    with (ROOT/'results'/name).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def read_data():
    with (ROOT/'data/samples.csv').open(encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    keys=list(rows[0])[2:29]
    # Chronological holdout: oldest 260 for fitting, newest 65 held out.
    order=sorted(range(len(rows)),key=lambda i:datetime.strptime(rows[i]['time'].strip(),'%Y/%m/%d %H:%M:%S'))
    rows=[rows[i] for i in order]
    x=np.array([[float(r[k]) for k in keys] for r in rows])
    y=np.array([[float(r['product_ron']),float(r['product_sulfur'])] for r in rows])
    ids=np.array([int(r['sample_id']) for r in rows])
    return rows,x,y,ids

def select_models(x,y,cfg):
    n=cfg['train_size']; stop=n-cfg['validation_size']
    tuning=[];models=[];records=[]
    for j,target in enumerate(['ron','sulfur']):
        candidates=[]
        for seed in cfg['model_seeds']:
            m=Regressor(cfg['hidden'],seed).fit(x[:stop],y[:stop,j:j+1],epochs=cfg['epochs'],validation=(x[stop:n],y[stop:n,j:j+1]))
            mse=float(np.mean((m.predict(x[stop:n])-y[stop:n,j:j+1])**2))
            tuning.append({'target':target,'seed':seed,'epoch':m.best_epoch,'validation_rmse':mse**.5})
            candidates.append((mse,seed,m.best_epoch))
        _,seed,epochs=min(candidates)
        m=Regressor(cfg['hidden'],seed).fit(x[:n],y[:n,j:j+1],epochs=epochs)
        m.save(ROOT/f'models/{target}.npz');models.append(m)
        pred=m.predict(x[n:]).ravel()
        ridge=ridge_predict(x[:n],y[:n,j:j+1],x[n:]).ravel()
        for method,p in [('MLP',pred),('Ridge',ridge),('TrainMean',np.repeat(y[:n,j].mean(),len(y)-n))]:
            records.append({'target':target,'method':method,**metrics(y[n:,j],p)})
    write_csv('model_tuning.csv',tuning);write_csv('model_metrics.csv',records)
    return models

def bounds_for(x,variables,n):
    physical_lo=np.array([v['lower'] for v in variables]);physical_hi=np.array([v['upper'] for v in variables])
    lo=np.maximum(physical_lo,x[:n,4:].min(0));hi=np.minimum(physical_hi,x[:n,4:].max(0))
    if np.any(lo>=hi): raise ValueError('Empty data-supported marginal box')
    return lo,hi,physical_lo,physical_hi

def one_case(job):
    idx,seed,pop,it,x,fixed,lo,hi,ron_path,sulfur_path=job
    obj=GasolineObjective(Regressor.load(ron_path),Regressor.load(sulfur_path),fixed)
    res=optimize(obj,lo,hi,population=pop,iterations=it,seed=seed,initial=x)
    return idx,res.value

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--quick',action='store_true');parser.add_argument('--reuse-models',action='store_true')
    args=parser.parse_args();cfg=json.loads((ROOT/'config/experiment.json').read_text())
    start=time.perf_counter()
    if args.quick:
        cfg.update(epochs=150,model_seeds=[11],population=32,iterations=20,benchmark_repeats=2,quality_seeds=3,quality_samples=2,quality_population=24,quality_iterations=15)
    rows,x,y,ids=read_data();n=cfg['train_size']
    models=[Regressor.load(ROOT/f'models/{t}.npz') for t in ['ron','sulfur']] if args.reuse_models else select_models(x,y,cfg)
    variables=json.loads((ROOT/'data/variables.json').read_text(encoding='utf-8'))
    lo,hi,plo,phi=bounds_for(x,variables,n)
    write_json('bounds.json',{'tags':[v['tag'] for v in variables],'lower':lo.tolist(),'upper':hi.tolist(),'physical_lower':plo.tolist(),'physical_upper':phi.tolist()})
    write_json('split.json',{'training_ids':ids[:n].tolist(),'holdout_ids':ids[n:].tolist(),'validation_ids':ids[n-cfg['validation_size']:n].tolist(),
                            'training_dates':[rows[0]['time'],rows[n-1]['time']],'holdout_dates':[rows[n]['time'],rows[-1]['time']]})
    pred=np.column_stack([m.predict(x).ravel() for m in models])
    write_csv('predictions.csv',[{'sample_id':int(ids[i]),'split':'train' if i<n else 'holdout','ron_actual':y[i,0],'ron_pred':pred[i,0],
                                  'sulfur_actual':y[i,1],'sulfur_pred':pred[i,1]} for i in range(len(x))])
    print('Models ready',flush=True)
    # Benchmark one fixed holdout input. Every call has the same objective, seed and evaluation budget.
    obj=GasolineObjective(*models,x[n,:4]);benchmark=[];equivalence=[]
    for pop in cfg['benchmark_populations']:
        kw=dict(population=pop,iterations=cfg['benchmark_iterations'],seed=cfg['seed'],initial=x[n,4:])
        refs={mode:optimize(obj,lo,hi,mode=mode,**kw) for mode in ['loop','vector']} # warmup
        equivalence.append({'population':pop,'max_history_abs_error':float(np.max(np.abs(refs['loop'].history-refs['vector'].history))),
                            'best_value_abs_error':abs(refs['loop'].value-refs['vector'].value),
                            'evaluations':refs['vector'].evaluations})
        for repeat in range(cfg['benchmark_repeats']):
            # Alternating order reduces a systematic warm-cache / clock-order bias.
            for mode in (['loop','vector'] if repeat%2==0 else ['vector','loop']):
                t=time.perf_counter();res=optimize(obj,lo,hi,mode=mode,**kw);elapsed=time.perf_counter()-t
                benchmark.append({'population':pop,'iterations':kw['iterations'],'repeat':repeat,'mode':mode,'seconds':elapsed,
                                  'evaluations':res.evaluations,'objective':res.value})
    write_csv('timings.csv',benchmark);write_csv('equivalence.csv',equivalence)
    print('Implementation benchmark complete',flush=True)
    # Quality experiment: 5 evenly spaced holdout rows x 20 seeds; identical evaluation budget.
    cases=np.linspace(n,len(x)-1,cfg['quality_samples'],dtype=int);quality=[];convergence=[]
    for case in cases:
        objective=GasolineObjective(*models,x[case,:4])
        for seed in range(cfg['quality_seeds']):
            kw=dict(population=cfg['quality_population'],iterations=cfg['quality_iterations'],seed=seed,initial=x[case,4:])
            t=time.perf_counter();p=optimize(objective,lo,hi,**kw);pt=time.perf_counter()-t
            t=time.perf_counter();r=random_search(objective,lo,hi,budget=p.evaluations,seed=seed,initial=x[case,4:]);rt=time.perf_counter()-t
            for method,res,seconds in [('PSO',p,pt),('Random',r,rt)]:
                out=objective.predict(res.x[None,:])[0]
                quality.append({'sample_id':int(ids[case]),'seed':seed,'method':method,'seconds':seconds,'budget':res.evaluations,
                                'feasible':int(res.value<1e6),'predicted_loss':float(x[case,1]-out[0]),'sulfur':float(out[1]),'objective':res.value})
            if case==cases[0]:
                convergence.extend({'seed':seed,'iteration':j,'objective':v} for j,v in enumerate(p.history))
    write_csv('quality.csv',quality);write_csv('convergence.csv',convergence)
    print('Repeated-search experiments complete',flush=True)
    # End-to-end application to all 325 cases; no stochastic claims from this single run per case.
    app=[];controls=[];trace=None
    training_z=(x[:n]-models[0].xmean)/models[0].xstd
    for i in range(len(x)):
        objective=GasolineObjective(*models,x[i,:4])
        initial=np.clip(x[i,4:],lo,hi)
        base=objective.predict(initial[None,:])[0]
        t=time.perf_counter();res=optimize(objective,lo,hi,population=cfg['population'],iterations=cfg['iterations'],seed=cfg['seed']+int(ids[i]),initial=initial);elapsed=time.perf_counter()-t
        opt=objective.predict(res.x[None,:])[0];loss=x[i,1]-opt[0];base_loss=x[i,1]-base[0]
        point_z=(np.r_[x[i,:4],res.x]-models[0].xmean)/models[0].xstd
        nn=float(np.sqrt(np.min(np.sum((training_z-point_z)**2,axis=1))))
        app.append({'sample_id':int(ids[i]),'split':'train' if i<n else 'holdout','seconds':elapsed,'feasible':int(res.value<1e6),
                    'raw_ron':x[i,1],'actual_product_ron':y[i,0],'observed_loss':x[i,1]-y[i,0],
                    'original_predicted_ron':pred[i,0],'original_predicted_sulfur':pred[i,1],
                    'clipped_initial_ron':base[0],'clipped_initial_loss':base_loss,'optimized_ron':opt[0],
                    'optimized_sulfur':opt[1],'optimized_loss':loss,'loss_reduction_vs_model':(base_loss-loss)/base_loss if base_loss>1e-9 else None,
                    'nearest_training_z_distance':nn,'original_out_of_box_dims':int(np.sum((x[i,4:]<lo)|(x[i,4:]>hi)))})
        controls.append({'sample_id':int(ids[i]),**{v['tag']:float(a) for v,a in zip(variables,res.x)}})
        if ids[i]==133:
            trace=res.history;np.savez(ROOT/'results/sample133_controls.npz',initial=initial,optimized=res.x)
    write_csv('application.csv',app);write_csv('optimized_controls.csv',controls)
    if trace is not None:write_csv('sample133_search.csv',[{'iteration':j,'objective':v} for j,v in enumerate(trace)])
    print('All 325 applications complete',flush=True)
    # Full engineering ranges vs training-supported marginal ranges on holdout cases.
    sensitivity=[]
    for i in cases:
        objective=GasolineObjective(*models,x[i,:4])
        for box,a,b in [('training_box',lo,hi),('physical_box',plo,phi)]:
            for seed in range(5):
                res=optimize(objective,a,b,population=96,iterations=80,seed=seed,initial=x[i,4:])
                point=np.r_[x[i,:4],res.x];out=objective.predict(res.x[None,:])[0]
                dz=(point-models[0].xmean)/models[0].xstd
                nearest=float(np.sqrt(np.min(np.sum((training_z-dz)**2,axis=1))))
                sensitivity.append({'sample_id':int(ids[i]),'seed':seed,'box':box,'feasible':int(res.value<1e6),'predicted_loss':float(x[i,1]-out[0]),
                                    'sulfur':float(out[1]),'nearest_training_z_distance':nearest})
    write_csv('bounds_sensitivity.csv',sensitivity)
    # Parallelism is across independent cases. Wall time includes process startup, model loading and IPC.
    jobs=[(int(ids[i]),cfg['seed']+int(ids[i]),96,80,x[i,4:],x[i,:4],lo,hi,str(ROOT/'models/ron.npz'),str(ROOT/'models/sulfur.npz')) for i in np.linspace(n,len(x)-1,12,dtype=int)]
    parallel=[]
    for repeat in range(3):
        reference=None
        for workers in [1,2,4]:
            t=time.perf_counter()
            if workers==1: values=list(map(one_case,jobs))
            else:
                with ProcessPoolExecutor(max_workers=workers) as executor: values=list(executor.map(one_case,jobs))
            seconds=time.perf_counter()-t
            if reference is None:reference=np.array([v for _,v in values])
            err=float(np.max(np.abs(reference-np.array([v for _,v in values]))))
            parallel.append({'repeat':repeat,'workers':workers,'cases':len(jobs),'seconds':seconds,'max_abs_error':err})
    write_csv('parallel.csv',parallel)
    # Analytic functions have known global minimum 0; report distribution, not a proof of optimality.
    analytic=[]
    for name,objective,lower,upper in [('sphere',lambda z:np.sum(z*z,axis=1),-5.12,5.12),
      ('rastrigin',lambda z:10*z.shape[1]+np.sum(z*z-10*np.cos(2*np.pi*z),axis=1),-5.12,5.12),
      ('rosenbrock',lambda z:np.sum(100*(z[:,1:]-z[:,:-1]**2)**2+(1-z[:,:-1])**2,axis=1),-2.,2.)]:
        for dim in [2,10]:
            for seed in range(20):
                res=optimize(objective,np.repeat(lower,dim),np.repeat(upper,dim),population=96,iterations=200,seed=seed)
                analytic.append({'function':name,'dimension':dim,'seed':seed,'best_value':res.value,'budget':res.evaluations})
    write_csv('analytic.csv',analytic)
    env=io.StringIO()
    with contextlib.redirect_stdout(env):np.show_config()
    write_json('environment.json',{'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'processor':platform.processor(),
                                  'logical_cpu_count':os.cpu_count(),'blas_threads':1,'numpy_build':env.getvalue(),'config':cfg,
                                  'experiment_seconds':time.perf_counter()-start,'run_mode':'quick' if args.quick else 'full'})
    print('Completed in',round(time.perf_counter()-start,2),'seconds',flush=True)

if __name__=='__main__':main()
