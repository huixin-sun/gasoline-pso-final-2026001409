"""Plot recorded results; this script performs no timed numerical experiments."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def read(name):
    with (ROOT/'results'/name).open(encoding='utf-8') as f:return list(csv.DictReader(f))
def save(fig,name):
    fig.savefig(ROOT/f'figures/{name}.png',dpi=190,bbox_inches='tight',facecolor='white');plt.close(fig)
def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.titleweight':'bold','axes.labelcolor':'#243447','text.color':'#243447','axes.edgecolor':'#9aa9b8'})
    summary=json.loads((ROOT/'results/summary.json').read_text(encoding='utf-8'))
    blue,orange='#176b89','#e39436'
    fig,ax=plt.subplots(1,2,figsize=(10,3.4));pop=[r['population'] for r in summary['timings']];xx=np.arange(len(pop))
    for j,(mode,color) in enumerate([('loop',orange),('vector',blue)]):
        vals=np.array([r[mode]['median'] for r in summary['timings']])*1000
        ax[0].bar(xx+(j-.5)*.32,vals,.32,label=mode,color=color)
    ax[0].set(yscale='log',xticks=xx,xticklabels=pop,xlabel='Population',ylabel='Median wall time (ms)',title='Same search, different implementation');ax[0].legend(frameon=False)
    speed=[r['speedup_median'] for r in summary['timings']]
    ax[1].plot(pop,speed,'o-',color=blue,lw=2)
    for x,v in zip(pop,speed):ax[1].annotate(f'{v:.1f}x',(x,v),xytext=(0,9),textcoords='offset points',ha='center')
    ax[1].set(xlabel='Population',ylabel='Loop / vector median time',title='Batching reduces Python overhead');ax[1].set_ylim(0,max(speed)*1.3)
    fig.tight_layout();save(fig,'timing')
    pred=[r for r in read('predictions.csv') if r['split']=='holdout']
    fig,ax=plt.subplots(1,2,figsize=(10,3.4))
    for a,t in zip(ax,['ron','sulfur']):
        actual=np.array([float(r[t+'_actual']) for r in pred]);estimate=np.array([float(r[t+'_pred']) for r in pred])
        a.scatter(actual,estimate,s=22,alpha=.75,color=blue);lo=min(actual.min(),estimate.min());hi=max(actual.max(),estimate.max())
        a.plot([lo,hi],[lo,hi],'--',color=orange,lw=1);a.set(xlabel='Observed value',ylabel='MLP prediction',title='RON' if t=='ron' else 'Sulfur (microgram/g)')
    fig.tight_layout();save(fig,'prediction')
    q=read('quality.csv');conv=read('convergence.csv')
    fig,ax=plt.subplots(1,2,figsize=(10,3.4))
    vals=[np.maximum([float(r['predicted_loss']) for r in q if r['method']==m and r['feasible']=='1'],1e-10) for m in ['PSO','Random']]
    bp=ax[0].boxplot(vals,tick_labels=['PSO','Random'],patch_artist=True)
    for patch,c in zip(bp['boxes'],[blue,orange]):patch.set_facecolor(c);patch.set_alpha(.6)
    ax[0].set(yscale='log',ylabel='Predicted RON loss (feasible runs)',title='Equal budget: 7,776 evaluations')
    arr=np.array([[float(r['objective']) for r in conv if int(r['seed'])==seed] for seed in sorted({int(r['seed']) for r in conv})])
    med=np.median(arr,0);lower,upper=np.quantile(arr,[.25,.75],axis=0)
    ax[1].plot(med,color=blue);ax[1].fill_between(np.arange(len(med)),np.maximum(lower,1e-12),np.maximum(upper,1e-12),color=blue,alpha=.2)
    ax[1].set(yscale='log',xlabel='PSO iteration',ylabel='Best objective',title='First holdout case: median and IQR')
    fig.tight_layout();save(fig,'quality')
    fig,ax=plt.subplots(figsize=(7,3.2));par=summary['parallel']
    ax.bar([str(r['workers']) for r in par],[r['median'] for r in par],color=[blue,orange,orange])
    ax.set(xlabel='Worker processes',ylabel='Median end-to-end seconds',title='12 short cases: process startup outweighs parallel work')
    fig.tight_layout();save(fig,'parallel')
    fig,ax=plt.subplots(1,2,figsize=(10,3.4));app=read('application.csv')
    for split,color in [('train',blue),('holdout',orange)]:
        rows=[r for r in app if r['split']==split]
        ax[0].scatter([float(r['nearest_training_z_distance']) for r in rows],np.maximum([float(r['optimized_loss']) for r in rows],1e-12),s=12,alpha=.55,label=split,color=color)
    ax[0].set(yscale='log',xlabel='Nearest training distance (standardized)',ylabel='Optimized predicted RON loss',title='Very low loss does not validate a surrogate');ax[0].legend(frameon=False)
    sens=read('bounds_sensitivity.csv');vals=[[float(r['nearest_training_z_distance']) for r in sens if r['box']==box] for box in ['training_box','physical_box']]
    ax[1].boxplot(vals,tick_labels=['Training box','Physical box']);ax[1].set(ylabel='Nearest training distance',title='Marginal bounds do not ensure joint data support')
    fig.tight_layout();save(fig,'reliability')
    r133=[r for r in app if r['sample_id']=='133'][0]
    fig,ax=plt.subplots(figsize=(7,3.1));rr=read('sample133_search.csv');ax.plot([int(r['iteration']) for r in rr],np.maximum([float(r['objective']) for r in rr],1e-12),color=blue)
    ax.set(yscale='log',xlabel='PSO search iteration',ylabel='Best objective',title='Sample 133: actual search trace, not plant adjustment steps')
    fig.tight_layout();save(fig,'sample133')
    print('Wrote six figures from raw records.')
if __name__=='__main__':main()
