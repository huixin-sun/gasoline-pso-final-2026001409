"""Synchronous global-best PSO. Both modes use identical random-number shapes.

Only population arithmetic/evaluation granularity changes. Invalid objective values
are rejected rather than silently converted. Bounds are closed numerical boxes.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Result:
    x: np.ndarray
    value: float
    history: np.ndarray
    evaluations: int


def _evaluate(objective, x, mode):
    y = np.asarray(objective(x) if mode == 'vector' else
                   [np.asarray(objective(row[None,:])).item() for row in x],dtype=float)
    if y.shape != (len(x),) or not np.all(np.isfinite(y)):
        raise ValueError('Objective must return one finite scalar per particle')
    return y


def optimize(objective, lower, upper, population=128, iterations=120, seed=42,
             mode='vector', initial=None, w=.8, c1=.5, c2=.5, vmax=.2):
    lower,upper=np.asarray(lower,float),np.asarray(upper,float)
    if lower.ndim!=1 or lower.shape!=upper.shape or not np.all(np.isfinite([lower,upper])) or np.any(lower>=upper):
        raise ValueError('Finite, strictly ordered vector bounds required')
    if population<2 or iterations<0 or mode not in ['loop','vector']:
        raise ValueError('Invalid population, iterations or implementation mode')
    rng=np.random.default_rng(seed)
    shape=(population,len(lower))
    x=rng.random(shape)
    if initial is not None:
        x[0]=np.clip((np.asarray(initial)-lower)/(upper-lower),0,1)
    velocity=rng.uniform(-vmax,vmax,shape)
    physical=lambda z: lower+z*(upper-lower)
    score=_evaluate(objective,physical(x),mode)
    personal=x.copy(); personal_score=score.copy()
    k=int(np.argmin(personal_score)); best=personal[k].copy(); best_score=float(personal_score[k])
    history=[best_score]
    for _ in range(iterations):
        r1,r2=rng.random(shape),rng.random(shape)
        if mode=='vector':
            velocity=np.clip(w*velocity+c1*r1*(personal-x)+c2*r2*(best-x),-vmax,vmax)
            trial=x+velocity
            hit=(trial<0)|(trial>1)
            x=np.clip(trial,0,1)
            velocity[hit]=0
        else:
            for i in range(population):
                velocity[i]=np.clip(w*velocity[i]+c1*r1[i]*(personal[i]-x[i])+c2*r2[i]*(best-x[i]),-vmax,vmax)
                trial=x[i]+velocity[i]
                hit=(trial<0)|(trial>1)
                x[i]=np.clip(trial,0,1)
                velocity[i,hit]=0
        score=_evaluate(objective,physical(x),mode)
        better=score<personal_score
        personal[better]=x[better]; personal_score[better]=score[better]
        k=int(np.argmin(personal_score))
        if personal_score[k]<best_score:
            best=personal[k].copy(); best_score=float(personal_score[k])
        history.append(best_score)
    return Result(physical(best),best_score,np.asarray(history),population*(iterations+1))


class GasolineObjective:
    """Two separately fitted surrogate models; first four input features fixed.

    A feasible value is [0, raw RON]. An infeasible value is 1e6 plus normalized
    violation, hence feasibility dominates all physically meaningful feasible values.
    """
    def __init__(self, ron_model, sulfur_model, fixed):
        self.ron_model,self.sulfur_model=ron_model,sulfur_model
        self.fixed=np.asarray(fixed,float)

    def predict(self, controls):
        x=np.column_stack([np.broadcast_to(self.fixed,(len(controls),4)),controls])
        return np.column_stack([self.ron_model.predict(x).ravel(),self.sulfur_model.predict(x).ravel()])

    def __call__(self, controls):
        pred=self.predict(controls)
        ron,sulfur=pred[:,0],pred[:,1]
        violation=np.maximum(ron-self.fixed[1],0)+np.maximum(-ron,0)+np.maximum(-sulfur,0)+np.maximum(sulfur-5,0)
        return np.where(violation<=1e-10,self.fixed[1]-ron,1e6+violation)


def random_search(objective, lower, upper, budget, seed=42, initial=None, batch=512):
    rng=np.random.default_rng(seed)
    best_score=np.inf; best=None
    used=0
    while used<budget:
        n=min(batch,budget-used)
        x=rng.uniform(lower,upper,(n,len(lower)))
        if used==0 and initial is not None: x[0]=np.clip(initial,lower,upper)
        score=objective(x); k=int(np.argmin(score))
        if score[k]<best_score: best_score=float(score[k]);best=x[k].copy()
        used+=n
    return Result(best,best_score,np.array([best_score]),used)
