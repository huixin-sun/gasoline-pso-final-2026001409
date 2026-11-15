"""NumPy-only standardized regression and one-hidden-layer ReLU networks."""
import numpy as np


class Regressor:
    def __init__(self, hidden=50, seed=42):
        self.hidden, self.seed = hidden, seed

    def fit(self, x, y, epochs=1500, validation=None, lr=0.003):
        self.xmean, self.xstd = x.mean(0), x.std(0)
        self.xstd = np.where(self.xstd < 1e-12, 1., self.xstd)
        self.ymean, self.ystd = y.mean(0), y.std(0)
        self.ystd = np.where(self.ystd < 1e-12, 1., self.ystd)
        a, target = (x-self.xmean)/self.xstd, (y-self.ymean)/self.ystd
        rng = np.random.default_rng(self.seed)
        self.params = [rng.normal(0, np.sqrt(2/x.shape[1]), (x.shape[1], self.hidden)),
                       np.zeros(self.hidden), rng.normal(0, np.sqrt(1/self.hidden), (self.hidden, y.shape[1])),
                       np.zeros(y.shape[1])]
        first = [np.zeros_like(p) for p in self.params]
        second = [np.zeros_like(p) for p in self.params]
        best, best_params, best_epoch = np.inf, None, epochs
        self.history = []
        for step in range(1, epochs+1):
            w1, b1, w2, b2 = self.params
            h = np.maximum(a@w1+b1, 0)
            e = h@w2+b2-target
            de = 2*e/e.size
            dh = (de@w2.T)*(h>0)
            grads = [a.T@dh+2e-4*w1, dh.sum(0), h.T@de+2e-4*w2, de.sum(0)]
            for k, g in enumerate(grads):
                first[k] = .9*first[k]+.1*g
                second[k] = .999*second[k]+.001*g*g
                self.params[k] -= lr*(first[k]/(1-.9**step))/(np.sqrt(second[k]/(1-.999**step))+1e-8)
            if step == 1 or step % 25 == 0:
                val = np.mean(((self.predict(validation[0])-validation[1])/self.ystd)**2) if validation is not None else float(np.mean(e*e))
                self.history.append([step, float(np.mean(e*e)), float(val)])
                if val < best:
                    best, best_epoch = val, step
                    best_params = [p.copy() for p in self.params]
        if validation is not None:
            self.params = best_params
        self.best_epoch = best_epoch
        return self

    def predict(self, x):
        w1,b1,w2,b2=self.params
        return (np.maximum(((np.asarray(x)-self.xmean)/self.xstd)@w1+b1,0)@w2+b2)*self.ystd+self.ymean

    def save(self, path):
        np.savez(path, xmean=self.xmean, xstd=self.xstd, ymean=self.ymean, ystd=self.ystd,
                 **{f'p{i}':p for i,p in enumerate(self.params)})

    @classmethod
    def load(cls, path):
        obj=cls()
        with np.load(path) as f:
            for n in ['xmean','xstd','ymean','ystd']: setattr(obj,n,f[n])
            obj.params=[f[f'p{i}'] for i in range(4)]
        return obj


def ridge_predict(x, y, test, alpha=10.):
    mean,std=x.mean(0),x.std(0)
    std=np.where(std<1e-12,1.,std)
    a=np.column_stack([np.ones(len(x)),(x-mean)/std])
    b=np.column_stack([np.ones(len(test)),(test-mean)/std])
    regularizer=np.eye(a.shape[1])*alpha
    regularizer[0,0]=0
    return b@np.linalg.solve(a.T@a+regularizer,a.T@y)


def metrics(actual, predicted):
    return {'rmse':float(np.sqrt(np.mean((actual-predicted)**2))),
            'mae':float(np.mean(np.abs(actual-predicted))),
            'r2':float(1-np.sum((actual-predicted)**2)/np.sum((actual-actual.mean())**2))}
