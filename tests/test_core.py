import unittest,tempfile,csv,json
from pathlib import Path
import numpy as np
from src.pso import optimize,GasolineObjective,random_search
from src.model import Regressor,ridge_predict

class CoreTests(unittest.TestCase):
    def test_sphere_accuracy_and_monotonicity(self):
        r=optimize(lambda x:(x*x).sum(1),[-5,-5],[5,5],iterations=200,population=64,seed=8)
        self.assertLess(r.value,1e-10)
        self.assertTrue(np.all(np.diff(r.history)<=0))
        self.assertEqual(r.evaluations,64*201)

    def test_loop_vector_same_randomness(self):
        fn=lambda x:(x*x).sum(1)
        kw=dict(population=24,iterations=40,seed=51)
        a=optimize(fn,[-3]*7,[4]*7,mode='loop',**kw)
        b=optimize(fn,[-3]*7,[4]*7,mode='vector',**kw)
        np.testing.assert_allclose(a.history,b.history,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(a.x,b.x,rtol=1e-12,atol=1e-12)

    def test_boundary_optimum(self):
        r=optimize(lambda x:-x.sum(1),[0,0],[1,1],iterations=60,population=40)
        np.testing.assert_allclose(r.x,[1,1],atol=1e-8)

    def test_zero_iterations_initial_candidate(self):
        r=optimize(lambda x:(x*x).sum(1),[-1]*3,[1]*3,iterations=0,initial=[0]*3)
        self.assertEqual(r.value,0)

    def test_invalid_bounds_and_nan(self):
        with self.assertRaises(ValueError):optimize(lambda x:x.sum(1),[1],[0])
        with self.assertRaises(ValueError):optimize(lambda x:np.full(len(x),np.nan),[0],[1])

    def test_objective_constraints(self):
        class Fake:
            def __init__(self,col):self.col=col
            def predict(self,x):return x[:,self.col:self.col+1]
        obj=GasolineObjective(Fake(4),Fake(5),[100,90,50,720])
        scores=obj(np.array([[89,3],[91,3],[89,6],[89,-1]]))
        self.assertEqual(scores[0],1)
        self.assertTrue(np.all(scores[1:]>=1e6))

    def test_mlp_forward_scalar_reference(self):
        rng=np.random.default_rng(12);x=rng.normal(size=(20,3));y=x[:,:1]
        m=Regressor(8,1).fit(x,y,epochs=4)
        w,b,v,c=m.params
        a=(x[0]-m.xmean)/m.xstd
        h=np.array([max(0,sum(a[k]*w[k,j] for k in range(3))+b[j]) for j in range(8)])
        expected=(sum(h[j]*v[j,0] for j in range(8))+c[0])*m.ystd[0]+m.ymean[0]
        self.assertAlmostEqual(m.predict(x[:1])[0,0],expected,places=12)

    def test_model_learns_and_serializes(self):
        rng=np.random.default_rng(1);x=rng.uniform(-1,1,(80,3));y=(x[:,0]+.5*x[:,1])[:,None]
        m=Regressor(12,3).fit(x,y,epochs=500)
        self.assertLess(np.sqrt(np.mean((m.predict(x)-y)**2)),.08)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'m.npz';m.save(p)
            np.testing.assert_array_equal(m.predict(x),Regressor.load(p).predict(x))

    def test_ridge_known_linear_relation(self):
        x=np.arange(30.).reshape(10,3);y=(2*x[:,0]+3)[:,None]
        np.testing.assert_allclose(ridge_predict(x,y,x,alpha=1e-8),y,atol=1e-6)

    def test_random_search_budget_and_bounds(self):
        r=random_search(lambda x:np.sum(x*x,axis=1),np.array([-2.,-3.]),np.array([2.,3.]),budget=123,initial=[0,0])
        self.assertEqual(r.evaluations,123);self.assertEqual(r.value,0)

    def test_dataset_mapping(self):
        root=Path(__file__).resolve().parents[1]
        with (root/'data/samples.csv').open(encoding='utf-8') as f:
            rows=list(csv.DictReader(f))
        v=json.loads((root/'data/variables.json').read_text(encoding='utf-8'))
        self.assertEqual(len(rows),325);self.assertEqual(len(v),23)
        self.assertEqual(len(set(x['tag'] for x in v)),23)
        self.assertEqual(v[1]['upper'],145);self.assertEqual(v[2]['upper'],900)
        self.assertTrue(all(x['tag'] in rows[0] for x in v))

if __name__=='__main__':unittest.main()
