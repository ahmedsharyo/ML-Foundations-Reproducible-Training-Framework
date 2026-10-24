import numpy as np
from ..models.numpy_models import LinearRegressionGD, LogisticRegressionGD

def check_gradient(kind,epsilon=1e-6):
    rng=np.random.default_rng(7); X=rng.normal(size=(5,3)); y=rng.integers(0,2,size=5).astype(float); w=rng.normal(size=3); b=.2
    cls=LinearRegressionGD if kind=="linear" else LogisticRegressionGD
    loss,dw,db=cls.loss_and_gradients(X,y,w,b,.07); numeric=np.zeros_like(w)
    for i in range(len(w)):
        wp=w.copy(); wm=w.copy(); wp[i]+=epsilon; wm[i]-=epsilon
        numeric[i]=(cls.loss_and_gradients(X,y,wp,b,.07)[0]-cls.loss_and_gradients(X,y,wm,b,.07)[0])/(2*epsilon)
    nb=(cls.loss_and_gradients(X,y,w,b+epsilon,.07)[0]-cls.loss_and_gradients(X,y,w,b-epsilon,.07)[0])/(2*epsilon)
    err=max(float(np.max(np.abs(dw-numeric))),abs(db-nb))
    return {"max_abs_error":err,"passed":err<1e-5}
