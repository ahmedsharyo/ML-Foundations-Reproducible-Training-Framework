import numpy as np

from ml_foundations.experiments.gradient_check import check_gradient
from ml_foundations.models.numpy_models import LinearRegressionGD, LogisticRegressionGD


def test_linear_forward_and_gradient():
    X=np.array([[1.,2.],[2.,1.]]); m=LinearRegressionGD(epochs=2).fit(X,np.array([5.,4.])); assert m.predict(X).shape==(2,)
    assert check_gradient("linear")["passed"]

def test_logistic_stability_and_gradient():
    assert np.isfinite(LogisticRegressionGD.sigmoid(np.array([-1000.,1000.]))).all()
    assert check_gradient("logistic")["passed"]
