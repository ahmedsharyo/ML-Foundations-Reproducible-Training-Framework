from __future__ import annotations

import json
from pathlib import Path

from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

from .gradient_check import check_gradient
from ..models.numpy_models import LinearRegressionGD, LogisticRegressionGD


def run(kind,config):
    seed=config.get("seed",42); out=Path(config.get("output_dir","results")); out.mkdir(parents=True,exist_ok=True)
    if kind=="linear": X,y=make_regression(n_samples=1000,n_features=8,noise=10,random_state=seed); Model=LinearRegressionGD
    else: X,y=make_classification(n_samples=1000,n_features=8,n_informative=5,random_state=seed); Model=LogisticRegressionGD
    a,b,c,d=train_test_split(X,y,test_size=.2,random_state=seed,stratify=y if kind!="linear" else None)
    model=Model(**config.get("training",{})).fit(a,b)
    baseline=LinearRegression().fit(a,b) if kind=="linear" else LogisticRegression(max_iter=1000).fit(a,b)
    score=model.score(c,d); base=baseline.score(c,d)
    report={"model":kind,"seed":seed,"test_score":score,"sklearn_score":float(base),"gradient_check":check_gradient(kind)}
    (out/f"{kind}.json").write_text(json.dumps(report,indent=2),encoding="utf8"); print(json.dumps(report,indent=2))
