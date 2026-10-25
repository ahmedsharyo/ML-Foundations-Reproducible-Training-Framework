import numpy as np

from ..models.numpy_models import LinearRegressionGD, LogisticRegressionGD


DEFAULT_EPSILON = 1e-6
RANDOM_SEED = 7
NUM_SAMPLES = 5
NUM_FEATURES = 3
INITIAL_BIAS = 0.2
L2_REGULARIZATION = 0.07
ERROR_TOLERANCE = 1e-5


# Gradient checking compares analytical gradients against numerical approximations using finite differences.
def check_gradient(kind, epsilon=DEFAULT_EPSILON):
    rng = np.random.default_rng(RANDOM_SEED)
    X = rng.normal(size=(NUM_SAMPLES, NUM_FEATURES))
    y = rng.integers(0, 2, size=NUM_SAMPLES).astype(float)
    w = rng.normal(size=NUM_FEATURES)
    b = INITIAL_BIAS
    cls = LinearRegressionGD if kind == "linear" else LogisticRegressionGD
    loss, dw, db = cls.loss_and_gradients(X, y, w, b, L2_REGULARIZATION)
    numeric = np.zeros_like(w)
    for i in range(len(w)):
        wp = w.copy()
        wm = w.copy()
        wp[i] += epsilon
        wm[i] -= epsilon
        numeric[i] = (cls.loss_and_gradients(X, y, wp, b, L2_REGULARIZATION)[0] - cls.loss_and_gradients(X, y, wm, b, L2_REGULARIZATION)[0]) / (2 * epsilon)
    nb = (cls.loss_and_gradients(X, y, w, b + epsilon, L2_REGULARIZATION)[0] - cls.loss_and_gradients(X, y, w, b - epsilon, L2_REGULARIZATION)[0]) / (2 * epsilon)
    err = max(float(np.max(np.abs(dw - numeric))), abs(db - nb))
    return {"max_abs_error": err, "passed": err < ERROR_TOLERANCE}
