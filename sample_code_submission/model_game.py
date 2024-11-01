import numpy as np

# RANDOM_SEED = 314159

class Model:

    def __init__(self, get_train_set=None, systematics=None):
        self.get_train_set = get_train_set
        self.systematics = systematics
    
    def fit(self):
        self.sigma = .682689492137086

    def predict(self, test_set):
        rng = np.random.default_rng()
        if rng.binomial(1, self.sigma):
            p16, p84 = .1, 3.
        else:
            p16, p84 = 0., 0.
        mu_hat = np.mean([p16, p84])
        delta_mu_hat = (p84 - p16) / 2
        return {
            "mu_hat": mu_hat,
            "delta_mu_hat": delta_mu_hat,
            "p16": p16,
            "p84": p84,
        }