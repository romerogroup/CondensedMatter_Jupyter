import numpy as np


class PolynomialDerivative(object):
    def __init__(self, poly_feature, coef, i):
        dpowers = poly_feature.powers_
        self.coef_ = coef * dpowers[:, i]
        dpowers[:, i][dpowers[:, i] > 0] -= 1
        self.powers_ = dpowers

    def predict(self, inp):
        inp = np.array(inp)
        return np.dot(self.coef_, np.product(inp**self.powers_, axis=1))

    def get_feature_names(self):
        features = []
        for i, p in enumerate(self.powers_):
            f = ''
            n = np.where(p > 0)[0]
            for nn in n:
                f += "x%s^%s " % (nn, p[nn])
            if f == '':
                f = 1
            features.append(f)
        return features

    def print_info(self, i):
        for c, f in zip(self.coef_[i], self.get_feature_names()):
            if (abs(c) > 1e-5):
                print(c, f)
