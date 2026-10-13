import math
import time

class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def alpha(self, t_e, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / t_e)

    def __call__(self, t, x):
        t_e = t - self.t_prev

        if t_e > 0:
            a_d = self.alpha(t_e, self.dcutoff)
            dx = (x - self.x_prev) / t_e
            dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)
        else:
            dx_hat = self.dx_prev

        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self.alpha(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev
