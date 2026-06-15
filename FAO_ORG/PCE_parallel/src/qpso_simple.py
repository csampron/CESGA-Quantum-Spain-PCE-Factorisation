import numpy as np
import random

# -------------------- Particle y Swarm base -------------------- #
class Particle(object):
    def __init__(self, bounds):
        self._x = np.zeros(len(bounds))
        for idx, (lo, hi) in enumerate(bounds):
            self._x[idx] = random.uniform(lo, hi)
        self._best = self._x.copy()
        self._best_value = np.NaN

    def __getitem__(self, key):
        return self._x[key]

    def __setitem__(self, key, val):
        self._x[key] = val

    @property
    def best(self):
        return self._best

    def set_best(self, x):
        self._best = x.copy()

    @property
    def best_value(self):
        return self._best_value

    def set_best_value(self, v):
        self._best_value = v


class Swarm(object):
    def __init__(self, size, dim, bounds):
        self._particles = [Particle(bounds) for _ in range(size)]
        self._dim = dim
        self._bounds = bounds
        self._gbest = None
        self._gbest_value = None

    def particles(self):
        return self._particles

    def update_gbest(self):
        pg = min(self._particles, key=lambda p: p.best_value)
        if (self._gbest_value is None) or (pg.best_value < self._gbest_value):
            self._gbest = pg.best.copy()
            self._gbest_value = pg.best_value

    @property
    def gbest(self):
        return self._gbest

    @property
    def gbest_value(self):
        return self._gbest_value

class QPSO(Swarm):
    def __init__(self, cf, size, dim, bounds, maxIters):
        super().__init__(size, dim, bounds)
        self._cf = cf
        self._maxIters = maxIters
        self._iters = 0
        self.init_eval()

    @property
    def iters(self):
        return self._iters

    def init_eval(self):
        for p in self._particles:
            p.set_best_value(self._cf(p[:]))
        self.update_gbest()

    def update_best(self):
        for p in self._particles:
            v = self._cf(p[:])
            if np.isnan(p.best_value) or v < p.best_value:
                p.set_best(p[:])
                p.set_best_value(v)
        self.update_gbest()

    def kernel_update(self):
        for p in self._particles:
            for i in range(self._dim):
                u1 = random.uniform(0., 1.)
                u2 = random.uniform(0., 1.)
                u3 = random.uniform(0., 1.)
                rand_sign = 1 if random.random() > 0.5 else -1
                c = (u1 * p.best[i] + u2 * self.gbest[i]) / (u1 + u2)
                L = abs(p[i] - c)
                p[i] = c + rand_sign * L * np.log(1. / u3)

                # Enforce bounds
                lo, hi = self._bounds[i]
                p[i] = np.clip(p[i], lo, hi)

    def update(self, callback=None, interval=None,
               base_loss_func=None, verbose=False):

        while self._iters < self._maxIters:

            self.kernel_update()
            self.update_best()

            if callback and interval and self._iters % interval == 0:
                callback(self)

            # ---- NUEVO CRITERIO DE PARADA ----
            if base_loss_func is not None and self._gbest is not None:
                if base_loss_func(self._gbest) == 0:
                    if verbose:
                        print(f"\nBase loss alcanzó 0 en iteración {self._iters}.")
                    break

            self._iters += 1

class QDPSO(QPSO):
    def __init__(self, cf, size, dim, bounds, maxIters, g):
        super().__init__(cf, size, dim, bounds, maxIters)
        self._g = g

    def kernel_update(self):
        for p in self._particles:
            for i in range(self._dim):
                u1 = random.uniform(0., 1.)
                u2 = random.uniform(0., 1.)
                u3 = random.uniform(0., 1.)
                rand_sign = 1 if random.random() > 0.5 else -1
                c = (u1 * p.best[i] + u2 * self.gbest[i]) / (u1 + u2)
                L = (1 / self._g) * abs(p[i] - c)
                p[i] = c + rand_sign * L * np.log(1. / u3)

                lo, hi = self._bounds[i]
                p[i] = np.clip(p[i], lo, hi)

def minimize_qdpso(loss_func,
                   base_loss_func,
                   x0,
                   bounds,
                   num_particles=20,
                   maxiter=1000,
                   g=0.96,
                   verbose=False):

    dim = len(x0)
    qpso = QDPSO(loss_func, num_particles, dim, bounds, maxiter, g)
    iteration_costs = []

    def cb(opt):
        iteration_costs.append(opt.gbest_value)
        if verbose:
            print(f"Iter {opt.iters}: best_value = {opt.gbest_value}")

    qpso.update(callback=cb,
                interval=1,
                base_loss_func=base_loss_func,
                verbose=verbose)

    best_position = np.atleast_1d(qpso.gbest)
    best_value = qpso.gbest_value

    return best_value, best_position, iteration_costs