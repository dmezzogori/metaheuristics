
import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
import progressbar
import random
import statistics

from itertools import islice, tee, groupby
from multiprocessing import Pool


def worker(args):
    solver, _, n = args
    print(f"{solver.__class__.__name__} [{_+1}/{n}]")
    solver()
    return copy.deepcopy(solver)


class BaseSolutionMixin:
    
    def __repr__(self):
        return f"{self.sequence} - {round(self.cost, 2)}"

    def __getitem__(self, index):
        return self.sequence[index]

    def __setitem__(self, index, value):
        self.sequence[index] = value
        self.__cost = None

    def __hash__(self):
        return hash(tuple(self.sequence))

    def __eq__(self, other):
        return self.sequence == other.sequence

    def __lt__(self, other):
        return self.cost < other.cost

    def __gt__(self, other):
        return self.cost > other.cost

    def __len__(self):
        return len(self.sequence)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return self.cost + other
        return self.cost + other.cost

    def __radd__(self, other):
        return Solution.__add__(self, other)

    def __sub__(self, other):
        return self.cost - other.cost

    def __rsub__(self, other):
        return Solution.__sub__(self, other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self.cost / other
        return self.cost / other.cost


class Race:
    def __init__(self, solvers):
        self.solvers = solvers
        self.__results = []

    def __call__(self, n=2, parallel=False):
        self.n = n
        self.results = self._parallel() if parallel else self._serial()
        return self

    def __repr__(self):
        self.plot
        s = []
        grouper = groupby(self.results, key=lambda x: x.__class__.__name__)
        for _, group in grouper:
            run = list(group)
            s.append(_)
            s.append(
                f"""Min: {
                round(min(i.best_solution.cost for i in run), 2)}"""
            )
            s.append(
                f"""Mean: {
                round(statistics.mean(
                    i.best_solution.cost for i in run), 2)}"""
            )
            s.append(
                f"""Std: {
                round(statistics.stdev(
                    i.best_solution.cost for i in run), 2)}\n"""
            )
        return "\n".join(s)

    def __getitem__(self, value):
        return list(filter(lambda x: x[0].__class__.__name__ == value, self.results))

    @property
    def todo(self):
        for solver in self.solvers:
            for _ in range(self.n):
                yield (solver, _, self.n)

    def _parallel(self):

        results = []
        pool = Pool()
        multiple_results = pool.map_async(worker, self.todo, callback=results.extend)
        multiple_results.wait()

        return results

    def _serial(self):
        return [worker(args) for args in self.todo]

    @property
    def results(self):
        return self.__results

    @results.setter
    def results(self, value):
        self.__results = value

    @property
    def bests(self):
        res = sorted(self.results, key=lambda x: x.__class__.__name__)
        grouper = groupby(res, key=lambda x: x.__class__.__name__)
        return [min(trials) for _, trials in grouper]

    @property
    def plot(self):
        plot(self.bests)
        show()

    def dump(self):
        with open("race.pkl", "wb") as f:
            pickle.dump(self, f)


def bar(iterable):
    bar = progressbar.ProgressBar()
    return bar(iterable)


def generate_cut_points(cut_points, n):
    if cut_points:
        cut1, cut2 = cut_points
    else:
        high = n - 1
        mid_point = int(high / 2)
        cut1 = random.randint(1, mid_point)
        cut2 = random.randint(mid_point + 1, high)

    return cut1, cut2


def discrete_order_crossover(p1, p2, cut_points=None):
    cut1, cut2 = generate_cut_points(cut_points, len(p1))

    center1 = p1[cut1:cut2]
    center2 = p2[cut1:cut2]

    remainder_p2 = [i for i in p2 if i not in center1]
    left_p2, right_p2 = remainder_p2[:cut1], remainder_p2[cut1:]
    
    o1 = left_p2 + center1 + right_p2
    
    remainder_p1 = [i for i in p1 if i not in center2]
    left_p1, right_p1 = remainder_p1[:cut1], remainder_p1[cut1:]
    
    o2 = left_p1 + center2 + right_p1
    
    return o1, o2


def continuous_order_crossover(p1, p2, cut_points=None):
    cut1, cut2 = generate_cut_points(cut_points, len(p1))
        
    o1 = p2[:cut1] + p1[cut1:cut2] + p2[cut2:]
    o2 = p1[:cut1] + p2[cut1:cut2] + p1[cut2:]
    
    return o1, o2


def nthwise(iterable, n, step=1):
    """
    step: sliding window offset
    n: items per set

    step=1, n=2, s -> (s0,s1), (s1,s2), (s2,s3), ...
    step=2, n=2, s -> (s0,s1), (s2,s3), (s4,s5), ...
    step=2, n=3, s -> (s0,s1,s2), (s2,s3,s4), (s4,s5,s6), ...
    """
    iterables = tee(iterable, n)
    for i, it in enumerate(iterables):
        next(islice(it, i, i), None)
    return zip(*(islice(it, None, None, step) for it in iterables))


def pairwise(iterable, step=1):
    return nthwise(iterable, step=step, n=2)


def weights(iterable):
    iterable = tuple(math.exp(e) for e in iterable)
    tot = sum(e for e in iterable)
    return [e / tot for e in iterable]


def pitch(sol, pos1=None, pos2=None):
    a, b = roulette_wheel(range(len(sol)), size=2)
    pos1 = pos1 or a
    pos2 = pos2 or b
    pitched = copy.copy(sol)
    pitched_seq = list(pitched.sequence)
    pitched_seq[pos1], pitched_seq[pos2] = pitched_seq[pos2], pitched_seq[pos1]
    pitched.sequence = pitched_seq
    while not pitched.correct:
        pitched = pitch(sol)
    return pitched


def tweak(n, length, mu, sigma):

    def inner(sol):

        tweaked = copy.copy(sol)
        tweaked_seq = list(tweaked.sequence)
        for i in range(n):
            pos = random.randint(0, length-1)
            tweaked_seq[pos] = tweaked_seq[pos] + random.gauss(mu, sigma)
        tweaked.sequence = tweaked_seq

        return tweaked

    return inner


def roulette_wheel(iterable, size=None, replace=False):
    return np.random.choice(
        len(iterable), size=size or len(iterable), replace=replace, p=weights(iterable)
    ).tolist()


def uniqueness(iterable):
    l = len(iterable)
    return 0.0 if l == 1 else (l - len(set(iterable))) / (l - 1)


def argmin(iterable):
    return iterable.index(min(iterable))


def argmax(iterable):
    return iterable.index(max(iterable))


def dist(iterable_a, iterable_b, method="eucl"):

    if method not in ("eucl", "manhattan"):
        raise

    if method == "eucl":
        dist = math.sqrt(sum([(i - j) ** 2 for i, j in zip(iterable_a, iterable_b)]))
    elif method == "manhattan":
        dist = sum(abs(i - j) for i, j in zip(iterable_a, iterable_b))

    return dist


def plot(solvers):
    if not isinstance(solvers, list):
        solvers = [solvers]
    for solver in solvers:
        plt.plot([i.cost for i in solver.bests], label=solver.__class__.__name__)
    plt.legend(
        bbox_to_anchor=(0., 1.02, 1., .102),
        loc=3,
        ncol=2,
        mode="expand",
        borderaxespad=0.,
    )
