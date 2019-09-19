import abc
import copy
import math
import random
import statistics
import time

from colorama import Fore, Style
from itertools import chain, permutations
from functools import total_ordering
from metaheuristics import utils


@total_ordering
class Metaheuristic(abc.ABC):

    abort_iter = 0.2

    def __init__(self, problem, *args, **kwargs):
        super().__init__()
        kwargs.setdefault("n_iters", 5000 * problem.difficulty)
        kwargs.setdefault("search_operator", utils.pitch)
        self.problem = problem
        self.config = kwargs

    def __repr__(self):
        return f"{self.__class__.__name__}(problem={self.problem!r}, **{self.config})"

    def __str__(self):
        if not self.best_population:
            return "[] - Not Run Yet"
        l = [
            f"\n{self.__class__.__name__}\n",
            f"Duration: {round(self.duration, 2)} sec",
            f"Rate: {round(self.rate, 2)} iter/sec",
            f"Cache Stats: {self.cache_info}",
            (
                f"Iterations: {self.iteration}/{self.n_iters}"
                f" (Aborted @ {int(self.abort_iter * self.n_iters)})"
                if self.aborted
                else ""
            ),
            f"\nBest Population:",
        ]
        for i in self.best_population:
            if i == self.best_solution:
                l.append(Fore.GREEN + f"\n{i.sequence} - {round(i.cost, 2)} *")
                l.append(Style.RESET_ALL)
            else:
                l.append(f"{i.sequence} - {round(i.cost, 2)}")
        return "\n".join(l)

    def __iter__(self):
        self.population = []
        self.best_population = []
        self.rep = []
        self.bests = []
        self.iteration = 0
        self.aborted = False
        self.no_impr = 0
        self.start = time.time()
        return self

    def __next__(self):
        self.iteration += 1
        self.no_impr += 1

        self.population = self._run()

        if self.has_improved:
            self.no_impr = 0
            self.best_population = list(self.population)

        self.aborted = self.no_impr == self.abort_iter * self.n_iters
        if self.aborted or self.iteration >= self.n_iters:
            self.duration = time.time() - self.start
            raise StopIteration

        return self.population, self.best_solution

    def __call__(self, *args, bar=None, **kwargs):
        iterable = self if not bar else utils.bar(iterable)
        for pop, best in iterable:
            self.rep.append(list(pop))
            self.bests.append(copy.copy(best))
        self.cache_info = str(self.best_solution._cost.cache_info())
        return self

    def __getitem__(self, items):
        return list(zip(self.rep[items], self.bests[items]))

    def __len__(self):
        return self.n_iters

    def __eq__(self, other):
        return (
            self.__class__.__name__ == other.__class__.__name__
            and self.config == other.config
            and self.best_solution == other.best_solution
        )

    def __lt__(self, other):
        return self.best_solution < other.best_solution

    def __gt__(self, other):
        return self.best_solution > other.best_solution

    def __sub__(self, other):
        return self.best_solution - other.best_solution

    @abc.abstractmethod
    def _run(self):
        ...

    @property
    def config(self):
        return self.__config

    @config.setter
    def config(self, value):
        self.__config = value
        for k, v in self.__config.items():
            setattr(self, k, v)

    @property
    def population(self):
        if not self.__population:
            self.__population = [
                self.problem.random_solution for _ in range(self.pop_size)
            ]
            self.best_population = list(self.__population)
        return self.__population

    @population.setter
    def population(self, value):
        self.__population = value

    @property
    def best_solution(self):
        return min(self.best_population, default=None)

    @property
    def worst_solution(self):
        return max(self.best_population, default=None)

    @property
    def unique_solutions(self):
        return {s for r in self.rep for s in r}

    @property
    def saturation(self):
        return utils.uniqueness(self.population)

    @property
    def has_improved(self):
        return min(self.population) < self.best_solution

    @property
    def rate(self):
        return sum([len(r) for r in self.rep]) / self.duration

    @property
    def plot(self):
        utils.plot(self.bests)
        utils.show()


class Genetic(Metaheuristic):
    def __init__(self, problem, *args, **kwargs):
        kwargs.setdefault("pop_size", problem.difficulty)
        kwargs.setdefault("crossover", utils.discrete_order_crossover)
        kwargs.setdefault("prob_mutation", 0.2)
        kwargs.setdefault("search_operator", utils.pitch)
        super().__init__(problem, *args, **kwargs)

    def mate(self, p1, p2):
        o1, o2 = self.crossover(p1, p2)
        o1 = self.problem.solution(sequence=o1)
        o2 = self.problem.solution(sequence=o2)

        if random.random() < self.prob_mutation:
            o1 = self.search_operator(o1)
            o2 = self.search_operator(o2)

        while not all((o1.correct, o2.correct)):
            o1, o2 = self.crossover(p1, p2)
            o1 = self.problem.solution(sequence=o1)
            o2 = self.problem.solution(sequence=o2)
        return o1, o2

    def _run(self):
        pop = []
        mating_pool = utils.roulette_wheel(self.population)
        for i, j in utils.pairwise(mating_pool, step=2):
            a = self.population[i]
            b = self.population[j]
            c, d = self.mate(a, b)
            pop.extend([c, d])
        return pop


class SimulatedAnnealing(Metaheuristic):

    startprob = 0.1
    endprob = 0.02
    pop_size = 1
    recursive = False

    def __init__(self, problem, *args, **kwargs):
        kwargs.setdefault("neighborhood_size", 2 * problem.difficulty)
        super().__init__(problem, *args, **kwargs)

    def calibrate(self):

        current = self.problem.random_solution
        pop = [current]
        deltas = []
        for _ in range(self.n_iters):
            new = self.search_operator(current)  #  utils.pitch(current)
            pop.append(new)

            worst = new > current
            if worst:
                deltas.append(new - current)
            else:
                current = new

        mu = statistics.mean(deltas)
        sigma = statistics.stdev(i.cost for i in pop)
        self.ti = -mu / math.log(self.startprob)
        self.tf = -sigma / math.log(self.endprob)
        self.dt = (self.ti - self.tf) / self.n_iters

    def __iter__(self):
        self.calibrate()
        return super().__iter__()

    @property
    def temp(self):
        return self.ti - self.dt * self.iteration

    def accept(self, sol):
        delta = self.best_solution - sol
        return random.random() < math.exp(delta / self.temp)

    def _run(self):
        pop = self.population
        current = pop[0]

        neighborhood = []
        for _ in range(self.neighborhood_size):
            new = self.search_operator(current)  #  utils.pitch(current)

            if new < self.best_solution:
                return [new]

            neighborhood.append(new)
            if self.recursive:
                # current = copy.copy(new)
                current = new

        best_neighbor = min(neighborhood)

        return [best_neighbor] if self.accept(best_neighbor) else pop


class HarmonySearch(Metaheuristic):

    par = 0.1

    def __init__(self, problem, *args, **kwargs):
        kwargs.setdefault("pop_size", problem.difficulty)
        super().__init__(problem, *args, **kwargs)

    def harmony_memory_consideration(self):
        harmony_memory = self.population

        new_harmony = []
        end = len(harmony_memory[0])
        start = random.randint(0, end - 1)
        seq = list(range(start, end)) + list(range(0, start))

        for pos in seq:
            rw = utils.roulette_wheel(harmony_memory)
            notes = [harmony_memory[i][pos] for i in rw]
            notes = [n for n in notes if n not in new_harmony]
            if notes:
                note = notes[0]
            else:
                poss = list(set(range(end)) - set(new_harmony))
                note = random.choice(poss)
            new_harmony.append(note)

        new_harmony = new_harmony[start:] + new_harmony[:start]
        new_harmony = self.problem.solution(sequence=new_harmony)

        while not new_harmony.correct:
            new_harmony = self.harmony_memory_consideration()

        return new_harmony

    def _run(self):

        harmony_memory = self.population
        new_harmony = self.harmony_memory_consideration()

        if random.random() < self.par:
            new_harmony = self.search_operator(new_harmony)  #  utils.pitch(new_harmony)
        if new_harmony < self.worst_solution:
            harmony_memory[utils.argmax(harmony_memory)] = new_harmony

        return harmony_memory


class ModifiedHarmonySearch(Genetic, HarmonySearch):

    low_lprr = 0.09
    high_lprr = 0.9
    low_sat = 0.2
    high_sat = 0.8

    def __init__(self, problem, *args, **kwargs):
        super().__init__(problem, *args, **kwargs)

    @property
    def lprr(self):
        if self.saturation < self.low_sat:
            lprr = self.low_lprr
        elif self.saturation > self.high_sat:
            lprr = 0.0
        else:
            lprr = self.high_lprr
        return lprr

    def mate(self, p1, p2):
        return super().mate(p1, p2)

    def large_portion_recovery(self):
        return Genetic._run(self)

    def _run(self):

        if random.random() < self.lprr:
            pop = self.large_portion_recovery()
        else:
            pop = HarmonySearch._run(self)

        return pop


class ModifiedHarmonySearch_v2(ModifiedHarmonySearch):
    def __init__(self, problem, *args, **kwargs):
        super().__init__(problem, *args, **kwargs)

        r = (self.low_sat / self.high_sat) ** 3
        self.c = (self.low_lprr - r) / (1 - r)
        self.m = (1 / (self.high_sat ** 3)) * ((1 - self.low_lprr) / (1 - r))

    @property
    def lprr(self):
        if self.low_sat < self.saturation < self.high_sat:
            return self.c + self.m * (self.saturation) ** 3
        return 0.0

    def large_portion_recovery(self):
        mating_pool = list(set(self.population))
        n_uniques = len(mating_pool)
        n_copies = len(self.population) - n_uniques
        repeat = round(n_copies / n_uniques) or 1

        roulette_wheel = chain(utils.roulette_wheel(mating_pool) * repeat)

        children = []
        for i, j in utils.pairwise(roulette_wheel, step=2):
            a = mating_pool[i]
            b = mating_pool[j]
            c, d = self.mate(a, b)
            children.extend([c, d])

        #  indici delle copie nell'hm
        idx = [n for n, x in enumerate(self.population) if x in self.population[:n]]
        for i, child in zip(idx, children):
            self.population[i] = child

        return self.population


class CuckooSearch(Metaheuristic):

    levy_alpha = 2
    abort_flight = True
    real_flight = True
    p_rgn = 0.1

    def __init__(self, problem, *args, **kwargs):
        kwargs.setdefault("pop_size", problem.difficulty)
        kwargs.setdefault("levy_min", 2 * problem.difficulty)
        super().__init__(problem, *args, **kwargs)

    @property
    def levy_lenght(self):
        u = random.random()
        return int(self.levy_min * (1 - u) ** (-1 / self.levy_alpha))

    def fly(self, sol):

        flight = []
        current = copy.copy(sol)
        for _ in range(self.levy_lenght):
            new = self.search_operator(current)  #  utils.pitch(current)
            if self.real_flight:
                current = new
            if self.abort_flight and new < sol:
                return [new]
            flight.append(new)
        return flight

    def _run(self):

        pop = self.population

        which_cuckoo = utils.roulette_wheel(pop, size=1)[0]
        cuckoo = pop[which_cuckoo]
        flight = self.fly(cuckoo)
        new = min(flight)

        if new < cuckoo:
            pop[which_cuckoo] = new
        else:
            i = random.randint(0, len(self.population) - 1)
            random_cuckoo = pop[i]
            if new < random_cuckoo:
                pop[i] = new

        split = int(self.p_rgn * len(self.population))
        for i in range(split, len(self.population)):
            pop[i] = self.problem.random_solution

        return pop


class ModifiedCuckooSearch(CuckooSearch, SimulatedAnnealing):

    p_split = 0.1
    epoch = 500

    def __init__(self, problem, *args, **kwargs):
        super().__init__(problem, *args, **kwargs)

    def __iter__(self):
        self = super().__iter__()
        self.starting_points = copy.deepcopy(self.population)
        self.useless_attempts = [0] * len(self.population)
        return self

    def _run(self):

        pop = self.population

        if self.iteration % self.epoch == 0:
            split = int(len(pop) * self.p_split)
            bests = pop[:split]
            for i, sol in enumerate(bests):
                new = self.search_operator(sol)  #  utils.pitch(sol)
                idx = -split + i
                pop[idx] = copy.copy(new)
                self.starting_points[idx] = copy.copy(new)

        if not sum(self.useless_attempts):
            which_cuckoo = random.randint(0, len(pop) - 1)
        else:
            which_cuckoo = utils.roulette_wheel(self.useless_attempts, size=1)[0]
        cuckoo = self.starting_points[which_cuckoo]
        flight = self.fly(cuckoo)
        new = min(flight)
        self.useless_attempts[which_cuckoo] += 1

        if new < pop[which_cuckoo]:
            pop[which_cuckoo] = new
            self.starting_points[which_cuckoo] = new
            self.useless_attempts[which_cuckoo] = 0
        elif self.accept(new):
            self.starting_points[which_cuckoo] = new

        return pop


class Exaustive(Metaheuristic):
    def __init__(self, problem, *args, **kwargs):
        super().__init__(problem, *args, **kwargs)

    def __call__(self):

        self.start = time.time()
        self.rep = []

        j = self.problem.n_jobs
        m = self.problem.n_maintenance
        all_sequences = permutations(range(j + m))
        sequence = next(all_sequences)
        best = self.problem.solution(sequence=sequence)
        while not best.correct:
            sequence = next(all_sequences)
            best = self.problem.solution(sequence=sequence)

        self.rep.append(best)
        for sequence in all_sequences:
            new = self.problem.solution(sequence=sequence)
            if new.correct and new < best:
                best = new
                self.rep.append(best)

        self.population = [best]
        self.best_population = [best]

        self.duration = time.time() - self.start
        self.cache_info = ""
        self.aborted = False

        return self

    def _run(self):
        ...


class ParticleSwarmOptimization(Metaheuristic):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()
