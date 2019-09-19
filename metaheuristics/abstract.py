import abc


class Problem(abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def config(self):
        ...

    @property
    @abc.abstractmethod
    def random_solution(self):
        ...

    @abc.abstractmethod
    def solution(self):
        ...

    @abc.abstractmethod
    def reset(self):
        ...

    @abc.abstractmethod
    def difficulty(self):
        ...

    @abc.abstractmethod
    def __hash__(self):
        ...


class Solution(abc.ABC):
    def __init__(self, problem, sequence=None):
        super().__init__()
        self.problem = problem
        self.__sequence = sequence

    @abc.abstractmethod
    def __repr__(self):
        ...

    @abc.abstractmethod
    def __getitem__(self):
        ...

    @abc.abstractmethod
    def __setitem__(self):
        ...

    @abc.abstractmethod
    def __hash__(self):
        ...

    @abc.abstractmethod
    def __eq__(self):
        ...

    @abc.abstractmethod
    def __lt__(self):
        ...

    @abc.abstractmethod
    def __gt__(self):
        ...

    @abc.abstractmethod
    def __len__(self):
        ...

    @abc.abstractmethod
    def __add__(self):
        ...

    @abc.abstractmethod
    def __radd__(self):
        ...

    @abc.abstractmethod
    def __sub__(self):
        ...

    @abc.abstractmethod
    def __rsub__(self):
        ...

    @abc.abstractmethod
    def __truediv__(self):
        ...

    @property
    @abc.abstractmethod
    def sequence(self):
        ...

    @sequence.setter
    @abc.abstractmethod
    def sequence(self, value):
        ...

    @property
    @abc.abstractmethod
    def is_complete(self):
        ...

    @property
    @abc.abstractmethod
    def is_partial(self):
        ...

    @abc.abstractmethod
    def _correct(self):
        ...

    @property
    @abc.abstractmethod
    def correct(self):
        ...

    @abc.abstractmethod
    def _cost(self):
        ...

    @property
    @abc.abstractmethod
    def cost(self):
        ...

    def __int__(self):
        return int(self.cost)

    def __float__(self):
        return self.cost
