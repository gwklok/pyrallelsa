import random
import math
import json

from pyrallelsa import Annealer
from pyrallelsa import State
from pyrallelsa import ProblemSet, ProblemClassPath, ProblemStatePath


class TSPState(State):

    def __init__(self, route, locked_range=0):
        self.route = route
        self.locked_range = locked_range

    def copy(self):
        return TSPState(route=self.route, locked_range=self.locked_range)

    def serialize(self):
        kwargs = {"route": self.route, "locked_range": self.locked_range}
        return json.dumps(kwargs)

    @classmethod
    def load(cls, s):
        return cls(**json.loads(s))


def distance(a, b):
    """Calculates distance between two latitude-longitude coordinates."""
    R = 3963  # radius of Earth (miles)
    # print("a: {} b: {}".format(a,b))
    try:
        lat1, lon1 = math.radians(a[0]), math.radians(a[1])
        lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    except:
        raise
    return math.acos(math.sin(lat1) * math.sin(lat2) +
                     math.cos(lat1) * math.cos(lat2) * math.cos(lon1 - lon2)) * R


def get_distance_matrix(cities):
    # create a distance matrix
    distance_matrix = {}
    for ka, va in cities.items():
        distance_matrix[ka] = {}
        for kb, vb in cities.items():
            if kb == ka:
                distance_matrix[ka][kb] = 0.0
            else:
                distance_matrix[ka][kb] = distance(va, vb)
    return distance_matrix


class TSPProblem(Annealer):
    """Traveling Salesman Problem Annealer
    :param dict job_data: Unused currently
    :param State state: state of the current annealer process
    """
    def __init__(self, state, problem_data):
        problem_data = json.loads(problem_data)
        self.cities = problem_data["cities"]
        self.distance_matrix = problem_data["distance_matrix"]
        super(TSPProblem, self).__init__(state)  # important!

    def move(self, state=None):
        """Swaps two cities in the route.

        :type state: TSPState
        """
        state = self.state if state is None else state
        route = state.route
        a = random.randint(state.locked_range, len(route) - 1)
        b = random.randint(state.locked_range, len(route) - 1)
        route[a], route[b] = route[b], route[a]

    def energy(self, state=None):
        """Calculates the length of the route."""
        state = self.state if state is None else state
        route = state.route
        e = 0
        for i in range(len(route)):
            e += self.distance_matrix[route[i-1]][route[i]]
        return e

    def copy_state(self, state):
        """Return copy of state

        :type state: TSPState
        :rtype: State
        """
        return state.copy()


class TSPProblemSet(ProblemSet):
    def __init__(self, cities, start_city=None):
        self.cities = cities
        self.start_city = self.cities[0] if start_city is None else start_city
        assert self.start_city in self.cities
        self.distance_matrix = get_distance_matrix(cities)
        self._problem_data = {"cities": self.cities,
                              "distance_matrix": self.distance_matrix}
        self._problem_data_str = json.dumps(self._problem_data)

    @property
    def problem_data_str(self):
        return self._problem_data_str

    @property
    def pcp(self):
        return ProblemClassPath("tsp", "TSPProblem")

    @property
    def psp(self):
        return ProblemStatePath("tsp", "TSPState")

    def divide(self):
        for city in self.cities:
            if city == self.start_city:
                continue
            cities = self.cities.keys()
            cities.remove(self.start_city)
            cities.remove(city)
            random.shuffle(cities)
            route = [self.start_city, city] + cities
            assert len(set(route)) == len(route)
            assert len(route) == len(self.cities)
            yield TSPState(route=route, locked_range=2)
