import random
import math
import json

from pyrallelsa import Annealer
from pyrallelsa import ProblemSet
from pyrallelsa import ProblemClassPath


def distance(a, b):
    """Calculates distance between two latitude-longitude coordinates."""
    R = 3963  # radius of Earth (miles)
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
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
        self.locked_range = problem_data["locked_range"]
        if not problem_data["updates_enabled"]:
            self.update = lambda *args, **kwargs: None
        self.copy_strategy = "slice"
        super(TSPProblem, self).__init__(state)  # important!

    def move(self, state=None):
        """Swaps two cities in the route.

        :type state: TSPState
        """
        state = self.state if state is None else state
        route = state
        a = random.randint(self.locked_range, len(route) - 1)
        b = random.randint(self.locked_range, len(route) - 1)
        route[a], route[b] = route[b], route[a]

    def energy(self, state=None):
        """Calculates the length of the route."""
        state = self.state if state is None else state
        route = state
        e = 0
        if self.distance_matrix:
            for i in range(len(route)):
                e += self.distance_matrix[route[i-1]][route[i]]
        else:
            for i in range(len(route)):
                e += distance(self.cities[route[i-1]], self.cities[route[i]])
        return e


class TSPProblemSet(ProblemSet):

    # We have this because a distance matrix grows exponentially
    #  and your RAM will die
    MAX_CITIES_FOR_DISTANCE_MATRIX = 150

    def __init__(self, cities, start_city=None, updates_enabled=False):
        self.cities = cities
        self.start_city = self.cities[0] if start_city is None else start_city
        assert self.start_city in self.cities
        if len(cities) < self.MAX_CITIES_FOR_DISTANCE_MATRIX:
            self.distance_matrix = get_distance_matrix(cities)
        else:
            self.distance_matrix = None
        self._problem_data = {"cities": self.cities,
                              "distance_matrix": self.distance_matrix,
                              "updates_enabled": updates_enabled,
                              "locked_range": 2}
        self._problem_data_str = json.dumps(self._problem_data)

    @property
    def problem_data_str(self):
        return self._problem_data_str

    @property
    def pcp(self):
        return ProblemClassPath("tsp", "TSPProblem")

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
            yield route

    # def divide(self):
    #     yield self.cities.keys()
