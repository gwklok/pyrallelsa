import json
import math
import random

import jsonschema

from pyrallelsa import Problem


# https://github.com/perrygeo/simanneal/blob/master/LICENSE.txt
# https://github.com/perrygeo/simanneal/blob/master/examples/salesman.py


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
        for kb, vb in cities.items():
            if kb == ka:
                distance_matrix["{},{}".format(ka, kb)] = 0.0
            else:
                distance_matrix["{},{}".format(ka, kb)] = distance(va, vb)
    return distance_matrix


class TSPProblem(Problem):
    """Traveling Salesman Problem Annealer

    :param State|None state: state of the current annealer process; if this
        is None, a new random state will be generated
    """

    # We have this because a distance matrix grows exponentially
    #  and your RAM will die
    MAX_CITIES_FOR_DISTANCE_MATRIX = 150

    def __init__(self, state=None, **problem_data):
        self._validate_problem_data(problem_data)

        self.cities = problem_data["cities"]
        self.locked_range = int(problem_data.get("locked_range", 2))

        start_city = problem_data.get("start_city")
        self.start_city = self.cities[0] if start_city is None else start_city
        assert self.start_city in self.cities

        if not problem_data.get("updates_enabled", False):
            self.update = lambda *args, **kwargs: None
        if len(self.cities) < self.MAX_CITIES_FOR_DISTANCE_MATRIX:
            self.distance_matrix = get_distance_matrix(self.cities)
            print("Using distance matrix")
        else:
            self.distance_matrix = None

        if state is None:
            state = self.cities.keys()
            random.shuffle(state)
        self.copy_strategy = "slice"
        super(TSPProblem, self).__init__(state)  # important!

    def _validate_problem_data(self, problem_data):
        schema = {
            "type": "object",
            "properties": {
                "cities": {"type": "object"},
                "start_city": {"type": "string"},
                "updates_enabled": {"type": "boolean"},
                "locked_range": {"type": "number"}
            },
            "required": ["cities"]
        }
        jsonschema.validate(problem_data, schema)


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
                e += self.distance_matrix["{},{}".format(route[i-1], route[i])]
        else:
            for i in range(len(route)):
                e += distance(self.cities[route[i-1]], self.cities[route[i]])
        return e

    @classmethod
    def divide(cls, divisions, problem_data):
        """divide

        :type problem_data: dict
        """
        tspp = TSPProblem(**problem_data)

        def routes_for_subgroup(cs):
            for city in cs:
                if city == tspp.start_city:
                    continue
                cities = tspp.cities.keys()
                cities.remove(tspp.start_city)
                cities.remove(city)
                random.shuffle(cities)
                route = [tspp.start_city, city] + cities
                assert len(set(route)) == len(route)
                assert len(route) == len(tspp.cities)
                yield json.dumps(route)

        if divisions:
            chunk_size = int(math.ceil(len(tspp.cities) / divisions))
        else:
            chunk_size = 1
        for subgroup in chunks(tspp.cities.keys(), chunk_size):
            routes = list(routes_for_subgroup(subgroup))
            if routes:
                yield routes


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]
