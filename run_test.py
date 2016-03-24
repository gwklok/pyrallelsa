import json

import sys

from tsp import TSPProblemSet
from pyrallelsa import ParallelSAManager
from cities import cities_20, cities_7, cities_120

from tsp import TSPProblem, get_distance_matrix

def main():
    p = TSPProblemSet(cities_120, start_city="Houston", updates_enabled=True)
    psam = ParallelSAManager(p)
    psam.run(minutes=6.0, cpus=2)

    # cities = cities_120
    # initial_state = cities.keys()
    # t = TSPProblem(initial_state, json.dumps(dict(
    #     distance_matrix=get_distance_matrix(cities_120),
    #     cities=cities,
    #     updates_enabled=True,
    #     locked_range=0
    # )))

    # sys.path.append("../Capstone/mesos-magellan/traveling-sailor")
    # from problem import Problem
    # t = Problem(None, cities.keys(), cities)

    # s = t.auto(minutes=5.0)
    # t.set_schedule(s)
    # bs, be = t.anneal()
    # print(bs)
    # print(be)


if __name__ == '__main__':
    main()
