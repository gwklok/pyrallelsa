#!/usr/bin/env python

import json

import click
import time

from pyrallelsa import ParallelSAManager, ProblemClassPath, runner
from pyrallelsa.examples.tsp.cities import cities_120, cities_20, cities_100,\
    pd_cities_120_alt, pd_cities_150_alt


@click.command()
@click.option('-t', '--type', type=click.Choice(['seq', 'mc']), default='seq')
def main(type):
    # cities = cities_100
    # problem_data = dict(
    #     cities=cities,
    #     start_city="Bakous",
    #     updates_enabled=True
    # )
    problem_data = pd_cities_150_alt
    cities = pd_cities_150_alt['cities']
    pcp = ProblemClassPath("pyrallelsa.examples.tsp", "TSPProblem")

    if type == 'mc':
        psam = ParallelSAManager(pcp, problem_data)
        psam.run(minutes=2.0, cpus=3)
    elif type == 'seq':
        start = time.time()
        args = (0, pcp, json.dumps(cities.keys()), 10.0,
                json.dumps(problem_data), None)
        winner = runner(args)
        print("With an energy of {}; {} was the best.".format(
            winner.energy,
            winner.state
        ))
        print("Run took {}s".format(time.time() - start))


if __name__ == '__main__':
    main()
