from tsp import TSPProblem
from pyrallelsa import ParallelSAManager
from cities import cities_20, cities_7, cities_120, cities_999


def main():
    problem_data = dict(
        cities=cities_120,
        start_city="Houston",
        updates_enabled=False
    )
    psam = ParallelSAManager(TSPProblem.pcp(), problem_data)
    psam.run(minutes=2.0, cpus=3)


if __name__ == '__main__':
    main()
