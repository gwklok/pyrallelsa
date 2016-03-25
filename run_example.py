from pyrallelsa import ParallelSAManager, ProblemClassPath
from pyrallelsa.examples.tsp.cities import cities_120


def main():
    problem_data = dict(
        cities=cities_120,
        start_city="Houston",
        updates_enabled=False
    )
    pcp = ProblemClassPath("pyrallelsa.examples.tsp", "TSPProblem")
    psam = ParallelSAManager(pcp, problem_data)
    psam.run(minutes=2.0, cpus=3)


if __name__ == '__main__':
    main()
