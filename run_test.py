from tsp import TSPProblemSet
from pyrallelsa import ParallelSAManager
from cities import cities_20, cities_7, cities_120

def main():
    p = TSPProblemSet(cities_7, start_city="Houston", updates_enabled=True)
    psam = ParallelSAManager(p)
    psam.run(minutes=0.5, cpus=2)

if __name__ == '__main__':
    main()
