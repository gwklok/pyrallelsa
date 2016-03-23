from tsp import TSPProblemSet
from pyrallelsa import ParallelSAManager
from cities import cities_20, cities_7, cities_120

def main():
    p = TSPProblemSet(cities_120, start_city="Houston")
    psam = ParallelSAManager(p)
    psam.run(minutes=6.0)

if __name__ == '__main__':
    main()
