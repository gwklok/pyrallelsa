from __future__ import division

from abc import abstractmethod, abstractproperty
import time
import sys
import traceback
from multiprocessing import cpu_count, Pool
from collections import namedtuple
from importlib import import_module

import simanneal


class State(object):

    @abstractmethod
    def copy(self):
        raise NotImplementedError

    @abstractmethod
    def serialize(self):
        raise NotImplementedError

    @classmethod
    def load(cls, s):
        raise NotImplementedError


class Annealer(simanneal.Annealer):
    pass


class ProblemSet(object):

    @abstractmethod
    def divide(self):
        raise NotImplementedError

    @abstractproperty
    def problem_data_str(self):
        raise NotImplementedError

    @abstractproperty
    def pcp(self):
        raise NotImplementedError

    @abstractproperty
    def psp(self):
        raise NotImplementedError


ProblemClassPath = namedtuple('ProblemClassPath', ['module', 'cls'])
ProblemStatePath = namedtuple('ProblemStatePath', ['module', 'cls'])
Solution = namedtuple('Solution', ['state', 'energy'])

def runner((id, pcp, psp, serialized_state, minutes, problem_data)):
    try:
        print("Running subproblem with the following parameters: {}".format(
            (id, pcp, psp, minutes)
        ))
        pscls_module = import_module(psp.module)
        PSCls = getattr(pscls_module, psp.cls)
        state = PSCls.load(serialized_state)

        pccls_module = import_module(pcp.module)
        PCCls = getattr(pccls_module, pcp.cls)
        annealer = PCCls(state, problem_data)
        auto_schedule = annealer.auto(minutes=minutes)
        annealer.set_schedule(auto_schedule)
        best_state, best_fitness = annealer.anneal()

        return Solution(best_state.serialize(), best_fitness)
    except:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


class ParallelSAManager(object):
    """ParallelSAManager

    :type problem_set: ProblemSet
    """
    def __init__(self, problem_set):
        self.problem_set = problem_set

    def run(self, minutes, cpus=cpu_count()):
        start = time.time()
        process_pool = Pool(cpus)

        subproblems = list(self.problem_set.divide())
        available_cpu_time = minutes*cpus
        time_per_task = available_cpu_time/len(subproblems)

        pcp = self.problem_set.pcp
        psp = self.problem_set.psp
        problem_data = self.problem_set.problem_data_str

        # i, state = 0, subproblems[0]
        # args_list = (i, pcp, psp, state.serialize(), time_per_task,
        #              problem_data)
        # print(args_list)
        # runner(args_list)

        solutions = process_pool.map(
            runner,
            [
                (i, pcp, psp, state.serialize(), time_per_task,
                 problem_data) for i, state in
                enumerate(subproblems)
            ]
        )

        winner = sorted(solutions, key=lambda s: s.energy)[0]

        print("With an energy of {}; {} was the best.".format(
            winner.energy,
            winner.state
        ))
        print("Run took {}s".format(time.time() - start))

        return winner
