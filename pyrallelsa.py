from __future__ import division

import json
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

def runner((id, pcp, psp, serialized_state, minutes, problem_data,
            serialized_schedule)):
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
        schedule = json.loads(serialized_schedule)
        annealer.set_schedule(schedule)
        best_state, best_energy = annealer.anneal()

        return Solution(best_state.serialize(), best_energy)
    except ZeroDivisionError:
        print("Run {} failed!".format((id, pcp, psp, minutes)))
        print("".join(traceback.format_exception(*sys.exc_info())))
        return Solution(state, annealer.energy(state))
    except:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


def get_auto_schedule((id, pcp, psp, serialized_state, minutes, problem_data)):
    print("Canary run for finding schedule with the following parameters...:"
      " {}".format((id, pcp, psp, minutes)))
    pscls_module = import_module(psp.module)
    PSCls = getattr(pscls_module, psp.cls)
    state = PSCls.load(serialized_state)

    pccls_module = import_module(pcp.module)
    PCCls = getattr(pccls_module, pcp.cls)
    annealer = PCCls(state, problem_data)
    auto_schedule = annealer.auto(minutes=minutes)
    print("Found schedule: {}".format(auto_schedule))
    return json.dumps(auto_schedule)


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

        schedule = process_pool.map(
            get_auto_schedule,
            [
                (i, pcp, psp, state.serialize(), time_per_task,
                 problem_data) for i, state in
                enumerate(subproblems[:1])
            ]
        )[0]
        solutions = process_pool.map(
            runner,
            [
                (i, pcp, psp, state.serialize(), time_per_task,
                 problem_data, schedule) for i, state in
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
