import sys
import traceback
import random
import os
import json
import time
from abc import abstractmethod, abstractproperty, ABCMeta
from multiprocessing import cpu_count, Pool
from collections import namedtuple
from importlib import import_module

import simanneal


class Annealer(simanneal.Annealer):

    @classmethod
    def load_state(cls, s):
        return json.loads(s)

    @classmethod
    def dump_state(cls, state):
        return json.dumps(state)


class ProblemSet(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def divide(self, divisions):
        raise NotImplementedError

    @abstractproperty
    def problem_data_str(self):
        raise NotImplementedError

    @abstractproperty
    def pcp(self):
        raise NotImplementedError


ProblemClassPath = namedtuple('ProblemClassPath', ['module', 'cls'])

Solution = namedtuple('Solution', ['state', 'energy'])


def runner((id, pcp, serialized_state, minutes, problem_data,
            serialized_schedule)):
    random.seed(os.urandom(16))
    try:
        print("Running subproblem with the following parameters: {}".format(
            (id, pcp, minutes)
        ))
        pccls_module = import_module(pcp.module)
        PCCls = getattr(pccls_module, pcp.cls)
        state = PCCls.load_state(serialized_state)
        annealer = PCCls(state, problem_data)
        if serialized_schedule is None:
            schedule = annealer.auto(
                minutes=minutes,
            )
        else:
            schedule = json.loads(serialized_schedule)

        annealer.set_schedule(schedule)
        best_state, best_energy = annealer.anneal()

        return Solution(PCCls.dump_state(best_state), best_energy)
    except ZeroDivisionError:
        print("Run {} failed!".format((id, pcp, minutes)))
        print("".join(traceback.format_exception(*sys.exc_info())))
        return Solution(state, annealer.energy(state))
    except:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


def group_runner((id, pcp, sstates, minutes, problem_data, sschedule)):
    minutes_per_task = float(minutes)/len(sstates)
    return [runner(("{}_{}".format(id, i), pcp, s, minutes_per_task,
                    problem_data, sschedule))
            for i, s in enumerate(sstates)]


class ParallelSAManager(object):
    """ParallelSAManager

    :type problem_set: ProblemSet
    """
    def __init__(self, problem_set):
        self.problem_set = problem_set

    def run(self, minutes, cpus=cpu_count()):
        start = time.time()
        process_pool = Pool(cpus)

        subproblem_groups = list(self.problem_set.divide(cpus))
        available_cpu_time = float(minutes*cpus)
        time_per_group = available_cpu_time/len(subproblem_groups)

        pcp = self.problem_set.pcp
        problem_data = self.problem_set.problem_data_str

        solution_groups = process_pool.map(
            group_runner,
            [
                (i, pcp, sstates, time_per_group,
                 problem_data, None) for i, sstates in
                enumerate(subproblem_groups)
            ]
        )

        winner = sorted(
            (solution for solution_group in solution_groups
            for solution in solution_group),
            key=lambda s: s.energy
        )[0]
        print("With an energy of {}; {} was the best.".format(
            winner.energy,
            winner.state
        ))
        print("Run took {}s".format(time.time() - start))

        return winner
