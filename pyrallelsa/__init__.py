import json
import os
import random
import sys
import time
import traceback
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from importlib import import_module
from multiprocessing import cpu_count, Pool

import simanneal


class Problem(simanneal.Annealer):

    __metaclass__ = ABCMeta

    @classmethod
    def load_state(cls, s):
        return json.loads(s)

    @classmethod
    def dump_state(cls, state):
        return json.dumps(state)

    @classmethod
    def divide(cls, divisions, problem_data):
        """Divide the Problem defined by the problem_data into the
        given number of divisions

        Divide should be deterministic in the sense that it returns the
        same divisions in the same order on multiple calls. This is done
        because the scheduler does not store division values, but enumeration
        of the divisions as indices.

        :param int divisions: If divisions given is 0, use a default
            number of divisions determined programatically
        :type problem_data: dict
        """
        raise NotImplementedError

    @abstractmethod
    def move(self):
        raise NotImplementedError

    @abstractmethod
    def energy(self):
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
        annealer = PCCls(state, **json.loads(problem_data))
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
    def __init__(self, pcp, problem_data):
        self.problem_data = problem_data
        self.pcp = pcp

    def run(self, minutes, cpus=cpu_count()):
        start = time.time()
        process_pool = Pool(cpus)

        pccls_module = import_module(self.pcp.module)
        PCCls = getattr(pccls_module, self.pcp.cls)

        subproblem_groups = list(PCCls.divide(cpus, self.problem_data))
        available_cpu_time = float(minutes*cpus)
        time_per_group = available_cpu_time/len(subproblem_groups)

        solution_groups = process_pool.map(
            group_runner,
            [
                (i, self.pcp, sstates, time_per_group,
                 json.dumps(self.problem_data), None) for i, sstates in
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
