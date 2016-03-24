from __future__ import division

import sys
import traceback
import random
import math
import json
import time
from abc import abstractmethod, abstractproperty
from multiprocessing import cpu_count, Pool
from collections import namedtuple
from importlib import import_module

from simanneal.anneal import round_figures
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
    def auto(self, minutes, steps=2000, tmax_target_acceptance=0.98,
             tmin_target_improvement=0.0):
        """Minimizes the energy of a system by simulated annealing with
        automatic selection of the temperature schedule.

        Keyword arguments:
        state -- an initial arrangement of the system
        minutes -- time to spend annealing (after exploring temperatures)
        steps -- number of steps to spend on each stage of exploration

        Returns the best state and energy found."""

        def run(T, steps):
            """Anneals a system at constant temperature and returns the state,
            energy, rate of acceptance, and rate of improvement."""
            E = self.energy()
            prevState = self.copy_state(self.state)
            prevEnergy = E
            accepts, improves = 0, 0
            for step in range(steps):
                self.move()
                E = self.energy()
                dE = E - prevEnergy
                if dE > 0.0 and math.exp(-dE / T) < random.random():
                    self.state = self.copy_state(prevState)
                    E = prevEnergy
                else:
                    accepts += 1
                    if dE < 0.0:
                        improves += 1
                    prevState = self.copy_state(self.state)
                    prevEnergy = E
            return E, float(accepts) / steps, float(improves) / steps

        step = 0
        self.start = time.time()

        # Attempting automatic simulated anneal...
        # Find an initial guess for temperature
        T = 0.0
        E = self.energy()
        self.update(step, T, E, None, None)
        while T == 0.0:
            step += 1
            self.move()
            T = abs(self.energy() - E)

        # Search for Tmax - a temperature that gives 98% acceptance
        E, acceptance, improvement = run(T, steps)

        step += steps
        while acceptance > tmax_target_acceptance:
            T = round_figures(T / 1.5, 2)
            E, acceptance, improvement = run(T, steps)
            step += steps
            self.update(step, T, E, acceptance, improvement)
        while acceptance < tmax_target_acceptance:
            T = round_figures(T * 1.5, 2)
            E, acceptance, improvement = run(T, steps)
            step += steps
            self.update(step, T, E, acceptance, improvement)
        Tmax = T

        # Search for Tmin - a temperature that gives 0% improvement
        while improvement > tmin_target_improvement:
            T = round_figures(T / 1.5, 2)
            E, acceptance, improvement = run(T, steps)
            step += steps
            self.update(step, T, E, acceptance, improvement)
        Tmin = T

        # Calculate anneal duration
        elapsed = time.time() - self.start
        duration = round_figures(int(60.0 * minutes * step / elapsed), 2)

        # Don't perform anneal, just return params
        return {'tmax': Tmax, 'tmin': Tmin, 'steps': duration}


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
        if serialized_schedule is None:
            schedule = annealer.auto(
                minutes=minutes,
                tmax_target_acceptance=0.95,
                tmin_target_improvement=0.05
            )
        else:
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
    auto_schedule = annealer.auto(
        minutes=minutes,
        tmax_target_acceptance=0.98,
        tmin_target_improvement=0.01
    )
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

        # schedule = process_pool.map(
        #     get_auto_schedule,
        #     [
        #         (i, pcp, psp, state.serialize(), time_per_task,
        #          problem_data) for i, state in
        #         enumerate(subproblems[:1])
        #     ]
        # )[0]
        solutions = process_pool.map(
            runner,
            [
                (i, pcp, psp, state.serialize(), time_per_task,
                 problem_data, None) for i, state in
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
