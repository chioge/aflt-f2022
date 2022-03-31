import numpy as np
import operator
from numpy import linalg as LA

from collections import deque

from rayuela.base.semiring import Boolean, Real
from rayuela.fsa.pathsum import Pathsum, Strategy
from rayuela.fsa.fsa import FSA
from rayuela.fsa.state import State

class SCC:

    def __init__(self, fsa):
        self.fsa = fsa

    def scc(self):
        """
        Computes the SCCs of the FSA.
        Currently uses Kosaraju's algorithm.

        Guarantees SCCs come back in topological order.
        """
        for scc in self._kosaraju():
            yield scc

    def _kosaraju(self):
        """
        Kosaraju's algorithm [https://en.wikipedia.org/wiki/Kosaraju%27s_algorithm]
        Runs in O(E + V) time.
        Returns in the SCCs in topologically sorted order.
        """
		# Homework 3: Question 4
        cyclic, finished = self.fsa.dfs_2()
        fsa_r = self.fsa.reverse()

        scc_decomp = []

        visited = set([])
        fsa_r.λ = fsa_r.R.chart()

        finished = dict( sorted(finished.items(), key=operator.itemgetter(1),reverse=True))

        for q in finished:
            scc = set([])
            if q not in visited:
                fsa_r.set_I(q)
                cyclic_r, finished_r = fsa_r.dfs()
                for s in finished_r:
                    if(s not in visited):
                        scc.add(s)
                        visited.add(s)
                scc_decomp.append(scc)
                fsa_r.λ = fsa_r.R.chart()

        return scc_decomp

