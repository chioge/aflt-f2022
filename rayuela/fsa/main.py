from fractions import Fraction
from rayuela.base.semiring import Boolean, Real, Tropical, \
    String, Integer, Rational
from rayuela.base.symbol import Sym, ε
from rayuela.fsa.fsa import FSA
from rayuela.fsa.state import State

from rayuela.fsa.pathsum import Pathsum

from rayuela.fsa.scc import SCC

fsa = FSA(Real)

# We can directly add edges between the states without adding the states first.
# The states will be created automatically.
fsa.add_arc(State(1), Sym('a'), State(2),0.5)
fsa.add_arc(State(1), Sym('b'), State(3),0.2)

fsa.add_arc(State(2), Sym('c'), State(4), 1)

fsa.add_arc(State(3), Sym('c'), State(4),0.4)
fsa.add_arc(State(3), Sym('b'), State(5),0.2)
fsa.add_arc(State(5), Sym('a'), State(6),0.2)

fsa.add_arc(State(4), Sym('a'), State(6),0.3)

fsa.add_arc(State(4), Sym('a'), State(7),0.6)
# Add initial and final states
fsa.set_I(State(1))
fsa.set_F(State(6))

fsa_r = fsa.reverse()

fsa_t = fsa.closure()

pathsum = Pathsum(fsa)

t = pathsum.viterbi_fwd()
s = pathsum.viterbi_bwd()

scc = SCC(fsa)

scc._kosaraju()


fsa = FSA(Real)



fsa.add_arc(State(0), Sym('a'), State(3),0.5)
fsa.add_arc(State(3), Sym('a'), State(2),0.5)
fsa.add_arc(State(1), Sym('a'), State(0),0.5)
fsa.add_arc(State(2), Sym('a'), State(1),0.5)
fsa.add_arc(State(4), Sym('a'), State(2),0.5)
fsa.add_arc(State(5), Sym('a'), State(4),0.5)
fsa.add_arc(State(4), Sym('a'), State(6),0.5)
fsa.add_arc(State(6), Sym('a'), State(5),0.5)
fsa.add_arc(State(6), Sym('a'), State(7),0.5)
fsa.add_arc(State(8), Sym('a'), State(7),0.5)

fsa.set_I(State(0))
fsa.set_F(State(0))
fsa_r = fsa.reverse()
scc = SCC(fsa_r)

print(scc._kosaraju())