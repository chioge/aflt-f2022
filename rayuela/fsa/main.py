from fractions import Fraction
from rayuela.base.semiring import Boolean, Real, Tropical, \
    String, Integer, Rational
from rayuela.base.symbol import Sym, ε
from rayuela.fsa.fsa import FSA
from rayuela.fsa.state import State


fsa = FSA(Boolean)

# We can directly add edges between the states without adding the states first.
# The states will be created automatically.
fsa.add_arc(State(1), Sym('a'), State(2))
fsa.add_arc(State(1), Sym('b'), State(3))

fsa.add_arc(State(2), Sym('b'), State(2))
fsa.add_arc(State(2), Sym('c'), State(4))

fsa.add_arc(State(3), Sym('c'), State(4))
fsa.add_arc(State(3), Sym('b'), State(5))

fsa.add_arc(State(4), Sym('a'), State(6))
fsa.add_arc(State(5), Sym('a'), State(6))

# Add initial and final states
fsa.set_I(State(1))
fsa.set_F(State(6))

fsa_r = fsa.reverse()

fsa_t = fsa.trim()