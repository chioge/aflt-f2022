from __future__ import annotations
import copy
import numpy as np

from frozendict import frozendict
from itertools import product

from collections import Counter
from collections import defaultdict as dd

from rayuela.base.semiring import Boolean, String, ProductSemiring
from rayuela.base.misc import epsilon_filter
from rayuela.base.symbol import Sym, ε, ε_1, ε_2

from rayuela.fsa.state import State, PairState
from rayuela.fsa.pathsum import Pathsum, Strategy

class FSA:

	def __init__(self, R=Boolean):

		# DEFINITION
		# A weighted finite-state automaton is a 5-tuple <R, Σ, Q, δ, λ, ρ> where
		# • R is a semiring;
		# • Σ is an alphabet of symbols;
		# • Q is a finite set of states;
		# • δ is a finite relation Q × Σ × Q × R;
		# • λ is an initial weight function;
		# • ρ is a final weight function.

		# NOTATION CONVENTIONS
		# • single states (elements of Q) are denoted q
		# • multiple states not in sequence are denoted, p, q, r, ...
		# • multiple states in sequence are denoted i, j, k, ...
		# • symbols (elements of Σ) are denoted lowercase a, b, c, ...
		# • single weights (elements of R) are denoted w
		# • multiple weights (elements of R) are denoted u, v, w, ...

		# semiring
		self.R = R

		# alphabet of symbols
		self.Sigma = set([])

		# a finite set of states
		self.Q = set([])

		# transition function : Q × Σ × Q → R
		self.δ = dd(lambda : dd(lambda : dd(lambda : self.R.zero)))

		# initial weight function
		self.λ = R.chart()

		# final weight function
		self.ρ = R.chart()

	def add_state(self, q):
		self.Q.add(q)

	def add_states(self, Q):
		for q in Q:
			self.add_state(q)

	def add_arc(self, i, a, j, w=None):
		if w is None: w = self.R.one

		if not isinstance(i, State): i = State(i)
		if not isinstance(j, State): j = State(j)
		if not isinstance(a, Sym): a = Sym(a)
		if not isinstance(w, self.R): w = self.R(w)

		self.add_states([i, j])
		self.Sigma.add(a)
		self.δ[i][a][j] += w

	def set_arc(self, i, a, j, w):
		self.add_states([i, j])
		self.Sigma.add(a)
		self.δ[i][a][j] = w

	def set_I(self, q, w=None):
		if w is None: w = self.R.one
		self.add_state(q)
		self.λ[q] = w

	def set_F(self, q, w=None):
		if w is None: w = self.R.one
		self.add_state(q)
		self.ρ[q] = w

	def add_I(self, q, w):
		self.add_state(q)
		self.λ[q] += w

	def add_F(self, q, w):
		self.add_state(q)
		self.ρ[q] += w

	def freeze(self):
		self.Sigma = frozenset(self.Sigma)
		self.Q = frozenset(self.Q)
		self.δ = frozendict(self.δ)
		self.λ = frozendict(self.λ)
		self.ρ = frozendict(self.ρ)

	@property
	def I(self):
		for q, w in self.λ.items():
			if w != self.R.zero:
				yield q, w

	@property
	def F(self):
		for q, w in self.ρ.items():
			if w != self.R.zero:
				yield q, w

	def arcs(self, i, no_eps=False):
		for a, T in self.δ[i].items():
			if no_eps and a == ε:
				continue
			for j, w in T.items():
				if w == self.R.zero:
					continue
				yield a, j, w

	def accept(self, string):
		""" determines whether a string is in the language """
		assert isinstance(string, str)

		fsa = FSA(R=self.R)
		for i, x in enumerate(list(string)):
			fsa.add_arc(State(i), Sym(x), State(i+1), self.R.one)
		
		fsa.set_I(State(0), self.R.one)
		fsa.add_F(State(len(string)), self.R.one)

		return self.intersect(fsa).pathsum()

	@property
	def num_states(self):
		return len(self.Q)

	def copy(self):
		""" deep copies the machine """
		return copy.deepcopy(self)

	def spawn(self, keep_init=False, keep_final=False):
		""" returns a new FSA in the same semiring """
		F = FSA(R=self.R)

		if keep_init:
			for q, w in self.I:
				F.set_I(q, w)
		if keep_final:
			for q, w in self.F:
				F.set_F(q, w)

		return F

	def dfs_2(self):
		""" Depth-first search (Cormen et al. 2019; Section 22.3) """

		in_progress, finished = set([]), {}
		cyclic, counter = False, 0

		def _dfs(p):
			nonlocal in_progress
			nonlocal finished
			nonlocal cyclic
			nonlocal counter

			in_progress.add(p)

			for _, q, _ in self.arcs(p):
				if q in in_progress:
					cyclic = True
				elif q not in finished:
					_dfs(q)

			in_progress.remove(p)
			finished[p] = counter
			counter += 1

		for q, _ in self.I: _dfs(q)
		
		for q in self.Q: 
			if (q not in finished):
				_dfs(q)

		return cyclic, finished

	def dfs(self):
		""" Depth-first search (Cormen et al. 2019; Section 22.3) """

		in_progress, finished = set([]), {}
		cyclic, counter = False, 0

		def _dfs(p):
			nonlocal in_progress
			nonlocal finished
			nonlocal cyclic
			nonlocal counter

			in_progress.add(p)

			for _, q, _ in self.arcs(p):
				if q in in_progress:
					cyclic = True
				elif q not in finished:
					_dfs(q)

			in_progress.remove(p)
			finished[p] = counter
			counter += 1

		for q, _ in self.I: _dfs(q)

		return cyclic, finished

	def finish(self, rev=False, acyclic_check=False):
		"""
		Returns the nodes in order of their finishing time.
		"""

		cyclic, finished = self.dfs()

		if acyclic_check:
			assert self.acyclic

		sort = {}
		for s, n in finished.items():
			sort[n] = s
		if rev:
			for n in sorted(list(sort.keys())):
				yield sort[n]
		else:
			for n in reversed(sorted(list(sort.keys()))):
				yield sort[n]

	def toposort(self, rev=False):
		return self.finish(rev=rev, acyclic_check=True)

	@property
	def acyclic(self) -> bool:
		cyclic, _ = self.dfs()
		return not cyclic

	@property
	def deterministic(self) -> bool:
		for i in self.Q:
			symbols = set([])
			for a,j,w in self.arcs(i):
				if (a in symbols or a == ε) and w != self.R.zero:
					return False
				symbols.add(a)
		return True

	@property
	def pushed(self) -> bool:
		sum = self.R.zero
		for i in self.Q:
			sum += self.ρ[i]
			for a,j,w in self.arcs(i):
				sum += w
		
		return sum == self.R.one

	def reverse(self) -> FSA:
		""" computes the reverse of the FSA """
		fsa_r = FSA(self.R)

		for i in self.Q:
			for a,j,w in self.arcs(i):
				fsa_r.add_arc(j, a, i, w)

		for q,w in self.F:
			fsa_r.set_I(q)

		for q,w in self.I:
			fsa_r.set_F(q)
		
		return fsa_r

	def accessible(self) -> set:
		""" computes the set of acessible states """
		accessible = set([])

		
		cylic, finished = self.dfs()
		for s in finished:
			accessible.add(s)

		return accessible

	def coaccessible(self) -> set:
		""" computes the set of acessible states """
		coaccessible = set([])
		fsa_r = self.reverse()
		
		cylic, finished = fsa_r.dfs()
		for s in finished:
			coaccessible.add(s)

		return coaccessible
		

	def trim(self) -> FSA:
		""" keeps only those states that are both accessible and co-accessible """
		fsa_t = FSA(self.R)
		accessible = self.accessible()
		coaccessible = self.coaccessible()
		intersection = accessible.intersection(coaccessible)

		for i in self.Q:
			if i in intersection:
				for a,j,w in self.arcs(i):
					if j in intersection:
						fsa_t.add_arc(i, a, j, w)

		for q,w in self.F:
			if q in intersection:
				fsa_t.set_F(q)

		for q,w in self.I:
			if q in intersection:
				fsa_t.set_I(q)

		return fsa_t

	def union(self, fsa) -> FSA:
		fsa_u = self.copy()

		for i in fsa.Q:
			for a,j,w in fsa.arcs(i):
				fsa_u.add_arc(State(f'_{i.idx}'), a, State(f'_{j.idx}'), w)

		for q,w in fsa.F:
			fsa_u.set_F(State(f'_{q.idx}'))

		for q,w in self.I:
			fsa_u.set_I(State(f'_{q.idx}'))

		return fsa_u

	def concatenate(self, fsa) -> FSA:
		""" construct the concatenation of the two FSAs """
		fsa_c = self.union(fsa)

		for q1,w1 in self.F:
			for q2,w2 in fsa.I:
				fsa_c.add_arc(q1, ε, State(f'_{q2.idx}'), w1)


		return fsa_c

	def closure(self) -> FSA:
		fsa_k = self.copy()

		fsa_k.set_I(State('_i'))
		fsa_k.set_F(State('_i'))

		for q_i,w_i in fsa_k.I:
			fsa_k.add_arc(State('_i'), ε, q_i, w_i)
			for q_f,w_f in fsa_k.F:
				fsa_k.add_arc(q_f, ε, q_i, w_f)

		return fsa_k

	def pathsum(self, strategy=Strategy.LEHMANN):
		if self.acyclic:
			strategy = Strategy.VITERBI
		pathsum = Pathsum(self)
		return pathsum.pathsum(strategy)

	def edge_marginals(self) -> dict:
		# chart
		m = self.R.chart()

		pathsum = Pathsum(self)

		α = pathsum.viterbi_fwd()
		𝜷 = pathsum.viterbi_bwd()

		for i in self.Q:
			for a,j,w in self.arcs(i):
				m[(i,a,j)] = α[i] * 𝜷[j] * w


		return frozendict(m)

	def intersect(self, fsa):
		"""
		on-the-fly weighted intersection
		"""

		# the two machines need to be in the same semiring
		assert self.R == fsa.R

		# add initial states
		product_fsa = FSA(R=self.R)
		for (q1, w1), (q2, w2) in product(self.I, fsa.I):
			product_fsa.add_I(PairState(q1, q2), w=w1 * w2)
		
		self_initials = {q: w for q, w in self.I}
		fsa_initials = {q: w for q, w in fsa.I}

		visited = set([(i1, i2, State('0')) for i1, i2 in product(self_initials, fsa_initials)])
		stack = [(i1, i2, State('0')) for i1, i2 in product(self_initials, fsa_initials)]

		self_finals = {q: w for q, w in self.F}
		fsa_finals = {q: w for q, w in fsa.F}

		while stack:
			q1, q2, qf = stack.pop()

			E1 = [(a if a != ε else ε_2, j, w) for (a, j, w) in self.arcs(q1)] + \
                            [(ε_1, q1, self.R.one)]
			E2 = [(a if a != ε else ε_1, j, w) for (a, j, w) in fsa.arcs(q2)] + \
                            [(ε_2, q2, self.R.one)]

			M = [((a1, j1, w1), (a2, j2, w2))
				 for (a1, j1, w1), (a2, j2, w2) in product(E1, E2)
				 if epsilon_filter(a1, a2, qf) != State('⊥')]

			for (a1, j1, w1), (a2, j2, w2) in M:

				product_fsa.set_arc(
					PairState(q1, q2), a1,
					PairState(j1, j2), w=w1*w2)

				_qf = epsilon_filter(a1, a2, qf)
				if (j1, j2, _qf) not in visited:
					stack.append((j1, j2, _qf))
					visited.add((j1, j2, _qf))

			# final state handling
			if q1 in self_finals and q2 in fsa_finals:
				product_fsa.add_F(
					PairState(q1, q2), w=self_finals[q1] * fsa_finals[q2])

		return product_fsa

	def tikz(self, max_per_row=4):

		tikz_string = []
		previous_ids, positioning = [], ''
		rows = {}

		initial = {q: w for q, w in self.I}
		final = {q: w for q, w in self.F}

		for jj, q in enumerate(self.Q):
			options = 'state'
			additional = ''

			if q in initial:
				options += ', initial'
				additional = f' / {initial[q]}'
			if q in final:
				options += ', accepting'
				additional = f' / {final[q]}'

			if jj >= max_per_row:
				positioning = f'below = of {previous_ids[jj - max_per_row]}'
			elif len(previous_ids) > 0:
				positioning = f'right = of {previous_ids[-1]}'
			previous_ids.append(f'q{q.idx}')
			rows[q] = jj // max_per_row

			tikz_string.append(f'\\node[{options}] (q{q.idx}) [{positioning}] {{ ${q.idx}{additional}$ }}; \n')

		tikz_string.append('\\draw')

		seen_pairs, drawn_pairs = set(), set()

		for jj, q in enumerate(self.Q):
			target_edge_labels = dict()
			for a, j, w in self.arcs(q):
				if j not in target_edge_labels:
					target_edge_labels[j] = f'{a}/{w}'
				else:
					target_edge_labels[j] += f'\\\\{a}/{w}'
				seen_pairs.add(frozenset([q, j]))

			for ii, (target, label) in enumerate(target_edge_labels.items()):

				edge_options = 'align=left'
				if q == target:
					edge_options += ', loop above'
				elif frozenset([q, target]) not in seen_pairs:
					edge_options += 'a, bove'
				elif frozenset([q, target]) not in drawn_pairs:
					if rows[q] == rows[target]:
						edge_options += ', bend left, above'
					else:
						edge_options += ', bend left, right'
				else:
					if rows[q] == rows[target]:
						edge_options += ', bend left, below'
					else:
						edge_options += ', bend left, right'
				end = '\n'
				if jj == self.num_states - 1 and ii == len(target_edge_labels) - 1:
					end = '; \n'
				tikz_string.append(f'(q{q.idx}) edge[{edge_options}] node{{ ${label}$ }} (q{target.idx}) {end}')
				drawn_pairs.add(frozenset([q, j]))

		if not len(list(self.arcs(list(self.Q)[-1]))) > 0:
			tikz_string.append(';')

		return ''.join(tikz_string)

	def __truediv__(self, other):
		return self.intersect(other)

	def __add__(self, other):
		return self.concatenate(other)

	def __sub__(self, other):
		return self.difference(other)

	def __repr__(self):
		return f'WFSA({self.num_states} states, {self.R})'

	def __str__(self):
		""" ascii visualize """

		output = []
		for q, w in self.I:
			output.append(f"initial state:\t{q.idx}\t{w}")
		for q, w in self.F:
			output.append(f"final state:\t{q.idx}\t{w}")
		for p in self.Q:
			for a, q, w in self.arcs(p):
				output.append(f"{p}\t----{a}/{w}---->\t{q}")
		return "\n".join(output)

	def _repr_html_(self):
		"""
		When returned from a Jupyter cell, this will generate the FST visualization
		Based on: https://github.com/matthewfl/openfst-wrapper
		"""
		from uuid import uuid4
		import json
		from collections import defaultdict
		ret = []
		if self.num_states == 0:
			return '<code>Empty FST</code>'

		if self.num_states > 64:
			return f'FST too large to draw graphic, use fst.ascii_visualize()<br /><code>FST(num_states={self.num_states})</code>'

		finals = {q for q, _ in self.F}
		initials = {q for q, _ in self.I}

		# print initial
		for q, w in self.I:
			if q in finals:
				label = f'{str(q)} / [{str(w)} / {str(self.ρ[q])}]'
				color = 'af8dc3'
			else:
				label = f'{str(q)} / {str(w)}'
				color = '66c2a5'

			ret.append(
				f'g.setNode("{repr(q)}", {{ label: {json.dumps(label)} , shape: "circle" }});\n')
				# f'g.setNode("{repr(q)}", {{ label: {json.dumps(hash(label) // 1e8)} , shape: "circle" }});\n')

			ret.append(f'g.node("{repr(q)}").style = "fill: #{color}"; \n')

		# print normal
		for q in (self.Q - finals) - initials:

			label = str(q)

			ret.append(
				f'g.setNode("{repr(q)}", {{ label: {json.dumps(label)} , shape: "circle" }});\n')
				# f'g.setNode("{repr(q)}", {{ label: {json.dumps(hash(label) // 1e8)} , shape: "circle" }});\n')
			ret.append(f'g.node("{repr(q)}").style = "fill: #8da0cb"; \n')

		# print final
		for q, w in self.F:
			# already added
			if q in initials:
				continue

			if w == self.R.zero:
				continue
			label = f'{str(q)} / {str(w)}'

			ret.append(
				f'g.setNode("{repr(q)}", {{ label: {json.dumps(label)} , shape: "circle" }});\n')
				# f'g.setNode("{repr(q)}", {{ label: {json.dumps(hash(label) // 1e8)} , shape: "circle" }});\n')
			ret.append(f'g.node("{repr(q)}").style = "fill: #fc8d62"; \n')

		for q in self.Q:
			to = defaultdict(list)
			for a, j, w in self.arcs(q):
				if self.R is ProductSemiring and isinstance(w.score[0], String):
					# the imporant special case of encoding transducers
					label = f'{str(a)}:{str(w)}'
				else:
					label = f'{str(a)} / {str(w)}'
				to[j].append(label)

			for dest, values in to.items():
				if len(values) > 4:
					values = values[0:3] + ['. . .']
				label = '\n'.join(values)
				ret.append(
					f'g.setEdge("{repr(q)}", "{repr(dest)}", {{ arrowhead: "vee", label: {json.dumps(label)} }});\n')

		# if the machine is too big, do not attempt to make the web browser display it
		# otherwise it ends up crashing and stuff...
		if len(ret) > 256:
			return f'FST too large to draw graphic, use fst.ascii_visualize()<br /><code>FST(num_states={self.num_states})</code>'

		ret2 = ['''
		<script>
		try {
		require.config({
		paths: {
		"d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/4.13.0/d3",
		"dagreD3": "https://cdnjs.cloudflare.com/ajax/libs/dagre-d3/0.6.1/dagre-d3.min"
		}
		});
		} catch {
		  ["https://cdnjs.cloudflare.com/ajax/libs/d3/4.13.0/d3.js",
		   "https://cdnjs.cloudflare.com/ajax/libs/dagre-d3/0.6.1/dagre-d3.min.js"].forEach(function (src) {
			var tag = document.createElement('script');
			tag.src = src;
			document.body.appendChild(tag);
		  })
		}
		try {
		requirejs(['d3', 'dagreD3'], function() {});
		} catch (e) {}
		try {
		require(['d3', 'dagreD3'], function() {});
		} catch (e) {}
		</script>
		<style>
		.node rect,
		.node circle,
		.node ellipse {
		stroke: #333;
		fill: #fff;
		stroke-width: 1px;
		}

		.edgePath path {
		stroke: #333;
		fill: #333;
		stroke-width: 1.5px;
		}
		</style>
		''']

		obj = 'fst_' + uuid4().hex
		ret2.append(
			f'<center><svg width="850" height="600" id="{obj}"><g/></svg></center>')
		ret2.append('''
		<script>
		(function render_d3() {
		var d3, dagreD3;
		try { // requirejs is broken on external domains
		  d3 = require('d3');
		  dagreD3 = require('dagreD3');
		} catch (e) {
		  // for google colab
		  if(typeof window.d3 !== "undefined" && typeof window.dagreD3 !== "undefined") {
			d3 = window.d3;
			dagreD3 = window.dagreD3;
		  } else { // not loaded yet, so wait and try again
			setTimeout(render_d3, 50);
			return;
		  }
		}
		//alert("loaded");
		var g = new dagreD3.graphlib.Graph().setGraph({ 'rankdir': 'LR' });
		''')
		ret2.append(''.join(ret))

		ret2.append(f'var svg = d3.select("#{obj}"); \n')
		ret2.append(f'''
		var inner = svg.select("g");

		// Set up zoom support
		var zoom = d3.zoom().scaleExtent([0.3, 5]).on("zoom", function() {{
		inner.attr("transform", d3.event.transform);
		}});
		svg.call(zoom);

		// Create the renderer
		var render = new dagreD3.render();

		// Run the renderer. This is what draws the final graph.
		render(inner, g);

		// Center the graph
		var initialScale = 0.75;
		svg.call(zoom.transform, d3.zoomIdentity.translate(
		    (svg.attr("width") - g.graph().width * initialScale) / 2, 20).scale(initialScale));

		svg.attr('height', g.graph().height * initialScale + 50);
		}})();

		</script>
		''')

		return ''.join(ret2)
