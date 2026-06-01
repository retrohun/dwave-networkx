# Copyright 2018 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import dimod

__all__ = ["is_cycle",
           "is_vertex_coloring",
           "min_vertex_color",
           "min_vertex_coloring",   # alias for min_vertex_color
           "vertex_color",
           "vertex_coloring",       # alias for vertex_color
           ]


def vertex_color(G, colors, sampler, **sampler_args):
    """Returns an approximate vertex coloring.

    Vertex coloring is the problem of assigning a color to the
    vertices of a graph in a way that no adjacent vertices have the
    same color.

    Defines a QUBO [Dah2013]_ with ground states corresponding to valid
    vertex colorings and uses the sampler to sample from it.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to find a minimum vertex coloring.

    colors : int/sequence
        The colors. If an int, the colors are labelled `[0, n)`. The number of
        colors must be greater or equal to the chromatic number of the graph.

    sampler : :class:`dimod.Sampler`
        A dimod sampler.

    sampler_args
        Additional keyword parameters are passed to the sampler.

    Returns
    -------
    coloring : dict
        A coloring for each vertex in G such that no adjacent nodes
        share the same color. A dict of the form {node: color, ...}

    Notes
    -----
    Samplers by their nature may not return the optimal solution. This
    function does not attempt to confirm the quality of the returned
    sample.

    """
    bqm = dimod.generators.vertex_coloring(G, colors)

    # get the lowest energy sample
    sample = sampler.sample(bqm, **sampler_args).first.sample

    return {v: c for (v, c), val in sample.items() if val}


# alias
vertex_coloring = vertex_color


def min_vertex_color(G, sampler, chromatic_lb=None, chromatic_ub=None, **sampler_args):
    """Returns an approximate minimum vertex coloring.

    Vertex coloring is the problem of assigning a color to the
    vertices of a graph in a way that no adjacent vertices have the
    same color. A minimum vertex coloring is the problem of solving
    the vertex coloring problem using the smallest number of colors.

    Defines a QUBO [Dah2013]_ with ground states corresponding to minimum
    vertex colorings and uses the sampler to sample from it.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to find a minimum vertex coloring.

    sampler : :class:`dimod.Sampler`
        A dimod sampler.

    chromatic_lb : int, optional
         A lower bound on the chromatic number. If one is not provided, a
         bound is calculated.

    chromatic_ub : int, optional
        An upper bound on the chromatic number. If one is not provided, a bound
        is calculated.

    sampler_args
        Additional keyword parameters are passed to the sampler.

    Returns
    -------
    coloring : dict
        A coloring for each vertex in G such that no adjacent nodes
        share the same color. A dict of the form {node: color, ...}

    Notes
    -----
    Samplers by their nature may not return the optimal solution. This
    function does not attempt to confirm the quality of the returned
    sample.

    """

    bqm = dimod.generators.min_vertex_coloring(
        G, chromatic_lb=chromatic_lb, chromatic_ub=chromatic_ub)

    # get the lowest energy sample
    sample = sampler.sample(bqm, **sampler_args).first.sample

    return {v: c for (v, c), val in sample.items() if val}


# alias
min_vertex_coloring = min_vertex_color


def is_cycle(G):
    """Determines whether the given graph is a cycle or circle graph.

    A cycle graph or circular graph is a graph that consists of a single cycle.

    https://en.wikipedia.org/wiki/Cycle_graph

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    is_cycle : bool
        True if the graph consists of a single cycle.

    """
    if len(G) <= 2:
        return False

    trailing, leading = next(iter(G.edges))
    start_node = trailing

    # travel around the graph, checking that each node has degree exactly two
    # also track how many nodes were visited
    n_visited = 1
    while leading != start_node:
        neighbors = G[leading]

        if len(neighbors) != 2:
            return False

        node1, node2 = neighbors

        if node1 == trailing:
            trailing, leading = leading, node2
        else:
            trailing, leading = leading, node1

        n_visited += 1

    # if we haven't visited all of the nodes, then it is not a connected cycle
    return n_visited == len(G)


def is_vertex_coloring(G, coloring):
    """Determines whether the given coloring is a vertex coloring of graph G.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which the vertex coloring is applied.

    coloring : dict
        A coloring of the nodes of G. Should be a dict of the form
        {node: color, ...}.

    Returns
    -------
    is_vertex_coloring : bool
        True if the given coloring defines a vertex coloring; that is, no
        two adjacent vertices share a color.

    Example
    -------
    This example colors checks two colorings for a graph, G, of a single Chimera
    unit cell. The first uses one color (0) for the four horizontal qubits
    and another (1) for the four vertical qubits, in which case there are
    no adjacencies; the second coloring swaps the color of one node.

    >>> G = dwave.graphs.chimera_graph(1,1,4)
    >>> colors = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1}
    >>> dwave.graphs.is_vertex_coloring(G, colors)
    True
    >>> colors[4]=0
    >>> dwave.graphs.is_vertex_coloring(G, colors)
    False

   """
    return all(coloring[u] != coloring[v] for u, v in G.edges)
