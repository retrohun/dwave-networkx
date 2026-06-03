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

__all__ = ['maximal_matching',
           'min_maximal_matching',
           ]


def maximal_matching(G, sampler, **sampler_args):
    """Finds an approximate maximal matching.

    Defines a BQM with ground states corresponding to a maximal
    matching and uses the sampler to sample from it.

    A matching is a subset of edges in which no node occurs more than
    once. A maximal matching is one in which no edges from G can be
    added without violating the matching rule.

    Finding maximal matchings can be done is polynomial time, so this method
    is only useful pedagogically.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to find a maximal matching.

    sampler : :class:`dimod.Sampler`
        A dimod sampler.

    sampler_args
        Additional keyword parameters are passed to the sampler.

    Returns
    -------
    matching : set
        A maximal matching of the graph.

    Notes
    -----
    Samplers by their nature may not return the optimal solution. This
    function does not attempt to confirm the quality of the returned
    sample.

    References
    ----------

    `Matching on Wikipedia <https://w.wiki/r9s>`_

    `QUBO on Wikipedia <https://w.wiki/r9t>`_

    Based on the formulation presented in [Luc2014]_.

    """
    if not G.edges:
        return set()

    bqm = dimod.generators.maximal_matching(G)
    sampleset = sampler.sample(bqm, **sampler_args)
    sample = sampleset.first.sample

    # the matching are the edges that are 1 in the sample
    return set(tuple(edge) for edge, val in sample.items() if val > 0)


def min_maximal_matching(G, sampler, **sampler_args):
    """Returns an approximate minimum maximal matching.

    Defines a BQM with ground states corresponding to a minimum
    maximal matching and uses the sampler to sample from it.

    A matching is a subset of edges in which no node occurs more than
    once. A maximal matching is one in which no edges from G can be
    added without violating the matching rule. A minimum maximal
    matching is the smallest maximal matching for G.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to find a minimum maximal matching.

    sampler : :class:`dimod.Sampler`
        A dimod sampler.

    sampler_args
        Additional keyword parameters are passed to the sampler.

    Returns
    -------
    matching : set
        A minimum maximal matching of the graph.

    Example
    -------
    This example uses a sampler from
    `dimod <https://github.com/dwavesystems/dimod>`_ to find a minimum maximal
    matching for a Chimera unit cell.

    >>> import dimod
    >>> sampler = dimod.ExactSolver()
    >>> G = dwave.graphs.chimera_graph(1, 1, 4)
    >>> matching = dwave.graphs.min_maximal_matching(G, sampler)

    Notes
    -----
    Samplers by their nature may not return the optimal solution. This
    function does not attempt to confirm the quality of the returned
    sample.

    References
    ----------

    `Matching on Wikipedia <https://w.wiki/r9s>`_

    `QUBO on Wikipedia <https://w.wiki/r9t>`_

    Lucas, A. (2014). Ising formulations of many NP problems.
    Frontiers in Physics, Volume 2, Article 5.

    """
    if not G.edges:
        return set()

    bqm = dimod.generators.min_maximal_matching(G)
    sampleset = sampler.sample(bqm, **sampler_args)
    sample = sampleset.first.sample

    # the matching are the edges that are 1 in the sample
    return set(tuple(edge) for edge, val in sample.items() if val > 0)
