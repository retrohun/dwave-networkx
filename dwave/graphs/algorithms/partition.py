# Copyright 2021 D-Wave Systems Inc.
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

import math

import dimod

__all__ = ["partition",
           ]


def partition(G, sampler, num_partitions=2, **sampler_args):
    """Returns an approximate k-partition of G.
    
    Defines an CQM with ground states corresponding to a
    balanced k-partition of G and uses the sampler to sample from it.
    A k-partition is a collection of k subsets of the vertices
    of G such that each vertex is in exactly one subset, and
    the number of edges between vertices in different subsets
    is as small as possible. If G is a weighted graph, the sum
    of weights over those edges are minimized.
    
    Parameters
    ----------
    G : NetworkX graph
        The graph to partition.
    sampler : :class:`dimod.Sampler`
        A dimod sampler.
    num_partitions : int, optional (default 2)
        The number of subsets in the desired partition.
    sampler_args
        Additional keyword parameters are passed to the sampler.
    
    Returns
    -------
    node_partition : dict
        The partition as a dictionary mapping each node to subsets labelled
        as integers 0, 1, 2, ... num_partitions.
    
    Example
    -------
    This example uses a sampler from
    `dimod <https://github.com/dwavesystems/dimod>`_ to find a 2-partition
    for a graph of a Chimera unit cell created using the `chimera_graph()`
    function.
    
    >>> import dimod
    >>> sampler = dimod.ExactCQMSolver()
    >>> G = dwave.graphs.chimera_graph(1, 1, 4)
    >>> partitions = dwave.graphs.partition(G, sampler=sampler)
    
    Notes
    -----
    Samplers by their nature may not return the optimal solution. This
    function does not attempt to confirm the quality of the returned
    sample.
    """
    if not len(G.nodes):
        return {}
        
    cqm = dimod.generators.graph_partition(G, num_partitions)
    
    # Solve the problem using the CQM solver
    response = sampler.sample_cqm(cqm, **sampler_args)

    # Consider only results satisfying all constraints
    possible_partitions = response.filter(lambda d: d.is_feasible)
    
    if not possible_partitions: 
        raise RuntimeError("No feasible solution could be found for this problem instance.")

    # Reinterpret result as partition assignment over nodes
    indicators = (key for key, value in possible_partitions.first.sample.items() if math.isclose(value, 1.))
    node_partition = {key[0]: key[1] for key in indicators}
    
    return node_partition
