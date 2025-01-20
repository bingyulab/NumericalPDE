import numpy as np
import networkx as nx
from src.tools.sparse_matrix import SparseMatrix
from src.solver.base_solver import Solver  # Changed import to base_solver
import logging


class GraphSolver(Solver):

    def solve(self, L, b):
        logging.info("GraphSolver: Solving system")
        N = int(np.sqrt(L.shape[0]))
        G = nx.grid_2d_graph(N, N)
        pos = dict((n, n) for n in G.nodes())
        labels = dict(((i, j), i * N + j) for i, j in G.nodes())
        nx.set_node_attributes(G, pos, 'pos')
        nx.set_node_attributes(G, labels, 'label')

        L_dense = L.to_dense().toarray() if isinstance(L, SparseMatrix) else L

        for i in range(N):
            for j in range(N):
                if i > 0:
                    # Use absolute value to ensure non-negative weights
                    weight = abs(L_dense[i * N + j, (i - 1) * N + j])
                    G.add_edge((i, j), (i - 1, j), weight=weight)
                if i < N - 1:
                    weight = abs(L_dense[i * N + j, (i + 1) * N + j])
                    G.add_edge((i, j), (i + 1, j), weight=weight)
                if j > 0:
                    weight = abs(L_dense[i * N + j, i * N + j - 1])
                    G.add_edge((i, j), (i, j - 1), weight=weight)
                if j < N - 1:
                    weight = abs(L_dense[i * N + j, i * N + j + 1])
                    G.add_edge((i, j), (i, j + 1), weight=weight)

        b_dict = {
            labels[(i, j)]: b[i * N + j]
            for i in range(N)
            for j in range(N)
        }
        source = (0, 0)  # Use tuple label
        if source not in G:
            raise nx.NodeNotFound(f"Node {source} not found in graph")
        u_dict = nx.single_source_dijkstra_path_length(G,
                                                       source=source,
                                                       weight='weight')
        u = np.array([
            u_dict.get(labels[(i, j)], 0) for i in range(N) for j in range(N)
        ])
        return u
