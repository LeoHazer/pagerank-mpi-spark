import networkx as nx
import numpy as np

from src.comum.grafo import (
    criar_grafo,
    gerar_grafo_aleatorio,
    grafo_exemplo_3_paginas,
    grafo_exemplo_4_paginas,
)
from src.comum.metricas import erro_l1
from src.sequencial.pagerank import pagerank


def pagerank_networkx(grafo, d):
    """PageRank calculado pelo networkx, usado como gabarito."""
    g = nx.DiGraph()
    # adiciona todas as páginas, inclusive as que não têm nenhum link
    g.add_nodes_from(range(grafo.n_vertices))
    g.add_edges_from(zip(grafo.origens.tolist(), grafo.destinos.tolist()))
    ranks = nx.pagerank(g, alpha=d, tol=1e-13, max_iter=1000)
    return np.array([ranks[v] for v in range(grafo.n_vertices)])


def test_exemplo_dos_slides_sem_amortecimento():
    rank, _ = pagerank(grafo_exemplo_3_paginas(), d=1.0, tolerancia=1e-12, max_iteracoes=1000)
    assert np.allclose(rank, [0.4, 0.4, 0.2], atol=1e-9)


def test_soma_dos_ranks_e_um():
    rank, _ = pagerank(gerar_grafo_aleatorio(2000, 6, semente=5))
    assert abs(rank.sum() - 1.0) < 1e-9


def test_igual_ao_networkx_com_pagina_sem_saida():
    grafo = grafo_exemplo_4_paginas()
    rank, _ = pagerank(grafo, tolerancia=1e-13)
    assert erro_l1(rank, pagerank_networkx(grafo, 0.85)) < 1e-9


def test_igual_ao_networkx_em_grafo_aleatorio():
    grafo = gerar_grafo_aleatorio(3000, 8, semente=11)
    rank, _ = pagerank(grafo, tolerancia=1e-13)
    assert erro_l1(rank, pagerank_networkx(grafo, 0.85)) < 1e-9


def test_grafo_sem_arestas_fica_uniforme():
    rank, _ = pagerank(criar_grafo(4, [], []))
    assert np.allclose(rank, 0.25)


def test_tolerancia_zero_roda_todas_as_iteracoes():
    _, iteracoes = pagerank(gerar_grafo_aleatorio(500, 4, semente=2), tolerancia=0, max_iteracoes=7)
    assert iteracoes == 7
