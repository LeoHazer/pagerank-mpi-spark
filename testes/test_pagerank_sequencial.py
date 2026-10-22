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


# networkx é uma biblioteca de grafos muito usada e confiável, feita por outras pessoas.
# O PageRank dela serve de gabarito: se a nossa versão sequencial der o mesmo resultado,
# ela está certa. Depois, MPI e Spark são comparados com a sequencial.
def pagerank_networkx(grafo, d):
    """PageRank calculado pelo networkx, usado como gabarito."""
    g = nx.DiGraph()
    # adiciona todas as páginas, inclusive as que não têm nenhum link
    g.add_nodes_from(range(grafo.n_vertices))
    g.add_edges_from(zip(grafo.origens.tolist(), grafo.destinos.tolist()))
    # alpha é o nome que o networkx dá ao amortecimento d; tol bem pequena = conta bem precisa
    ranks = nx.pagerank(g, alpha=d, tol=1e-13, max_iter=1000)
    return np.array([ranks[v] for v in range(grafo.n_vertices)])


# Exemplo de 3 páginas resolvido nos slides da disciplina: sem amortecimento (d = 1) a
# resposta conhecida é [0,4; 0,4; 0,2]. É o teste mais simples de mostrar: a conta já
# foi feita à mão.
def test_exemplo_dos_slides_sem_amortecimento():
    rank, _ = pagerank(grafo_exemplo_3_paginas(), d=1.0, tolerancia=1e-12, max_iteracoes=1000)
    # np.allclose compara posição por posição, aceitando diferença de até atol (1e-9)
    assert np.allclose(rank, [0.4, 0.4, 0.2], atol=1e-9)


# Os ranks são chances (probabilidades) de o navegador aleatório estar em cada página, então
# a soma precisa dar 1 (100%). Se desse menos, alguma nota teria "vazado", por exemplo
# nas páginas sem links de saída.
def test_soma_dos_ranks_e_um():
    rank, _ = pagerank(gerar_grafo_aleatorio(2000, 6, semente=5))
    assert abs(rank.sum() - 1.0) < 1e-9


# Compara com o gabarito do networkx no exemplo de 4 páginas, em que a página A (0) não
# tem links de saída. Esse caso especial é onde é mais fácil errar a fórmula.
def test_igual_ao_networkx_com_pagina_sem_saida():
    grafo = grafo_exemplo_4_paginas()
    rank, _ = pagerank(grafo, tolerancia=1e-13)
    # erro_l1 = soma das diferenças absolutas, posição por posição; abaixo de 1e-9
    # (0,000000001) quer dizer que os dois vetores são praticamente iguais
    assert erro_l1(rank, pagerank_networkx(grafo, 0.85)) < 1e-9


# O mesmo teste num grafo maior e sorteado (3000 páginas), parecido com os das medições.
# tolerancia=1e-13 faz o cálculo continuar até o vetor quase não mudar mais, para que a
# comparação com o gabarito seja justa.
def test_igual_ao_networkx_em_grafo_aleatorio():
    grafo = gerar_grafo_aleatorio(3000, 8, semente=11)
    rank, _ = pagerank(grafo, tolerancia=1e-13)
    assert erro_l1(rank, pagerank_networkx(grafo, 0.85)) < 1e-9


# Caso extremo: grafo sem nenhum link. Todas as páginas ficam "sem saída" e nenhuma se
# destaca, então cada uma das 4 termina com 1/4 = 0,25, sem dar erro.
def test_grafo_sem_arestas_fica_uniforme():
    rank, _ = pagerank(criar_grafo(4, [], []))
    assert np.allclose(rank, 0.25)


# Com tolerancia=0 o critério de parada (mudança menor que a tolerância) nunca é atingido,
# então o cálculo faz exatamente max_iteracoes rodadas. As medições de tempo usam isso
# para que todo experimento faça a mesma quantidade de trabalho.
def test_tolerancia_zero_roda_todas_as_iteracoes():
    _, iteracoes = pagerank(gerar_grafo_aleatorio(500, 4, semente=2), tolerancia=0, max_iteracoes=7)
    assert iteracoes == 7
