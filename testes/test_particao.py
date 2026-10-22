import numpy as np

from src.comum.grafo import gerar_grafo_aleatorio
from src.comum.particao import dividir_por_destino


def test_faixas_cobrem_todas_as_paginas_e_links():
    grafo = gerar_grafo_aleatorio(5000, 8, semente=3)
    limites, links = dividir_por_destino(grafo.destinos, grafo.n_vertices, 4)
    assert limites[0] == 0 and limites[-1] == grafo.n_vertices
    assert np.all(np.diff(limites) >= 0)
    assert links.sum() == grafo.n_arestas


def test_faixas_tem_quantidade_parecida_de_links():
    grafo = gerar_grafo_aleatorio(20000, 8, semente=3)
    _, links = dividir_por_destino(grafo.destinos, grafo.n_vertices, 4)
    media = grafo.n_arestas / 4
    # nenhuma faixa passa de 10% acima da média (com faixas de páginas iguais, a primeira teria ~60%)
    assert links.max() < 1.1 * media


def test_pagina_muito_popular_nao_e_cortada_ao_meio():
    # a página 0 recebe 6 dos 8 links: ela fica inteira numa parte só
    destinos = np.array([0, 0, 0, 0, 0, 0, 1, 2])
    limites, links = dividir_por_destino(destinos, 3, 4)
    assert links.sum() == 8
    assert 6 in links.tolist()


def test_sem_links_nao_quebra():
    limites, links = dividir_por_destino(np.array([], dtype=np.int64), 5, 3)
    assert limites.tolist()[-1] == 5 and links.sum() == 0
