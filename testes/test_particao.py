import numpy as np

from src.comum.grafo import gerar_grafo_aleatorio
from src.comum.particao import dividir_por_destino


# dividir_por_destino corta as páginas em faixas seguidas, uma por processo (usada na
# alteração 2 do MPI). Garante que as faixas começam na página 0, terminam na última, nunca
# andam para trás e que, somadas, têm todos os links: nenhum some nem aparece duas vezes.
def test_faixas_cobrem_todas_as_paginas_e_links():
    grafo = gerar_grafo_aleatorio(5000, 8, semente=3)
    limites, links = dividir_por_destino(grafo.destinos, grafo.n_vertices, 4)
    assert limites[0] == 0 and limites[-1] == grafo.n_vertices
    assert np.all(np.diff(limites) >= 0)
    assert links.sum() == grafo.n_arestas


# Garante o balanceamento de carga: cada faixa recebe mais ou menos a mesma quantidade de
# links, que é o trabalho de cada processo. Se uma faixa tivesse muito mais links, os
# outros processos ficariam parados esperando por ela.
def test_faixas_tem_quantidade_parecida_de_links():
    grafo = gerar_grafo_aleatorio(20000, 8, semente=3)
    _, links = dividir_por_destino(grafo.destinos, grafo.n_vertices, 4)
    media = grafo.n_arestas / 4
    # nenhuma faixa passa de 10% acima da média
    # (com faixas de páginas iguais, a primeira teria ~60% dos links)
    assert links.max() < 1.1 * media


# Caso extremo: uma página recebe quase todos os links. Os cortes só podem cair entre uma
# página e outra, nunca no meio de uma, porque cada página tem um único processo dono.
def test_pagina_muito_popular_nao_e_cortada_ao_meio():
    # a página 0 recebe 6 dos 8 links: ela fica inteira numa parte só
    destinos = np.array([0, 0, 0, 0, 0, 0, 1, 2])
    limites, links = dividir_por_destino(destinos, 3, 4)
    assert links.sum() == 8
    # alguma parte ficou com os 6 links da página 0 juntos
    assert 6 in links.tolist()


# Caso limite: grafo sem nenhum link. A divisão precisa funcionar mesmo assim, com todas
# as faixas sem links e a última terminando no total de páginas (5).
def test_sem_links_nao_quebra():
    limites, links = dividir_por_destino(np.array([], dtype=np.int64), 5, 3)
    assert limites.tolist()[-1] == 5 and links.sum() == 0
