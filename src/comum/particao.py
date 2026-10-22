"""
Divisão das páginas em faixas, usada pela alteração 2 do MPI.

Cada processo fica dono de uma faixa seguida de páginas e cuida dos links
que chegam nelas. As faixas NÃO têm o mesmo número de páginas: no nosso
grafo os links se concentram nas páginas de número baixo, então faixas
iguais deixariam o primeiro processo com quase todo o trabalho. Em vez
disso, cortamos as faixas para que cada uma receba mais ou menos a mesma
quantidade de links (isso se chama balanceamento de carga).
"""

import numpy as np


def dividir_por_destino(destinos, n_vertices, partes):
    """
    Devolve (limites, links_por_parte).
    A parte k fica com as páginas limites[k] até limites[k + 1] - 1.
    """
    n_arestas = len(destinos)
    # chegam[v] = quantos links chegam na página v
    chegam = np.bincount(destinos, minlength=n_vertices)
    # antes[v] = quantos links chegam nas páginas 0 .. v-1 (antes[0] = 0)
    antes = np.concatenate(([0], np.cumsum(chegam)))

    # a parte k começa na primeira página em que o acumulado chega a k/partes do total
    alvos = np.arange(1, partes) * n_arestas / partes
    # searchsorted acha, num vetor em ordem crescente, a posição de cada alvo
    cortes = np.searchsorted(antes, alvos, side="left")

    limites = np.concatenate(([0], cortes, [n_vertices]))
    # garante que os limites nunca andam para trás (partes vazias são permitidas)
    limites = np.maximum.accumulate(np.clip(limites, 0, n_vertices)).astype(np.int64)
    links_por_parte = np.diff(antes[limites]).astype(np.int64)
    return limites, links_por_parte
