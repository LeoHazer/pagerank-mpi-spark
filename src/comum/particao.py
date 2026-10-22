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
    # Exemplo usado nos comentários abaixo: 4 páginas, 10 links e 3 partes,
    # com 5 links chegando na página 0, 3 na página 1 e 1 em cada uma das outras.
    n_arestas = len(destinos)
    # chegam[v] = quantos links chegam na página v.
    # bincount conta quantas vezes cada número aparece no vetor: destinos
    # [0, 1, 0, 3] viram [2, 1, 0, 1]. O minlength garante uma posição para
    # cada página, mesmo as que não recebem link. No exemplo: [5, 3, 1, 1]
    chegam = np.bincount(destinos, minlength=n_vertices)
    # antes[v] = quantos links chegam nas páginas 0 .. v-1 (antes[0] = 0).
    # cumsum é a soma acumulada: [5, 3, 1, 1] vira [5, 8, 9, 10]. Colocando
    # um 0 na frente fica antes = [0, 5, 8, 9, 10]; o último é o total de links
    antes = np.concatenate(([0], np.cumsum(chegam)))

    # a parte k começa na primeira página em que o acumulado chega a k/partes do total.
    # arange(1, partes) gera [1, 2, ..., partes-1]; no exemplo, alvos = [3,33; 6,67]
    # (um terço e dois terços dos 10 links)
    alvos = np.arange(1, partes) * n_arestas / partes
    # searchsorted acha, num vetor em ordem crescente, a posição de cada alvo:
    # com side="left", a primeira posição i em que antes[i] >= alvo.
    # No exemplo: 3,33 -> posição 1 (antes[1] = 5) e 6,67 -> posição 2 (antes[2] = 8)
    cortes = np.searchsorted(antes, alvos, side="left")

    # junta o começo (página 0) e o fim (n_vertices) aos cortes: [0, 1, 2, 4]
    limites = np.concatenate(([0], cortes, [n_vertices]))
    # garante que os limites nunca andam para trás (partes vazias são permitidas).
    # clip prende cada valor entre 0 e n_vertices; maximum.accumulate troca cada
    # valor pelo maior visto até ali, ex.: [0, 3, 2, 5] vira [0, 3, 3, 5]
    limites = np.maximum.accumulate(np.clip(limites, 0, n_vertices)).astype(np.int64)
    # antes[limites] dá o acumulado em cada limite ([0, 5, 8, 10]) e diff faz a
    # diferença entre vizinhos ([5, 3, 2]): são os links de cada parte.
    # Repare que não fica perfeito: a página 0 sozinha tem 5 links e uma página
    # não pode ser dividida entre dois processos
    links_por_parte = np.diff(antes[limites]).astype(np.int64)
    return limites, links_por_parte
