"""
Representação de grafos usada pelas três versões do PageRank
(sequencial, MPI e Spark).

Analogia: imagine uma cidade onde cada vértice é uma casa e cada aresta é
uma carta enviada de uma casa para outra. Em vez de guardar uma tabela
gigante "quem escreve para quem" (matriz N x N, quase toda vazia), guardamos
só a lista das cartas: duas colunas, ORIGEM e DESTINO. Isso se chama
"lista de arestas" e é o formato mais fácil de dividir entre processos.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grafo:
    """Grafo dirigido guardado como lista de arestas."""

    n_vertices: int          # quantidade de páginas (vértices 0 .. n_vertices-1)
    origens: np.ndarray      # origens[i]  = página que tem o link i
    destinos: np.ndarray     # destinos[i] = página para a qual o link i aponta

    @property
    def n_arestas(self) -> int:
        return int(self.origens.shape[0])

    def grau_saida(self) -> np.ndarray:
        """Quantos links saem de cada página (vetor de tamanho n_vertices)."""
        return np.bincount(self.origens, minlength=self.n_vertices)


def _limpar_arestas(n_vertices, origens, destinos):
    """
    Regras do PageRank: links de uma página para ela mesma são
    ignorados e links repetidos contam como um só.
    """
    origens = np.asarray(origens, dtype=np.int64)
    destinos = np.asarray(destinos, dtype=np.int64)

    # 1) remove auto-links (página apontando para si mesma)
    manter = origens != destinos
    origens, destinos = origens[manter], destinos[manter]

    # 2) remove links repetidos: transformamos cada par (origem, destino)
    #    em um número único e deixamos o NumPy descartar os repetidos.
    #    Ex.: com N = 1.000.000, o link 3 -> 7 vira 3.000.007; dois links
    #    iguais viram o mesmo número, e o np.unique deixa só um de cada.
    chaves = np.unique(origens * n_vertices + destinos)
    # desfaz a conta: a divisão inteira devolve a origem e o resto, o destino
    origens = (chaves // n_vertices).astype(np.int64)
    destinos = (chaves % n_vertices).astype(np.int64)
    return origens, destinos


def criar_grafo(n_vertices, origens, destinos) -> Grafo:
    """Cria um Grafo já limpo (sem auto-links e sem repetições)."""
    origens, destinos = _limpar_arestas(n_vertices, origens, destinos)
    return Grafo(n_vertices, origens, destinos)


def grafo_exemplo_3_paginas() -> Grafo:
    """
    Exemplo dos slides: 3 páginas. Sem amortecimento (d = 1) o PageRank
    converge para [0.4, 0.4, 0.2].

        0 -> 1, 0 -> 2, 1 -> 0, 2 -> 1
    """
    return criar_grafo(3, [0, 0, 1, 2], [1, 2, 0, 1])


def grafo_exemplo_4_paginas() -> Grafo:
    """
    Exemplo de quatro páginas (páginas A, B, C, D = 0, 1, 2, 3).
    A página A não tem links de saída: é um nó "sem saída" (dangling),
    ótimo para testar esse caso especial.

        B -> C, B -> A, C -> A, D -> A, D -> B, D -> C
    """
    return criar_grafo(4, [1, 1, 2, 3, 3, 3], [2, 0, 0, 0, 1, 2])


def gerar_grafo_aleatorio(n_vertices, grau_medio, semente=42) -> Grafo:
    """
    Gera um grafo sintético parecido com a web: poucas páginas "famosas"
    recebem muitos links e a maioria recebe poucos.

    - A ORIGEM de cada link é sorteada uniformemente.
    - O DESTINO é sorteado com viés para os números baixos (u ** 3), o que
      cria páginas muito populares (ids pequenos).
    - A mesma 'semente' sempre gera o mesmo grafo (reprodutível).
    """
    # sorteador de números; com a mesma semente, o sorteio sai sempre igual
    gerador = np.random.default_rng(semente)
    # total de links sorteados (ex.: 1 milhão de páginas x 10 = 10 milhões)
    n_sorteios = int(n_vertices * grau_medio)

    # ORIGEM: uma página de 0 a N-1 para cada link, todas com a mesma chance
    origens = gerador.integers(0, n_vertices, size=n_sorteios)

    # DESTINO: sorteia um número entre 0 e 1 e eleva ao cubo. Elevar ao cubo
    # puxa os valores para perto de zero (0,5 vira 0,125; 0,1 vira 0,001),
    # então números de página baixos saem muito mais: são as páginas "famosas".
    # Multiplicar por N e cortar a parte decimal transforma em número de página.
    destinos = (n_vertices * gerador.random(n_sorteios) ** 3).astype(np.int64)
    # segurança: se o sorteio der exatamente 1,0, a página N não existiria
    destinos = np.minimum(destinos, n_vertices - 1)

    # tira links de uma página para ela mesma e links repetidos
    return criar_grafo(n_vertices, origens, destinos)


def salvar_grafo(grafo: Grafo, caminho: str) -> None:
    """Salva o grafo em um arquivo .npz (formato binário do NumPy)."""
    np.savez(
        caminho,
        n_vertices=np.int64(grafo.n_vertices),
        origens=grafo.origens,
        destinos=grafo.destinos,
    )


def carregar_grafo(caminho: str) -> Grafo:
    """Lê um grafo salvo por salvar_grafo()."""
    with np.load(caminho) as arquivo:
        return Grafo(
            n_vertices=int(arquivo["n_vertices"]),
            origens=arquivo["origens"],
            destinos=arquivo["destinos"],
        )