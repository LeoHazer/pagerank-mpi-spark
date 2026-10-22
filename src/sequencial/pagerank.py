"""
PageRank sequencial (um único processo).

É a versão de referência: as versões MPI e Spark precisam chegar no mesmo
vetor de ranks que esta.

Analogia: cada página tem uma "nota" e, a cada rodada, divide essa nota
igualmente entre as páginas para as quais aponta. A nova nota de uma página
é a soma do que ela recebeu. Repetimos as rodadas até as notas pararem de
mudar.

Uso (a partir da raiz do projeto):
    python -m src.sequencial.pagerank --grafo dados/grafo_100k.npz
"""

import argparse
import time

import numpy as np

from src.comum.grafo import carregar_grafo
from src.comum.relatorio import linhas_do_top, porcentagem
from src.comum.resultados import registrar

AMORTECIMENTO = 0.85
TOLERANCIA = 1e-10
MAX_ITERACOES = 200


def somar_contribuicoes(origens, destinos, rank, grau_saida, n_vertices):
    """
    Cada link origem -> destino leva rank[origem] / grau_saida[origem] para
    o destino. Devolve um vetor com o total que cada página recebeu.
    """
    # parte da nota que viaja por cada link
    por_link = rank[origens] / grau_saida[origens]
    # bincount com pesos soma todos os valores que chegam no mesmo destino;
    # sem nenhuma aresta ele devolve inteiros, então forçamos float64
    # (no MPI todos os processos precisam mandar o mesmo tipo de dado)
    recebido = np.bincount(destinos, weights=por_link, minlength=n_vertices)
    return recebido.astype(np.float64, copy=False)


def aplicar_formula(recebido, massa_sem_saida, n_vertices, d):
    """
    Fórmula do PageRank com amortecimento d:
      - (1 - d) / N: chance de o navegador largar os links e abrir uma
        página qualquer;
      - massa_sem_saida / N: a nota das páginas sem links de saída é
        repartida igualmente entre todas (senão ela sumiria do sistema).
    """
    return (1.0 - d) / n_vertices + d * (recebido + massa_sem_saida / n_vertices)


def pagerank(grafo, d=AMORTECIMENTO, tolerancia=TOLERANCIA, max_iteracoes=MAX_ITERACOES):
    """
    Calcula o PageRank por iteração de potência.
    Devolve (vetor de ranks, número de iterações feitas).
    Com tolerancia=0 roda sempre max_iteracoes rodadas (útil para medir tempo).
    """
    n = grafo.n_vertices
    grau = grafo.grau_saida()
    sem_saida = grau == 0
    # começa com todas as páginas empatadas
    rank = np.full(n, 1.0 / n)

    iteracao = 0
    for iteracao in range(1, max_iteracoes + 1):
        recebido = somar_contribuicoes(grafo.origens, grafo.destinos, rank, grau, n)
        massa = rank[sem_saida].sum()
        novo = aplicar_formula(recebido, massa, n, d)

        # quanto o vetor mudou nesta rodada (distância L1)
        variacao = np.abs(novo - rank).sum()
        rank = novo
        if variacao < tolerancia:
            break

    return rank, iteracao


def main() -> None:
    parser = argparse.ArgumentParser(description="PageRank sequencial")
    parser.add_argument("--grafo", type=str, required=True, help="arquivo .npz do gerar_grafo")
    parser.add_argument("--amortecimento", type=float, default=AMORTECIMENTO)
    parser.add_argument("--tolerancia", type=float, default=TOLERANCIA,
                        help="critério de parada; 0 roda todas as iterações")
    parser.add_argument("--max-iteracoes", type=int, default=MAX_ITERACOES)
    parser.add_argument("--top", type=int, default=5, help="quantas páginas mostrar no ranking")
    parser.add_argument("--csv", type=str, default=None, help="planilha onde anotar os tempos")
    parser.add_argument("--variante", type=str, default="base", help="rótulo da alteração testada")
    args = parser.parse_args()
    inicio_programa = time.perf_counter()

    inicio = time.perf_counter()
    grafo = carregar_grafo(args.grafo)
    tempo_leitura = time.perf_counter() - inicio

    inicio = time.perf_counter()
    rank, iteracoes = pagerank(grafo, args.amortecimento, args.tolerancia, args.max_iteracoes)
    tempo_calculo = time.perf_counter() - inicio
    tempo_total = time.perf_counter() - inicio_programa

    if args.csv:
        registrar(args.csv, versao="sequencial", variante=args.variante, processos=1,
                  vertices=grafo.n_vertices, arestas=grafo.n_arestas, iteracoes=iteracoes,
                  tempo_leitura=tempo_leitura, tempo_distribuicao=0.0,
                  tempo_calculo=tempo_calculo, tempo_total=tempo_total)

    print(f"Grafo: {grafo.n_vertices} vértices, {grafo.n_arestas} arestas")
    print(f"  iterações: {iteracoes}")
    # as chances de todas as páginas juntas precisam dar 100%
    print(f"  soma das chances: {porcentagem(rank.sum())}")
    print(f"  tempo de leitura: {tempo_leitura:.3f} s")
    print(f"  tempo de cálculo: {tempo_calculo:.3f} s")
    print("\n".join(linhas_do_top(rank, args.top)))


if __name__ == "__main__":
    main()
