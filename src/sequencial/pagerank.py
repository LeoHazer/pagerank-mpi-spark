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

# d da fórmula: em 85% das vezes o navegador segue um link da página atual;
# nos outros 15% ele "se cansa" e abre uma página qualquer
AMORTECIMENTO = 0.85
# para quando a soma das mudanças entre duas rodadas fica abaixo de 1e-10
# (0,0000000001), ou seja, as notas praticamente pararam de mudar
TOLERANCIA = 1e-10
# limite de segurança, caso a tolerância nunca seja atingida
MAX_ITERACOES = 200


def somar_contribuicoes(origens, destinos, rank, grau_saida, n_vertices):
    """
    Cada link origem -> destino leva rank[origem] / grau_saida[origem] para
    o destino. Devolve um vetor com o total que cada página recebeu.
    """
    # parte da nota que viaja por cada link. rank[origens] usa o vetor origens
    # como lista de posições: pega a nota da origem de CADA link de uma vez
    # (ex.: origens [2, 0, 2] -> [rank[2], rank[0], rank[2]]), sem laço for.
    # Não há divisão por zero: toda página que aparece em origens tem grau >= 1.
    # Ex.: página com nota 0,3 e 3 links de saída manda 0,1 por cada link
    por_link = rank[origens] / grau_saida[origens]
    # bincount com pesos soma todos os valores que chegam no mesmo destino;
    # ex.: destinos [1, 1, 0] e por_link [0,1; 0,2; 0,4] -> recebido [0,4; 0,3].
    # minlength garante uma posição para cada página, mesmo sem links chegando.
    # sem nenhuma aresta ele devolve inteiros, então forçamos float64
    # (número real de 64 bits; no MPI todos os processos precisam mandar o mesmo
    # tipo de dado). copy=False evita copiar o vetor quando ele já é float64
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
    # PR(v) = (1 - d)/N + d * (recebido[v] + massa_sem_saida/N), para todas as
    # páginas ao mesmo tempo: recebido é um vetor e o resto são números, então o
    # NumPy aplica a mesma conta em cada posição do vetor
    return (1.0 - d) / n_vertices + d * (recebido + massa_sem_saida / n_vertices)


def pagerank(grafo, d=AMORTECIMENTO, tolerancia=TOLERANCIA, max_iteracoes=MAX_ITERACOES):
    """
    Calcula o PageRank por iteração de potência.
    Devolve (vetor de ranks, número de iterações feitas).
    Com tolerancia=0 roda sempre max_iteracoes rodadas (útil para medir tempo).
    """
    n = grafo.n_vertices
    # o grau de saída e a lista de páginas sem saída não mudam entre rodadas,
    # então são calculados uma vez só, fora do laço
    grau = grafo.grau_saida()
    # vetor de True/False: True nas páginas que não têm nenhum link de saída
    sem_saida = grau == 0
    # começa com todas as páginas empatadas (ex.: 4 páginas -> 0,25 cada)
    rank = np.full(n, 1.0 / n)

    # valor inicial só para a variável existir mesmo se max_iteracoes for 0
    iteracao = 0
    # cada volta do laço é uma rodada: todas as páginas repassam suas notas
    for iteracao in range(1, max_iteracoes + 1):
        recebido = somar_contribuicoes(grafo.origens, grafo.destinos, rank, grau, n)
        # rank[sem_saida] pega só as notas das páginas sem saída (onde há True);
        # a soma delas é redistribuída igualmente dentro da fórmula
        massa = rank[sem_saida].sum()
        novo = aplicar_formula(recebido, massa, n, d)

        # quanto o vetor mudou nesta rodada (distância L1: soma das diferenças
        # absolutas, a mesma conta de erro_l1 em metricas.py)
        variacao = np.abs(novo - rank).sum()
        rank = novo
        # critério de parada: as notas quase não mudaram, então convergiu
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
    # perf_counter é um cronômetro de alta precisão; medimos cada etapa separada
    # para comparar com as versões MPI e Spark etapa por etapa
    inicio_programa = time.perf_counter()

    inicio = time.perf_counter()
    grafo = carregar_grafo(args.grafo)
    tempo_leitura = time.perf_counter() - inicio

    # só as rodadas do PageRank: é este tempo que entra no cálculo do speedup
    # (quantas vezes a versão paralela foi mais rápida que esta)
    inicio = time.perf_counter()
    rank, iteracoes = pagerank(grafo, args.amortecimento, args.tolerancia, args.max_iteracoes)
    tempo_calculo = time.perf_counter() - inicio
    tempo_total = time.perf_counter() - inicio_programa

    # anota a execução na planilha, se o --csv foi passado. Aqui há um processo
    # só e não há nada para repartir, por isso processos=1 e tempo_distribuicao=0
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
