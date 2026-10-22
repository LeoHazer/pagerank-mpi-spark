"""
PageRank paralelo com MPI (mpi4py).

Ideia: as arestas do grafo são divididas entre os processos. Em cada
iteração, cada processo calcula o que as SUAS arestas levam para cada
página e depois todos somam esses vetores parciais com um Allreduce. No
fim do Allreduce todo processo tem o vetor completo e consegue aplicar a
fórmula sozinho, sem mais comunicação.

Analogia: uma apuração de eleição. Cada mesário (processo) conta os votos
da sua urna (suas arestas) e depois todos somam as parciais para chegar
no resultado final, que fica conhecido por todos.

Coletivas usadas:
  - bcast:     o rank 0 avisa quantos vértices e arestas o grafo tem
  - Scatterv:  o rank 0 reparte as listas de origens e destinos
  - Bcast:     o rank 0 envia o vetor de graus de saída (tamanho N)
  - Allreduce: a cada iteração, soma dos vetores parciais (tamanho N)
  - reduce:    pega o tempo do processo mais lento para a medição

Uso (a partir da raiz do projeto):
    mpirun -n 4 python -m mpi4py -m src.mpi.pagerank --grafo dados/grafo_100k.npz

O "-m mpi4py" faz com que, se um processo der erro, todos sejam
encerrados; sem ele, os outros ficariam esperando para sempre.
"""

import argparse

import numpy as np
from mpi4py import MPI

from src.comum.grafo import carregar_grafo
from src.comum.relatorio import linhas_do_top, porcentagem
from src.comum.resultados import registrar
from src.sequencial.pagerank import (
    AMORTECIMENTO,
    MAX_ITERACOES,
    TOLERANCIA,
    aplicar_formula,
    somar_contribuicoes,
)


def dividir(total, partes):
    """Quantos itens cada processo recebe e em que posição começa a fatia de cada um."""
    base, resto = divmod(total, partes)
    # os primeiros 'resto' processos ficam com um item a mais
    contagens = np.array([base + (1 if i < resto else 0) for i in range(partes)], dtype=np.int64)
    deslocamentos = np.concatenate(([0], np.cumsum(contagens)[:-1])).astype(np.int64)
    return contagens, deslocamentos


def distribuir_grafo(comm, grafo):
    """
    Só o rank 0 tem o grafo inteiro (os outros recebem None).
    Devolve, em todos os processos: (n_vertices, origens locais,
    destinos locais, grau de saída de todas as páginas).
    """
    eu = comm.Get_rank()
    tamanhos = (grafo.n_vertices, grafo.n_arestas) if eu == 0 else None
    # bcast com letra minúscula envia um objeto Python qualquer (aqui, uma tupla pequena)
    n_vertices, n_arestas = comm.bcast(tamanhos, root=0)

    contagens, deslocamentos = dividir(n_arestas, comm.Get_size())
    origens = np.empty(contagens[eu], dtype=np.int64)
    destinos = np.empty(contagens[eu], dtype=np.int64)

    if eu == 0:
        # formato [buffer, quanto vai para cada um, onde começa cada fatia, tipo do dado]
        envio_origens = [np.ascontiguousarray(grafo.origens, dtype=np.int64),
                         contagens, deslocamentos, MPI.INT64_T]
        envio_destinos = [np.ascontiguousarray(grafo.destinos, dtype=np.int64),
                          contagens, deslocamentos, MPI.INT64_T]
    else:
        envio_origens = envio_destinos = None

    # Scatterv com letra maiúscula manda o buffer do NumPy direto, sem serializar;
    # o "v" permite que cada processo receba uma quantidade diferente
    comm.Scatterv(envio_origens, origens, root=0)
    comm.Scatterv(envio_destinos, destinos, root=0)

    # todo processo precisa do grau de TODAS as páginas, não só das suas arestas
    if eu == 0:
        grau = grafo.grau_saida().astype(np.int64)
    else:
        grau = np.empty(n_vertices, dtype=np.int64)
    comm.Bcast(grau, root=0)

    return n_vertices, origens, destinos, grau


def pagerank_mpi(comm, n_vertices, origens, destinos, grau,
                 d=AMORTECIMENTO, tolerancia=TOLERANCIA, max_iteracoes=MAX_ITERACOES):
    """Mesmo algoritmo da versão sequencial, mas cada processo só olha as suas arestas."""
    sem_saida = grau == 0
    rank = np.full(n_vertices, 1.0 / n_vertices)
    recebido = np.empty(n_vertices)

    iteracao = 0
    for iteracao in range(1, max_iteracoes + 1):
        parcial = somar_contribuicoes(origens, destinos, rank, grau, n_vertices)
        # soma os vetores parciais de todos os processos; todos recebem o total
        comm.Allreduce(parcial, recebido, op=MPI.SUM)

        massa = rank[sem_saida].sum()
        novo = aplicar_formula(recebido, massa, n_vertices, d)

        # todos os processos têm o mesmo vetor, então todos decidem parar juntos
        variacao = np.abs(novo - rank).sum()
        rank = novo
        if variacao < tolerancia:
            break

    return rank, iteracao


def main() -> None:
    parser = argparse.ArgumentParser(description="PageRank com MPI")
    parser.add_argument("--grafo", type=str, required=True, help="arquivo .npz do gerar_grafo")
    parser.add_argument("--amortecimento", type=float, default=AMORTECIMENTO)
    parser.add_argument("--tolerancia", type=float, default=TOLERANCIA,
                        help="critério de parada; 0 roda todas as iterações")
    parser.add_argument("--max-iteracoes", type=int, default=MAX_ITERACOES)
    parser.add_argument("--top", type=int, default=5, help="quantas páginas mostrar no ranking")
    parser.add_argument("--salvar-ranks", type=str, default=None,
                        help="arquivo .npy para guardar o vetor final (usado nos testes)")
    parser.add_argument("--csv", type=str, default=None, help="planilha onde anotar os tempos")
    parser.add_argument("--variante", type=str, default="base", help="rótulo da alteração testada")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    eu = comm.Get_rank()
    inicio_programa = MPI.Wtime()

    # só o rank 0 lê o arquivo; os outros recebem a sua parte pela rede
    inicio = MPI.Wtime()
    grafo = carregar_grafo(args.grafo) if eu == 0 else None
    tempo_leitura = MPI.Wtime() - inicio

    # Barrier: todo mundo espera aqui para o cronômetro começar junto
    comm.Barrier()
    inicio = MPI.Wtime()
    n_vertices, origens, destinos, grau = distribuir_grafo(comm, grafo)
    tempo_distribuicao = MPI.Wtime() - inicio
    # o rank 0 não precisa mais da cópia inteira do grafo
    del grafo

    comm.Barrier()
    inicio = MPI.Wtime()
    rank, iteracoes = pagerank_mpi(comm, n_vertices, origens, destinos, grau,
                                   args.amortecimento, args.tolerancia, args.max_iteracoes)
    tempo_calculo = MPI.Wtime() - inicio
    tempo_total = MPI.Wtime() - inicio_programa

    # o tempo que vale é o do processo mais lento
    tempo_distribuicao = comm.reduce(tempo_distribuicao, op=MPI.MAX, root=0)
    tempo_calculo = comm.reduce(tempo_calculo, op=MPI.MAX, root=0)
    tempo_total = comm.reduce(tempo_total, op=MPI.MAX, root=0)
    # confere que nenhuma aresta se perdeu na divisão
    total_arestas = comm.reduce(len(origens), op=MPI.SUM, root=0)

    if eu != 0:
        return

    print(f"Grafo: {n_vertices} vértices, {total_arestas} arestas")
    print(f"  processos: {comm.Get_size()}")
    print(f"  iterações: {iteracoes}")
    print(f"  soma das chances: {porcentagem(rank.sum())}")
    print(f"  tempo de leitura: {tempo_leitura:.3f} s")
    print(f"  tempo de distribuição: {tempo_distribuicao:.3f} s")
    print(f"  tempo de cálculo: {tempo_calculo:.3f} s")
    print("\n".join(linhas_do_top(rank, args.top)))

    if args.salvar_ranks:
        np.save(args.salvar_ranks, rank)
    if args.csv:
        registrar(args.csv, versao="mpi", variante=args.variante, processos=comm.Get_size(),
                  vertices=n_vertices, arestas=total_arestas, iteracoes=iteracoes,
                  tempo_leitura=tempo_leitura, tempo_distribuicao=tempo_distribuicao,
                  tempo_calculo=tempo_calculo, tempo_total=tempo_total)


if __name__ == "__main__":
    main()
