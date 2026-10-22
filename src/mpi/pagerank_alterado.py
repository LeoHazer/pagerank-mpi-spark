"""
Versões alteradas do PageRank com MPI, para comparar com a original
(src/mpi/pagerank.py, que continua igual).

Escolha a alteração com --alteracao:
  1          Trabalho repetido em cada rodada: o peso de cada link
             (1 / grau da origem) é calculado uma vez só, antes das rodadas.
             A original refaz essa divisão em toda rodada.
  2          Divisão dos dados por página de destino: cada processo é dono
             de uma faixa de páginas e recebe só os links que chegam nelas.
             No fim da rodada cada um manda as notas das suas páginas para
             todos com Allgatherv (no lugar do Allreduce de um vetor inteiro).
  3          Formato do envio: o Allreduce (maiúsculo, envia a memória
             direto) vira allreduce (minúsculo), que "embrulha" o vetor
             como objeto Python (pickle) antes de enviar.
  otimizada  alterações 1 e 2 juntas.

Uso (a partir da raiz do projeto):
    mpirun -n 4 python -m mpi4py -m src.mpi.pagerank_alterado --alteracao 1 \
        --grafo dados/grafo_100k.npz
"""

import argparse

import numpy as np
from mpi4py import MPI

from src.comum.grafo import carregar_grafo
from src.comum.particao import dividir_por_destino
from src.comum.relatorio import linhas_do_top, porcentagem
from src.comum.resultados import registrar
from src.mpi.pagerank import distribuir_grafo
from src.sequencial.pagerank import (
    AMORTECIMENTO,
    MAX_ITERACOES,
    TOLERANCIA,
    aplicar_formula,
    somar_contribuicoes,
)

# nome de cada alteração na planilha de resultados
NOMES = {"1": "alteracao1", "2": "alteracao2", "3": "alteracao3", "otimizada": "otimizada"}


def pagerank_por_aresta(comm, n_vertices, origens, destinos, grau,
                        d=AMORTECIMENTO, tolerancia=TOLERANCIA, max_iteracoes=MAX_ITERACOES,
                        pesos_prontos=False, embrulhado=False):
    """Mesma divisão por arestas da versão original, com a alteração 1 ou a 3 ligada."""
    sem_saida = grau == 0
    rank = np.full(n_vertices, 1.0 / n_vertices)
    recebido = np.empty(n_vertices)

    if pesos_prontos:
        # alteração 1: a fração da nota que passa por cada link não muda entre as rodadas
        peso = 1.0 / grau[origens]

    iteracao = 0
    for iteracao in range(1, max_iteracoes + 1):
        if pesos_prontos:
            # multiplicar pelo peso pronto custa menos que buscar o grau e dividir
            parcial = np.bincount(destinos, weights=rank[origens] * peso, minlength=n_vertices)
            parcial = parcial.astype(np.float64, copy=False)
        else:
            parcial = somar_contribuicoes(origens, destinos, rank, grau, n_vertices)

        if embrulhado:
            # alteração 3: o vetor é serializado (pickle), enviado como bytes,
            # desembrulhado do outro lado e somado pelo próprio Python
            recebido = comm.allreduce(parcial, op=MPI.SUM)
        else:
            comm.Allreduce(parcial, recebido, op=MPI.SUM)

        massa = rank[sem_saida].sum()
        novo = aplicar_formula(recebido, massa, n_vertices, d)
        variacao = np.abs(novo - rank).sum()
        rank = novo
        if variacao < tolerancia:
            break

    return rank, iteracao


def distribuir_por_destino(comm, grafo):
    """
    Alteração 2: o rank 0 separa os links pela faixa de páginas de destino
    e entrega a cada processo os links que chegam nas páginas dele.
    Devolve (n_vertices, origens, destinos já contados a partir do início
    da faixa, grau de saída de todas as páginas, limites das faixas).
    """
    eu = comm.Get_rank()
    origens_ordenadas = destinos_ordenados = None
    info = None
    if eu == 0:
        limites, links = dividir_por_destino(grafo.destinos, grafo.n_vertices, comm.Get_size())
        # faixa[i] = número do processo dono da página de destino do link i
        faixa = np.searchsorted(limites, grafo.destinos, side="right") - 1
        # agrupa os links por faixa e, dentro de cada faixa, deixa em ordem de origem;
        # assim rank[origens] lê a memória em sequência, como na versão original
        # (lexsort ordena pela última chave e desempata pela anterior)
        ordem = np.lexsort((grafo.origens, faixa))
        origens_ordenadas = np.ascontiguousarray(grafo.origens[ordem], dtype=np.int64)
        destinos_ordenados = np.ascontiguousarray(grafo.destinos[ordem], dtype=np.int64)
        info = (grafo.n_vertices, limites, links)
    n_vertices, limites, links = comm.bcast(info, root=0)

    deslocamentos = np.concatenate(([0], np.cumsum(links)[:-1])).astype(np.int64)
    origens = np.empty(links[eu], dtype=np.int64)
    destinos = np.empty(links[eu], dtype=np.int64)
    if eu == 0:
        envio_origens = [origens_ordenadas, links, deslocamentos, MPI.INT64_T]
        envio_destinos = [destinos_ordenados, links, deslocamentos, MPI.INT64_T]
    else:
        envio_origens = envio_destinos = None
    comm.Scatterv(envio_origens, origens, root=0)
    comm.Scatterv(envio_destinos, destinos, root=0)

    grau = grafo.grau_saida().astype(np.int64) if eu == 0 else np.empty(n_vertices, dtype=np.int64)
    comm.Bcast(grau, root=0)

    # página limites[eu] vira posição 0 do vetor local, a seguinte vira 1, e assim por diante
    destinos -= limites[eu]
    return n_vertices, origens, destinos, grau, limites


def pagerank_por_destino(comm, n_vertices, origens, destinos_locais, grau, limites,
                         d=AMORTECIMENTO, tolerancia=TOLERANCIA, max_iteracoes=MAX_ITERACOES,
                         pesos_prontos=False):
    """Alteração 2 (e a otimizada, com pesos_prontos=True)."""
    eu = comm.Get_rank()
    inicio, fim = limites[eu], limites[eu + 1]
    # quantas páginas cada processo tem (é o "v" do Allgatherv)
    paginas_por_processo = np.diff(limites)
    sem_saida = np.flatnonzero(grau == 0)
    rank = np.full(n_vertices, 1.0 / n_vertices)

    if pesos_prontos:
        peso = 1.0 / grau[origens]

    iteracao = 0
    for iteracao in range(1, max_iteracoes + 1):
        por_link = rank[origens] * peso if pesos_prontos else rank[origens] / grau[origens]
        # o vetor tem só o tamanho da faixa deste processo, não N: cabe melhor no cache
        recebido = np.bincount(destinos_locais, weights=por_link, minlength=fim - inicio)

        massa = rank[sem_saida].sum()
        novo_local = np.ascontiguousarray(
            aplicar_formula(recebido, massa, n_vertices, d), dtype=np.float64)

        novo = np.empty(n_vertices)
        # cada processo entra com as notas das suas páginas e todos recebem o vetor inteiro
        comm.Allgatherv(novo_local, [novo, paginas_por_processo, limites[:-1], MPI.DOUBLE])

        parar = False
        if tolerancia > 0:
            # cada um mede a mudança nas suas páginas e o allreduce soma tudo
            variacao = comm.allreduce(np.abs(novo_local - rank[inicio:fim]).sum(), op=MPI.SUM)
            parar = variacao < tolerancia
        rank = novo
        if parar:
            break

    return rank, iteracao


def main() -> None:
    parser = argparse.ArgumentParser(description="PageRank com MPI, versões alteradas")
    parser.add_argument("--alteracao", choices=list(NOMES), required=True)
    parser.add_argument("--grafo", type=str, required=True, help="arquivo .npz do gerar_grafo")
    parser.add_argument("--amortecimento", type=float, default=AMORTECIMENTO)
    parser.add_argument("--tolerancia", type=float, default=TOLERANCIA,
                        help="critério de parada; 0 roda todas as iterações")
    parser.add_argument("--max-iteracoes", type=int, default=MAX_ITERACOES)
    parser.add_argument("--top", type=int, default=5, help="quantas páginas mostrar no ranking")
    parser.add_argument("--salvar-ranks", type=str, default=None,
                        help="arquivo .npy para guardar o vetor final (usado nos testes)")
    parser.add_argument("--csv", type=str, default=None, help="planilha onde anotar os tempos")
    parser.add_argument("--variante", type=str, default=None,
                        help="rótulo na planilha; o padrão é o nome da alteração")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    eu = comm.Get_rank()
    inicio_programa = MPI.Wtime()
    por_destino = args.alteracao in ("2", "otimizada")

    inicio = MPI.Wtime()
    grafo = carregar_grafo(args.grafo) if eu == 0 else None
    tempo_leitura = MPI.Wtime() - inicio

    comm.Barrier()
    inicio = MPI.Wtime()
    if por_destino:
        n_vertices, origens, destinos, grau, limites = distribuir_por_destino(comm, grafo)
    else:
        n_vertices, origens, destinos, grau = distribuir_grafo(comm, grafo)
    tempo_distribuicao = MPI.Wtime() - inicio
    del grafo

    opcoes = (args.amortecimento, args.tolerancia, args.max_iteracoes)
    comm.Barrier()
    inicio = MPI.Wtime()
    if por_destino:
        rank, iteracoes = pagerank_por_destino(comm, n_vertices, origens, destinos, grau, limites,
                                               *opcoes, pesos_prontos=args.alteracao == "otimizada")
    else:
        rank, iteracoes = pagerank_por_aresta(comm, n_vertices, origens, destinos, grau, *opcoes,
                                              pesos_prontos=args.alteracao == "1",
                                              embrulhado=args.alteracao == "3")
    tempo_calculo = MPI.Wtime() - inicio
    tempo_total = MPI.Wtime() - inicio_programa

    tempo_distribuicao = comm.reduce(tempo_distribuicao, op=MPI.MAX, root=0)
    tempo_calculo = comm.reduce(tempo_calculo, op=MPI.MAX, root=0)
    tempo_total = comm.reduce(tempo_total, op=MPI.MAX, root=0)
    total_arestas = comm.reduce(len(origens), op=MPI.SUM, root=0)

    if eu != 0:
        return

    variante = args.variante or NOMES[args.alteracao]
    print(f"Grafo: {n_vertices} vértices, {total_arestas} arestas")
    print(f"  alteração: {variante}")
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
        registrar(args.csv, versao="mpi", variante=variante, processos=comm.Get_size(),
                  vertices=n_vertices, arestas=total_arestas, iteracoes=iteracoes,
                  tempo_leitura=tempo_leitura, tempo_distribuicao=tempo_distribuicao,
                  tempo_calculo=tempo_calculo, tempo_total=tempo_total)


if __name__ == "__main__":
    main()
