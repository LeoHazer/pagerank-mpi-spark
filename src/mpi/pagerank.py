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
    # divmod dá a divisão inteira e o resto de uma vez: divmod(10, 3) = (3, 1)
    base, resto = divmod(total, partes)
    # os primeiros 'resto' processos ficam com um item a mais, para não sobrar nada
    # ex.: 10 links e 3 processos -> contagens [4, 3, 3]
    contagens = np.array([base + (1 if i < resto else 0) for i in range(partes)], dtype=np.int64)
    # deslocamento = posição onde começa a fatia de cada um na lista completa.
    # É a soma acumulada das contagens, começando do zero e sem o último total.
    # ex.: contagens [4, 3, 3] -> soma acumulada [4, 7, 10] -> deslocamentos [0, 4, 7]
    deslocamentos = np.concatenate(([0], np.cumsum(contagens)[:-1])).astype(np.int64)
    return contagens, deslocamentos


def distribuir_grafo(comm, grafo):
    """
    Só o rank 0 tem o grafo inteiro (os outros recebem None).
    Devolve, em todos os processos: (n_vertices, origens locais,
    destinos locais, grau de saída de todas as páginas).
    """
    # no MPI, "rank" é o número do processo (0, 1, 2, ...). Guardamos em 'eu' para não
    # confundir com a variável 'rank' do PageRank, que é o vetor de notas das páginas.
    eu = comm.Get_rank()
    # só o rank 0 sabe os tamanhos; os outros entram no bcast com None
    tamanhos = (grafo.n_vertices, grafo.n_arestas) if eu == 0 else None
    # bcast (broadcast = transmitir para todos): o rank 0 envia e todos os processos,
    # inclusive ele, recebem a mesma tupla. Com letra minúscula o mpi4py aceita um objeto
    # Python qualquer: ele é serializado (transformado em bytes com pickle) antes de ir.
    # Para uma tupla pequena como esta, isso não custa quase nada.
    n_vertices, n_arestas = comm.bcast(tamanhos, root=0)

    # todos fazem a mesma conta e chegam nas mesmas contagens, sem precisar de mensagem
    contagens, deslocamentos = dividir(n_arestas, comm.Get_size())
    # buffers de recebimento: espaço de memória reservado (ainda sem valores) onde o MPI
    # vai escrever a fatia deste processo. O tipo (int64) tem que bater com o enviado.
    origens = np.empty(contagens[eu], dtype=np.int64)
    destinos = np.empty(contagens[eu], dtype=np.int64)

    if eu == 0:
        # formato [buffer, quanto vai para cada um, onde começa cada fatia, tipo do dado]
        # ex.: com contagens [4, 3, 3] e deslocamentos [0, 4, 7], o processo 0 recebe os
        # links 0 a 3, o processo 1 os links 4 a 6 e o processo 2 os links 7 a 9.
        # ascontiguousarray garante o vetor em um bloco seguido de memória e em int64,
        # porque o MPI lê a memória "crua", byte a byte, sem saber o que é NumPy.
        envio_origens = [np.ascontiguousarray(grafo.origens, dtype=np.int64),
                         contagens, deslocamentos, MPI.INT64_T]
        envio_destinos = [np.ascontiguousarray(grafo.destinos, dtype=np.int64),
                          contagens, deslocamentos, MPI.INT64_T]
    else:
        # quem não é o rank 0 só recebe, então não tem nada para enviar
        envio_origens = envio_destinos = None

    # Scatterv (scatter = espalhar): o rank 0 corta o vetor em fatias e manda uma para cada
    # processo (ele mesmo também fica com a sua). Cada um recebe no seu buffer 'origens'.
    # Com letra maiúscula o mpi4py manda o buffer do NumPy direto, sem serializar;
    # o "v" (de variável) permite que cada processo receba uma quantidade diferente
    comm.Scatterv(envio_origens, origens, root=0)
    comm.Scatterv(envio_destinos, destinos, root=0)

    # todo processo precisa do grau de TODAS as páginas, não só das suas arestas,
    # porque a origem de um link local pode ser qualquer página do grafo
    if eu == 0:
        grau = grafo.grau_saida().astype(np.int64)
    else:
        # os outros reservam um buffer de tamanho N para receber o vetor
        grau = np.empty(n_vertices, dtype=np.int64)
    # Bcast maiúsculo: o rank 0 envia o vetor NumPy direto da memória e cada processo
    # recebe dentro do seu próprio 'grau' (o mesmo nome é envio no rank 0 e recebimento
    # nos outros)
    comm.Bcast(grau, root=0)

    return n_vertices, origens, destinos, grau


def pagerank_mpi(comm, n_vertices, origens, destinos, grau,
                 d=AMORTECIMENTO, tolerancia=TOLERANCIA, max_iteracoes=MAX_ITERACOES):
    """Mesmo algoritmo da versão sequencial, mas cada processo só olha as suas arestas."""
    # páginas sem links de saída (vetor de True/False, uma posição por página)
    sem_saida = grau == 0
    # ATENÇÃO: aqui 'rank' é o vetor de notas do PageRank (uma nota por página), não o
    # número do processo MPI (esse está em 'eu'). Começa com todas as páginas empatadas.
    rank = np.full(n_vertices, 1.0 / n_vertices)
    # buffer onde o Allreduce vai escrever a soma; é reservado uma vez e reaproveitado
    recebido = np.empty(n_vertices)

    # vale 0 se o laço não rodar nenhuma vez (max_iteracoes = 0)
    iteracao = 0
    for iteracao in range(1, max_iteracoes + 1):
        # cada processo soma o que as SUAS arestas levam para cada página. O vetor tem
        # tamanho N (todas as páginas), com zero nas páginas que nenhum link local atinge.
        parcial = somar_contribuicoes(origens, destinos, rank, grau, n_vertices)
        # soma os vetores parciais de todos os processos; todos recebem o total.
        # A soma é feita posição a posição. Ex.: processo 0 tem [0.1, 0, 0.2] e
        # processo 1 tem [0, 0.3, 0.1] -> os dois ficam com [0.1, 0.3, 0.3] em 'recebido'
        comm.Allreduce(parcial, recebido, op=MPI.SUM)

        # todo processo tem o vetor 'rank' inteiro e o 'grau' inteiro, então calcula a
        # massa e aplica a fórmula sozinho, sem nenhuma mensagem
        massa = rank[sem_saida].sum()
        novo = aplicar_formula(recebido, massa, n_vertices, d)

        # todos os processos têm o mesmo vetor, então todos decidem parar juntos.
        # Isso é importante: se um parasse e outro não, o que continuou ficaria
        # esperando para sempre no próximo Allreduce (que precisa de todos)
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

    # comunicador: o grupo com todos os processos que o mpirun iniciou
    comm = MPI.COMM_WORLD
    # número deste processo (0, 1, 2, ...); o mesmo programa roda em todos
    eu = comm.Get_rank()
    # MPI.Wtime é o relógio do MPI: devolve um horário em segundos. A diferença
    # entre duas leituras dá quanto tempo passou.
    inicio_programa = MPI.Wtime()

    # só o rank 0 lê o arquivo; os outros recebem a sua parte por mensagens do MPI.
    # Assim o arquivo é lido uma vez só (e não por todos ao mesmo tempo) e cada processo
    # guarda na memória só a sua parte das arestas, não o grafo inteiro.
    inicio = MPI.Wtime()
    grafo = carregar_grafo(args.grafo) if eu == 0 else None
    # nos outros processos esse tempo fica perto de zero; o impresso é o do rank 0
    tempo_leitura = MPI.Wtime() - inicio

    # Barrier: todo mundo espera aqui para o cronômetro começar junto
    # (ninguém passa da barreira até todos os processos chegarem nela)
    comm.Barrier()
    inicio = MPI.Wtime()
    n_vertices, origens, destinos, grau = distribuir_grafo(comm, grafo)
    tempo_distribuicao = MPI.Wtime() - inicio
    # o rank 0 não precisa mais da cópia inteira do grafo
    del grafo

    # nova barreira para que a medição do cálculo também comece igual para todos
    comm.Barrier()
    inicio = MPI.Wtime()
    rank, iteracoes = pagerank_mpi(comm, n_vertices, origens, destinos, grau,
                                   args.amortecimento, args.tolerancia, args.max_iteracoes)
    tempo_calculo = MPI.Wtime() - inicio
    tempo_total = MPI.Wtime() - inicio_programa

    # o tempo que vale é o do processo mais lento, porque o trabalho só acaba quando
    # o último termina. reduce minúsculo: cada processo manda um número Python e só o
    # rank 0 (root) recebe o resultado da operação (aqui, o maior); os outros ficam com None
    tempo_distribuicao = comm.reduce(tempo_distribuicao, op=MPI.MAX, root=0)
    tempo_calculo = comm.reduce(tempo_calculo, op=MPI.MAX, root=0)
    tempo_total = comm.reduce(tempo_total, op=MPI.MAX, root=0)
    # confere que nenhuma aresta se perdeu na divisão (soma das fatias de todos)
    total_arestas = comm.reduce(len(origens), op=MPI.SUM, root=0)

    # daqui para baixo é só relatório: os outros processos já fizeram a sua parte
    # (e, depois do reduce, só o rank 0 tem os tempos)
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
