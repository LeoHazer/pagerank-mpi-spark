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
    # 'rank' é o vetor de notas do PageRank, não o número do processo MPI
    rank = np.full(n_vertices, 1.0 / n_vertices)
    # buffer reservado uma vez, onde o Allreduce maiúsculo escreve a soma
    recebido = np.empty(n_vertices)

    if pesos_prontos:
        # alteração 1: a fração da nota que passa por cada link não muda entre as rodadas
        # peso[i] = 1 / grau da origem do link i. Ex.: se a página 3 tem 4 links de saída,
        # cada um desses links tem peso 0,25. O grau nunca é zero aqui, porque a origem
        # tem pelo menos esse link.
        peso = 1.0 / grau[origens]

    iteracao = 0
    for iteracao in range(1, max_iteracoes + 1):
        if pesos_prontos:
            # multiplicar pelo peso pronto custa menos que buscar o grau e dividir
            # bincount com pesos soma, para cada página de destino, tudo o que chega nela
            parcial = np.bincount(destinos, weights=rank[origens] * peso, minlength=n_vertices)
            # sem nenhum link o bincount devolve inteiros; o MPI precisa de float64 em todos
            parcial = parcial.astype(np.float64, copy=False)
        else:
            # caminho da versão original: divide pelo grau em toda rodada
            parcial = somar_contribuicoes(origens, destinos, rank, grau, n_vertices)

        if embrulhado:
            # alteração 3: o vetor é serializado (pickle), enviado como bytes,
            # desembrulhado do outro lado e somado pelo próprio Python
            # Repare que o minúsculo devolve um vetor novo com o resultado, em vez de
            # escrever no buffer 'recebido' já reservado.
            recebido = comm.allreduce(parcial, op=MPI.SUM)
        else:
            # como na original: soma posição a posição direto na memória, e todos recebem
            comm.Allreduce(parcial, recebido, op=MPI.SUM)

        # daqui para baixo é igual à versão original; todos têm o mesmo vetor e por isso
        # decidem parar juntos
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
    # 'eu' = número deste processo no MPI (o "rank" do MPI)
    eu = comm.Get_rank()
    # nos outros processos estas variáveis ficam None: só o rank 0 tem o grafo
    origens_ordenadas = destinos_ordenados = None
    info = None
    if eu == 0:
        # limites tem (processos + 1) números: o processo k é dono das páginas de
        # limites[k] até limites[k + 1] - 1. As faixas são cortadas pela quantidade de
        # links que chegam nelas, não pela quantidade de páginas, para dividir o trabalho
        # por igual. Ex. (10 páginas, 3 processos, links concentrados nas páginas baixas):
        # limites [0, 1, 3, 10] -> processo 0 fica com a página 0, processo 1 com as
        # páginas 1 e 2, processo 2 com as páginas 3 a 9. links[k] = links da faixa k.
        limites, links = dividir_por_destino(grafo.destinos, grafo.n_vertices, comm.Get_size())
        # faixa[i] = número do processo dono da página de destino do link i
        # searchsorted conta quantos limites são <= destino; menos 1 dá a faixa.
        # Ex.: limites [0, 1, 3, 10] e destino 2 -> dois limites (0 e 1) -> faixa 1
        faixa = np.searchsorted(limites, grafo.destinos, side="right") - 1
        # agrupa os links por faixa e, dentro de cada faixa, deixa em ordem de origem;
        # assim rank[origens] lê a memória em sequência, como na versão original
        # (lexsort ordena pela última chave e desempata pela anterior)
        # Ex.: faixa [1, 0, 1, 0] e origens [5, 2, 1, 7] -> ordem [1, 3, 2, 0]
        # (primeiro os links da faixa 0, com origens 2 e 7; depois os da faixa 1, com 1 e 5).
        # Agrupar por faixa é obrigatório: o Scatterv manda pedaços seguidos do vetor.
        # Ler a memória em sequência é mais rápido porque o processador traz blocos
        # vizinhos para o cache (uma memória pequena e muito rápida dentro dele).
        ordem = np.lexsort((grafo.origens, faixa))
        # aplica a nova ordem nas duas listas, num bloco seguido de memória e em int64
        origens_ordenadas = np.ascontiguousarray(grafo.origens[ordem], dtype=np.int64)
        destinos_ordenados = np.ascontiguousarray(grafo.destinos[ordem], dtype=np.int64)
        info = (grafo.n_vertices, limites, links)
    # bcast minúsculo: o rank 0 envia a tupla (serializada com pickle) e todos recebem.
    # São poucos números (um ou dois por processo), então o pickle sai barato.
    n_vertices, limites, links = comm.bcast(info, root=0)

    # onde começa a fatia de cada processo no vetor ordenado (soma acumulada dos links)
    # ex.: links [4, 3, 3] -> deslocamentos [0, 4, 7]
    deslocamentos = np.concatenate(([0], np.cumsum(links)[:-1])).astype(np.int64)
    # buffers de recebimento, do tamanho da fatia deste processo
    origens = np.empty(links[eu], dtype=np.int64)
    destinos = np.empty(links[eu], dtype=np.int64)
    if eu == 0:
        # mesmo formato da original: [buffer, contagens, deslocamentos, tipo do dado],
        # só que agora as contagens são os links de cada faixa
        envio_origens = [origens_ordenadas, links, deslocamentos, MPI.INT64_T]
        envio_destinos = [destinos_ordenados, links, deslocamentos, MPI.INT64_T]
    else:
        envio_origens = envio_destinos = None
    # o rank 0 reparte as fatias (cada processo pode receber uma quantidade diferente)
    comm.Scatterv(envio_origens, origens, root=0)
    comm.Scatterv(envio_destinos, destinos, root=0)

    # igual à original: todos precisam do grau de todas as páginas, porque a origem de
    # um link pode ser qualquer página; o rank 0 envia e os outros recebem no buffer vazio
    grau = grafo.grau_saida().astype(np.int64) if eu == 0 else np.empty(n_vertices, dtype=np.int64)
    comm.Bcast(grau, root=0)

    # página limites[eu] vira posição 0 do vetor local, a seguinte vira 1, e assim por diante
    # Ex.: processo 1 com faixa de páginas 1 e 2: destino 1 vira 0 e destino 2 vira 1.
    # Assim o processo pode somar num vetor do tamanho da sua faixa, e não de tamanho N.
    destinos -= limites[eu]
    return n_vertices, origens, destinos, grau, limites


def pagerank_por_destino(comm, n_vertices, origens, destinos_locais, grau, limites,
                         d=AMORTECIMENTO, tolerancia=TOLERANCIA, max_iteracoes=MAX_ITERACOES,
                         pesos_prontos=False):
    """Alteração 2 (e a otimizada, com pesos_prontos=True)."""
    eu = comm.Get_rank()
    # faixa deste processo: páginas de 'inicio' até 'fim' - 1 ('fim' não entra)
    inicio, fim = limites[eu], limites[eu + 1]
    # quantas páginas cada processo tem (é o "v" do Allgatherv)
    # ex.: limites [0, 1, 3, 10] -> [1, 2, 7]
    paginas_por_processo = np.diff(limites)
    # posições das páginas sem links de saída (lista de números, não de True/False)
    sem_saida = np.flatnonzero(grau == 0)
    # 'rank' é o vetor de notas de TODAS as páginas (não o número do processo MPI).
    # Cada processo guarda o vetor inteiro, porque seus links podem vir de qualquer página.
    rank = np.full(n_vertices, 1.0 / n_vertices)

    if pesos_prontos:
        # versão otimizada: o peso de cada link (alteração 1) é calculado uma vez só
        peso = 1.0 / grau[origens]

    iteracao = 0
    for iteracao in range(1, max_iteracoes + 1):
        # parte da nota que viaja por cada link local: com peso pronto só multiplica,
        # sem ele divide pelo grau em toda rodada (como na original)
        por_link = rank[origens] * peso if pesos_prontos else rank[origens] / grau[origens]
        # o vetor tem só o tamanho da faixa deste processo, não N: cabe melhor no cache
        # Como este processo recebeu TODOS os links que chegam nas suas páginas, essa soma
        # já é o total final delas; não precisa somar com os outros processos.
        recebido = np.bincount(destinos_locais, weights=por_link, minlength=fim - inicio)

        # a fórmula vale página a página, então cada um aplica só nas suas páginas
        massa = rank[sem_saida].sum()
        # float64 em um bloco seguido de memória, formato que o Allgatherv vai ler
        novo_local = np.ascontiguousarray(
            aplicar_formula(recebido, massa, n_vertices, d), dtype=np.float64)

        # buffer de recebimento para o vetor inteiro de notas
        novo = np.empty(n_vertices)
        # cada processo entra com as notas das suas páginas e todos recebem o vetor inteiro
        # Formato [buffer, contagens, deslocamentos, tipo]: o pedaço do processo k entra
        # em novo a partir da posição limites[k]. Ex.: com limites [0, 1, 3, 10], o
        # processo 0 manda 1 nota (posição 0), o 1 manda 2 (posições 1 e 2) e o 2 manda 7
        # (posições 3 a 9). Na original cada processo mandava um vetor de tamanho N;
        # aqui cada um manda só a sua faixa.
        comm.Allgatherv(novo_local, [novo, paginas_por_processo, limites[:-1], MPI.DOUBLE])

        parar = False
        # com tolerancia 0 roda todas as rodadas e pula essa mensagem extra
        if tolerancia > 0:
            # cada um mede a mudança nas suas páginas e o allreduce soma tudo
            # É um número só, então o allreduce minúsculo (com pickle) sai barato. Todos
            # recebem a mesma soma e por isso decidem parar juntos.
            # Por que não calcular a variação no vetor inteiro, que todos já têm depois
            # do Allgatherv? Daria o mesmo resultado, mas cada processo percorreria as N
            # páginas; assim cada um percorre só as da sua faixa (N/p) e o trabalho
            # continua dividido, que é a ideia desta alteração.
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

    # grupo com todos os processos e o número deste processo (o "rank" do MPI)
    comm = MPI.COMM_WORLD
    eu = comm.Get_rank()
    # MPI.Wtime: relógio do MPI, em segundos; a diferença entre duas leituras é o tempo gasto
    inicio_programa = MPI.Wtime()
    # as alterações 2 e otimizada dividem os links por página de destino;
    # as alterações 1 e 3 mantêm a divisão em pedaços iguais da original
    por_destino = args.alteracao in ("2", "otimizada")

    # como na original, só o rank 0 lê o arquivo e depois reparte os dados
    inicio = MPI.Wtime()
    grafo = carregar_grafo(args.grafo) if eu == 0 else None
    tempo_leitura = MPI.Wtime() - inicio

    # Barrier: ninguém passa daqui até todos chegarem, para o cronômetro começar junto
    comm.Barrier()
    inicio = MPI.Wtime()
    if por_destino:
        n_vertices, origens, destinos, grau, limites = distribuir_por_destino(comm, grafo)
    else:
        n_vertices, origens, destinos, grau = distribuir_grafo(comm, grafo)
    tempo_distribuicao = MPI.Wtime() - inicio
    del grafo

    # mesmos parâmetros para qualquer alteração; o *opcoes abaixo "abre" a tupla
    # nos três argumentos d, tolerancia e max_iteracoes
    opcoes = (args.amortecimento, args.tolerancia, args.max_iteracoes)
    comm.Barrier()
    # só o cálculo do PageRank entra no tempo medido (leitura e distribuição ficam fora)
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

    # vale o tempo do processo mais lento: o reduce com MPI.MAX entrega o maior valor
    # só ao rank 0 (os outros ficam com None)
    tempo_distribuicao = comm.reduce(tempo_distribuicao, op=MPI.MAX, root=0)
    tempo_calculo = comm.reduce(tempo_calculo, op=MPI.MAX, root=0)
    tempo_total = comm.reduce(tempo_total, op=MPI.MAX, root=0)
    # soma dos links de todos, para conferir que nenhum se perdeu na divisão
    total_arestas = comm.reduce(len(origens), op=MPI.SUM, root=0)

    # só o rank 0 imprime e salva os resultados
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
