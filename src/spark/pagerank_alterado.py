"""
Versões alteradas do PageRank com Spark, para comparar com a original
(src/spark/pagerank.py, que continua igual).

Escolha a alteração com --alteracao:
  1          Trabalho repetido em cada rodada: a original faz duas "idas ao
             gerente" por rodada (uma para somar a nota das páginas sem
             saída e outra para calcular a tabela nova). Aqui o Spark anota
             essa soma enquanto calcula a tabela nova (observe), e a ida
             extra deixa de existir.
  2          Divisão dos dados por página de destino: os links já ficam
             divididos pelo destino e a tabela de notas vai inteira para
             cada executor (broadcast). Assim a soma por destino acontece
             sem reembaralhar os links entre os executores a cada rodada.
  3          Formato dos dados: tudo escrito com RDD, em que cada linha é um
             objeto Python serializado (pickle) e as contas são feitas pelo
             Python, em vez de DataFrames, em que tudo roda dentro do Spark.
  otimizada  alterações 1 e 2 juntas.

Uso (a partir da raiz do projeto):
    python -m src.spark.pagerank_alterado --alteracao 2 --grafo dados/grafo_100k.npz --master "local[4]"
"""

import argparse
import time
from operator import add

import numpy as np
import pandas as pd
from pyspark.sql import Observation
from pyspark.sql import functions as F

from src.comum.grafo import carregar_grafo
from src.comum.relatorio import linhas_do_top, porcentagem
from src.comum.resultados import registrar
from src.sequencial.pagerank import AMORTECIMENTO, MAX_ITERACOES, TOLERANCIA
from src.spark.pagerank import MEMORIA, criar_sessao, para_numpy

NOMES = {"1": "alteracao1", "2": "alteracao2", "3": "alteracao3", "otimizada": "otimizada"}


# ---------------------------------------------------------------------------
# Alterações 1, 2 e otimizada (DataFrames)
# ---------------------------------------------------------------------------

def preparar_tabelas(spark, grafo, particoes, por_destino):
    """Igual à original, mas com a divisão por destino (alteração 2) e a marca de página sem saída."""
    spark.conf.set("spark.sql.shuffle.partitions", str(particoes))

    grau = grafo.grau_saida()
    pares = pd.DataFrame({
        "origem": grafo.origens,
        "destino": grafo.destinos,
        "peso": 1.0 / grau[grafo.origens],
    })
    # alteração 2: cada pedaço da tabela fica com os links que chegam num grupo de páginas
    chave = "destino" if por_destino else "origem"
    arestas = spark.createDataFrame(pares).repartition(particoes, chave)

    # a coluna sem_saida marca as páginas que não apontam para ninguém (usada na alteração 1)
    paginas = pd.DataFrame({"id": np.arange(grafo.n_vertices), "sem_saida": grau == 0})
    vertices = spark.createDataFrame(paginas).repartition(particoes, "id")
    sem_saida = spark.createDataFrame(pd.DataFrame({"id": np.flatnonzero(grau == 0)}), "id long")

    for tabela in (arestas, vertices, sem_saida):
        tabela.cache().count()
    return arestas, vertices, sem_saida


def pagerank_tabelas(arestas, vertices, sem_saida, n_vertices,
                     d=AMORTECIMENTO, tolerancia=TOLERANCIA, max_iteracoes=MAX_ITERACOES,
                     observar_massa=False, broadcast=False):
    """Mesmo algoritmo da original, com a alteração 1 (observar_massa) e/ou a 2 (broadcast)."""
    # na alteração 1 a tabela de notas carrega junto a marca de página sem saída
    colunas = ["id", "sem_saida"] if observar_massa else ["id"]
    ranks = vertices.select(*colunas).withColumn("rank", F.lit(1.0 / n_vertices)).localCheckpoint()

    if observar_massa:
        # na primeira rodada todas valem 1/N; a massa é (quantas páginas sem saída) / N
        massa = sem_saida.count() / n_vertices

    iteracao = 0
    for iteracao in range(1, max_iteracoes + 1):
        # alteração 2: broadcast manda a tabela de notas inteira para cada executor,
        # então o join acontece ali mesmo, sem trocar links entre executores
        notas = F.broadcast(ranks) if broadcast else ranks
        recebido = (
            arestas.join(notas, arestas.origem == notas.id)
            .groupBy("destino")
            .agg(F.sum(F.col("rank") * F.col("peso")).alias("recebido"))
        )

        if not observar_massa:
            # original: uma ida extra ao gerente só para somar a nota das páginas sem saída
            massa = sem_saida.join(ranks, "id").agg(F.sum("rank")).first()[0] or 0.0

        novo = vertices.join(recebido, vertices.id == recebido.destino, "left").select(
            *colunas,
            (F.lit((1.0 - d) / n_vertices)
             + F.lit(d) * (F.coalesce("recebido", F.lit(0.0)) + F.lit(massa / n_vertices))
             ).alias("rank"),
        )

        if observar_massa:
            # alteração 1: enquanto monta a tabela nova, o Spark já anota a soma das
            # páginas sem saída, que é exatamente a massa da próxima rodada
            observacao = Observation()
            novo = novo.observe(
                observacao,
                F.sum(F.when(F.col("sem_saida"), F.col("rank")).otherwise(0.0)).alias("massa"),
            )
        novo = novo.localCheckpoint()
        if observar_massa:
            massa = observacao.get["massa"] or 0.0

        parar = False
        if tolerancia > 0:
            variacao = (
                novo.join(ranks.select("id", F.col("rank").alias("antigo")), "id")
                .agg(F.sum(F.abs(F.col("rank") - F.col("antigo"))))
                .first()[0]
            )
            parar = variacao < tolerancia

        ranks = novo
        if parar:
            break

    return ranks, iteracao


# ---------------------------------------------------------------------------
# Alteração 3 (RDD)
# ---------------------------------------------------------------------------

def preparar_rdds(spark, grafo, particoes):
    """
    As mesmas três tabelas da original, mas como RDDs de tuplas do Python.
    Cada link vira (origem, (destino, peso)); cada página vira (id, 0).
    """
    sc = spark.sparkContext
    grau = grafo.grau_saida()
    peso = 1.0 / grau[grafo.origens]

    # mandamos blocos do NumPy e cada executor transforma o seu bloco em tuplas;
    # criar 10 milhões de tuplas no driver de uma vez gastaria memória demais
    blocos = list(zip(np.array_split(grafo.origens, particoes * 4),
                      np.array_split(grafo.destinos, particoes * 4),
                      np.array_split(peso, particoes * 4)))
    arestas = (
        sc.parallelize(blocos, len(blocos))
        .flatMap(lambda b: zip(b[0].tolist(), zip(b[1].tolist(), b[2].tolist())))
        # partitionBy: todos os links de uma mesma origem vão para o mesmo pedaço
        .partitionBy(particoes)
    )
    vertices = sc.parallelize(range(grafo.n_vertices), particoes).map(lambda v: (v, 0))
    sem_saida = sc.parallelize([(int(v), 0) for v in np.flatnonzero(grau == 0)], particoes)

    for rdd in (arestas, vertices, sem_saida):
        rdd.cache().count()
    return arestas, vertices, sem_saida


def pagerank_rdd(arestas, vertices, sem_saida, n_vertices, particoes,
                 d=AMORTECIMENTO, tolerancia=TOLERANCIA, max_iteracoes=MAX_ITERACOES):
    """Mesmos passos da versão com tabelas: join, soma por destino e left join."""
    ranks = vertices.mapValues(lambda _: 1.0 / n_vertices)
    ranks.localCheckpoint()
    ranks.count()

    iteracao = 0
    for iteracao in range(1, max_iteracoes + 1):
        # (origem, ((destino, peso), rank)) -> (destino, rank * peso), somado por destino
        recebido = (
            arestas.join(ranks, particoes)
            .map(lambda par: (par[1][0][0], par[1][1] * par[1][0][1]))
            .reduceByKey(add, particoes)
        )
        massa = sem_saida.join(ranks, particoes).map(lambda par: par[1][1]).sum()

        # m=massa "congela" o valor desta rodada dentro da função
        novo = vertices.leftOuterJoin(recebido, particoes).mapValues(
            lambda par, m=massa: (1.0 - d) / n_vertices + d * ((par[1] or 0.0) + m / n_vertices)
        )
        # no RDD o localCheckpoint só marca; o count é que obriga o cálculo agora
        novo.localCheckpoint()
        novo.count()

        parar = False
        if tolerancia > 0:
            variacao = novo.join(ranks, particoes).map(lambda par: abs(par[1][0] - par[1][1])).sum()
            parar = variacao < tolerancia

        ranks = novo
        if parar:
            break

    return ranks, iteracao


def rdd_para_numpy(ranks):
    """Traz as notas para o driver, na ordem dos ids."""
    return np.array([rank for _, rank in sorted(ranks.collect())])


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PageRank com Spark, versões alteradas")
    parser.add_argument("--alteracao", choices=list(NOMES), required=True)
    parser.add_argument("--grafo", type=str, required=True, help="arquivo .npz do gerar_grafo")
    parser.add_argument("--amortecimento", type=float, default=AMORTECIMENTO)
    parser.add_argument("--tolerancia", type=float, default=TOLERANCIA,
                        help="critério de parada; 0 roda todas as iterações")
    parser.add_argument("--max-iteracoes", type=int, default=MAX_ITERACOES)
    parser.add_argument("--top", type=int, default=5, help="quantas páginas mostrar no ranking")
    parser.add_argument("--master", type=str, default=None,
                        help='ex.: "local[4]" (4 núcleos nesta máquina) ou spark://ip:7077')
    parser.add_argument("--nucleos", type=int, default=None,
                        help="limite de núcleos no cluster (no modo local use local[N] no --master)")
    parser.add_argument("--particoes", type=int, default=None,
                        help="pedaços de cada tabela; o padrão é o número de núcleos")
    parser.add_argument("--memoria", type=str, default=MEMORIA,
                        help="memória do driver e de cada executor, ex.: 4g")
    parser.add_argument("--salvar-ranks", type=str, default=None,
                        help="arquivo .npy para guardar o vetor final")
    parser.add_argument("--csv", type=str, default=None, help="planilha onde anotar os tempos")
    parser.add_argument("--variante", type=str, default=None,
                        help="rótulo na planilha; o padrão é o nome da alteração")
    args = parser.parse_args()
    inicio_programa = time.perf_counter()

    spark = criar_sessao(args.master, args.nucleos, args.memoria)
    nucleos = args.nucleos or spark.sparkContext.defaultParallelism
    particoes = args.particoes or nucleos
    usa_rdd = args.alteracao == "3"
    opcoes = (args.amortecimento, args.tolerancia, args.max_iteracoes)

    inicio = time.perf_counter()
    grafo = carregar_grafo(args.grafo)
    tempo_leitura = time.perf_counter() - inicio

    inicio = time.perf_counter()
    if usa_rdd:
        tabelas = preparar_rdds(spark, grafo, particoes)
    else:
        tabelas = preparar_tabelas(spark, grafo, particoes,
                                   por_destino=args.alteracao in ("2", "otimizada"))
    tempo_distribuicao = time.perf_counter() - inicio

    inicio = time.perf_counter()
    if usa_rdd:
        ranks, iteracoes = pagerank_rdd(*tabelas, grafo.n_vertices, particoes, *opcoes)
    else:
        ranks, iteracoes = pagerank_tabelas(
            *tabelas, grafo.n_vertices, *opcoes,
            observar_massa=args.alteracao in ("1", "otimizada"),
            broadcast=args.alteracao in ("2", "otimizada"),
        )
    tempo_calculo = time.perf_counter() - inicio
    tempo_total = time.perf_counter() - inicio_programa

    rank = rdd_para_numpy(ranks) if usa_rdd else para_numpy(ranks)

    variante = args.variante or NOMES[args.alteracao]
    print(f"Grafo: {grafo.n_vertices} vértices, {grafo.n_arestas} arestas")
    print(f"  alteração: {variante}")
    print(f"  núcleos: {nucleos} | partições: {particoes}")
    print(f"  iterações: {iteracoes}")
    print(f"  soma das chances: {porcentagem(rank.sum())}")
    print(f"  tempo de leitura: {tempo_leitura:.3f} s")
    print(f"  tempo de distribuição: {tempo_distribuicao:.3f} s")
    print(f"  tempo de cálculo: {tempo_calculo:.3f} s")
    print("\n".join(linhas_do_top(rank, args.top)))

    if args.salvar_ranks:
        np.save(args.salvar_ranks, rank)
    if args.csv:
        registrar(args.csv, versao="spark", variante=variante, processos=nucleos,
                  particoes=particoes, vertices=grafo.n_vertices, arestas=grafo.n_arestas,
                  iteracoes=iteracoes, tempo_leitura=tempo_leitura,
                  tempo_distribuicao=tempo_distribuicao, tempo_calculo=tempo_calculo,
                  tempo_total=tempo_total)
    spark.stop()


if __name__ == "__main__":
    main()
