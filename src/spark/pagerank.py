"""
PageRank com Apache Spark (PySpark, API de DataFrames).

No MPI nós mesmos dividimos o trabalho e organizamos a comunicação. No
Spark descrevemos o cálculo como operações de tabela (juntar, agrupar,
somar) e o próprio Spark decide como espalhar as linhas entre os
executores e como eles trocam dados.

Analogia: no MPI cada funcionário sabe o que fazer e combina com os colegas
por rádio. No Spark existe um gerente (o driver) que recebe a receita e
distribui as tarefas para os funcionários (os executores).

Tabelas usadas:
  - arestas:   (origem, destino, peso), com peso = 1 / grau de saída da origem
  - vertices:  (id), uma linha por página
  - sem_saida: (id), páginas que não apontam para ninguém
  - ranks:     (id, rank), a nota de cada página na rodada atual

Uso (a partir da raiz do projeto):
    python -m src.spark.pagerank --grafo dados/grafo_100k.npz
    python -m src.spark.pagerank --grafo dados/grafo_100k.npz --master "local[4]"
"""

import argparse
import time

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.comum.grafo import carregar_grafo
from src.comum.relatorio import linhas_do_top, porcentagem
from src.comum.resultados import registrar
from src.sequencial.pagerank import AMORTECIMENTO, MAX_ITERACOES, TOLERANCIA


MEMORIA = "4g"


def criar_sessao(master=None, nucleos=None, memoria=MEMORIA):
    """Abre a conexão com o Spark. Sem master, usa o que o spark-submit passar (ou local)."""
    construtor = SparkSession.builder.appName("pagerank")
    if master:
        construtor = construtor.master(master)
    # o padrão do Spark é 1 GB, pouco para o grafo de 10 milhões de links;
    # driver = o "gerente" (no modo local ele também faz as contas), executor = cada "funcionário"
    construtor = construtor.config("spark.driver.memory", memoria)
    construtor = construtor.config("spark.executor.memory", memoria)
    if nucleos:
        # no cluster, limita quantos núcleos o programa pode pegar no total
        construtor = construtor.config("spark.cores.max", str(nucleos))
    # Arrow acelera a passagem de tabelas entre o Python e o Spark (que roda em Java)
    construtor = construtor.config("spark.sql.execution.arrow.pyspark.enabled", "true")
    # tira a barra de progresso "[Stage 0:> ...]" do terminal
    construtor = construtor.config("spark.ui.showConsoleProgress", "false")
    spark = construtor.getOrCreate()
    # esconde as mensagens de log que não são erro
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def preparar_tabelas(spark, grafo, particoes):
    """
    Transforma o grafo (arrays do NumPy, no driver) em tabelas do Spark,
    já divididas em 'particoes' pedaços e guardadas na memória dos executores.
    """
    # partições = em quantos pedaços cada tabela é dividida entre os executores;
    # "shuffle" é a troca de linhas entre executores que acontece em join e groupBy
    spark.conf.set("spark.sql.shuffle.partitions", str(particoes))

    grau = grafo.grau_saida()
    pares = pd.DataFrame({
        "origem": grafo.origens,
        "destino": grafo.destinos,
        # a fração da nota que viaja por cada link não muda entre as rodadas
        "peso": 1.0 / grau[grafo.origens],
    })
    arestas = spark.createDataFrame(pares).repartition(particoes, "origem")
    # spark.range cria a tabela com a coluna "id" = 0, 1, ..., N-1
    vertices = spark.range(grafo.n_vertices).repartition(particoes, "id")
    sem_saida = spark.createDataFrame(pd.DataFrame({"id": np.flatnonzero(grau == 0)}), "id long")

    # cache guarda a tabela na memória; o count obriga o Spark a montá-la agora
    for tabela in (arestas, vertices, sem_saida):
        tabela.cache().count()
    return arestas, vertices, sem_saida


def pagerank_spark(arestas, vertices, sem_saida, n_vertices,
                   d=AMORTECIMENTO, tolerancia=TOLERANCIA, max_iteracoes=MAX_ITERACOES):
    """
    Mesmo algoritmo da versão sequencial, escrito com operações de tabela.
    Devolve (tabela de ranks, número de iterações feitas).
    """
    # começa com todas as páginas empatadas
    ranks = vertices.withColumn("rank", F.lit(1.0 / n_vertices)).localCheckpoint()

    iteracao = 0
    for iteracao in range(1, max_iteracoes + 1):
        # cada link leva rank[origem] * peso para o destino; somamos tudo por destino
        recebido = (
            arestas.join(ranks, arestas.origem == ranks.id)
            .groupBy("destino")
            .agg(F.sum(F.col("rank") * F.col("peso")).alias("recebido"))
        )

        # nota total das páginas sem saída; o first() traz esse número para o driver
        massa = sem_saida.join(ranks, "id").agg(F.sum("rank")).first()[0] or 0.0

        # left join: páginas que não receberam nenhum link também ficam na tabela
        novo = (
            vertices.join(recebido, vertices.id == recebido.destino, "left")
            .select(
                "id",
                (F.lit((1.0 - d) / n_vertices)
                 + F.lit(d) * (F.coalesce("recebido", F.lit(0.0)) + F.lit(massa / n_vertices))
                 ).alias("rank"),
            )
        )
        # o Spark é "preguiçoso": só calcula quando precisa e lembra de todas as
        # rodadas anteriores; o localCheckpoint calcula agora e esquece esse histórico
        novo = novo.localCheckpoint()

        parar = False
        if tolerancia > 0:
            # quanto o vetor mudou nesta rodada (distância L1)
            variacao = (
                novo.join(ranks.withColumnRenamed("rank", "antigo"), "id")
                .agg(F.sum(F.abs(F.col("rank") - F.col("antigo"))))
                .first()[0]
            )
            parar = variacao < tolerancia

        ranks = novo
        if parar:
            break

    return ranks, iteracao


def para_numpy(ranks):
    """Traz a tabela de ranks para o driver como um vetor na ordem dos ids."""
    tabela = ranks.orderBy("id").toPandas()
    return tabela["rank"].to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="PageRank com Spark")
    parser.add_argument("--grafo", type=str, required=True, help="arquivo .npz do gerar_grafo")
    parser.add_argument("--amortecimento", type=float, default=AMORTECIMENTO)
    parser.add_argument("--tolerancia", type=float, default=TOLERANCIA,
                        help="critério de parada; 0 roda todas as iterações")
    parser.add_argument("--max-iteracoes", type=int, default=MAX_ITERACOES)
    parser.add_argument("--top", type=int, default=5, help="quantas páginas mostrar no ranking")
    parser.add_argument("--master", type=str, default=None,
                        help='ex.: "local[4]" (4 núcleos nesta máquina) ou spark://ip:7077')
    parser.add_argument("--particoes", type=int, default=None,
                        help="pedaços de cada tabela; o padrão é o número de núcleos")
    parser.add_argument("--salvar-ranks", type=str, default=None,
                        help="arquivo .npy para guardar o vetor final")
    parser.add_argument("--nucleos", type=int, default=None,
                        help="limite de núcleos no cluster (no modo local use local[N] no --master)")
    parser.add_argument("--memoria", type=str, default=MEMORIA,
                        help="memória do driver e de cada executor, ex.: 4g")
    parser.add_argument("--csv", type=str, default=None, help="planilha onde anotar os tempos")
    parser.add_argument("--variante", type=str, default="base", help="rótulo da alteração testada")
    args = parser.parse_args()
    inicio_programa = time.perf_counter()

    spark = criar_sessao(args.master, args.nucleos, args.memoria)
    # defaultParallelism = total de núcleos que o Spark recebeu
    nucleos = args.nucleos or spark.sparkContext.defaultParallelism
    particoes = args.particoes or nucleos

    inicio = time.perf_counter()
    grafo = carregar_grafo(args.grafo)
    tempo_leitura = time.perf_counter() - inicio

    inicio = time.perf_counter()
    arestas, vertices, sem_saida = preparar_tabelas(spark, grafo, particoes)
    tempo_distribuicao = time.perf_counter() - inicio

    inicio = time.perf_counter()
    ranks, iteracoes = pagerank_spark(arestas, vertices, sem_saida, grafo.n_vertices,
                                      args.amortecimento, args.tolerancia, args.max_iteracoes)
    tempo_calculo = time.perf_counter() - inicio
    tempo_total = time.perf_counter() - inicio_programa

    rank = para_numpy(ranks)

    print(f"Grafo: {grafo.n_vertices} vértices, {grafo.n_arestas} arestas")
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
        registrar(args.csv, versao="spark", variante=args.variante, processos=nucleos,
                  particoes=particoes, vertices=grafo.n_vertices, arestas=grafo.n_arestas,
                  iteracoes=iteracoes, tempo_leitura=tempo_leitura,
                  tempo_distribuicao=tempo_distribuicao, tempo_calculo=tempo_calculo,
                  tempo_total=tempo_total)
    spark.stop()


if __name__ == "__main__":
    main()
