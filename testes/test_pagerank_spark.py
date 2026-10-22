"""
Testes da versão Spark. Todos comparam o resultado com a versão sequencial.
Abrir o Spark demora alguns segundos, então uma única sessão é usada por
todos os testes deste arquivo.
"""

import shutil

import pytest

from src.comum.grafo import gerar_grafo_aleatorio, grafo_exemplo_4_paginas
from src.comum.metricas import erro_l1
from src.sequencial.pagerank import pagerank

# sem pyspark ou sem Java (ex.: Windows) os testes são pulados em vez de falhar
pytest.importorskip("pyspark")
pytestmark = pytest.mark.skipif(shutil.which("java") is None, reason="Java não encontrado")

from src.spark.pagerank import criar_sessao, pagerank_spark, para_numpy, preparar_tabelas  # noqa: E402


@pytest.fixture(scope="module")
def spark():
    sessao = criar_sessao("local[2]")
    yield sessao
    sessao.stop()


def rodar_spark(spark, grafo, **opcoes):
    tabelas = preparar_tabelas(spark, grafo, particoes=4)
    ranks, iteracoes = pagerank_spark(*tabelas, grafo.n_vertices, **opcoes)
    return para_numpy(ranks), iteracoes


def test_spark_igual_ao_sequencial_com_iteracoes_fixas(spark):
    grafo = gerar_grafo_aleatorio(2000, 6, semente=11)
    esperado, _ = pagerank(grafo, tolerancia=0, max_iteracoes=15)
    obtido, iteracoes = rodar_spark(spark, grafo, tolerancia=0, max_iteracoes=15)
    assert iteracoes == 15
    assert erro_l1(obtido, esperado) < 1e-12


def test_spark_com_pagina_sem_saida_e_parada_por_tolerancia(spark):
    grafo = grafo_exemplo_4_paginas()
    esperado, _ = pagerank(grafo, tolerancia=1e-12)
    obtido, _ = rodar_spark(spark, grafo, tolerancia=1e-12)
    assert erro_l1(obtido, esperado) < 1e-9
