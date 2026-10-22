"""
Testes das versões alteradas do Spark. Uma alteração só vale se continuar
dando o mesmo resultado da versão sequencial.
"""

import shutil

import pytest

from src.comum.grafo import gerar_grafo_aleatorio, grafo_exemplo_4_paginas
from src.comum.metricas import erro_l1
from src.sequencial.pagerank import pagerank

pytest.importorskip("pyspark")
pytestmark = pytest.mark.skipif(shutil.which("java") is None, reason="Java não encontrado")

from src.spark.pagerank import criar_sessao  # noqa: E402
from src.spark.pagerank_alterado import (  # noqa: E402
    pagerank_rdd,
    pagerank_tabelas,
    para_numpy,
    preparar_rdds,
    preparar_tabelas,
    rdd_para_numpy,
)

ALTERACOES = ["1", "2", "3", "otimizada"]
PARTICOES = 4


@pytest.fixture(scope="module")
def spark():
    sessao = criar_sessao("local[2]")
    yield sessao
    sessao.stop()


def rodar_alterado(spark, alteracao, grafo, **opcoes):
    """Faz o mesmo que o main do pagerank_alterado, sem precisar abrir outro Spark."""
    if alteracao == "3":
        tabelas = preparar_rdds(spark, grafo, PARTICOES)
        ranks, iteracoes = pagerank_rdd(*tabelas, grafo.n_vertices, PARTICOES, **opcoes)
        return rdd_para_numpy(ranks), iteracoes

    por_destino = alteracao in ("2", "otimizada")
    tabelas = preparar_tabelas(spark, grafo, PARTICOES, por_destino)
    ranks, iteracoes = pagerank_tabelas(*tabelas, grafo.n_vertices, **opcoes,
                                        observar_massa=alteracao in ("1", "otimizada"),
                                        broadcast=por_destino)
    return para_numpy(ranks), iteracoes


@pytest.mark.parametrize("alteracao", ALTERACOES)
def test_alteracao_igual_ao_sequencial_com_iteracoes_fixas(spark, alteracao):
    grafo = gerar_grafo_aleatorio(2000, 6, semente=11)
    esperado, _ = pagerank(grafo, tolerancia=0, max_iteracoes=12)
    obtido, iteracoes = rodar_alterado(spark, alteracao, grafo, tolerancia=0, max_iteracoes=12)
    assert iteracoes == 12
    assert erro_l1(obtido, esperado) < 1e-12


@pytest.mark.parametrize("alteracao", ALTERACOES)
def test_alteracao_com_pagina_sem_saida_e_parada_por_tolerancia(spark, alteracao):
    grafo = grafo_exemplo_4_paginas()
    esperado, _ = pagerank(grafo, tolerancia=1e-10)
    obtido, _ = rodar_alterado(spark, alteracao, grafo, tolerancia=1e-10)
    assert erro_l1(obtido, esperado) < 1e-8
