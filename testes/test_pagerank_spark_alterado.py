"""
Testes das versões alteradas do Spark. Uma alteração só vale se continuar
dando o mesmo resultado da versão sequencial.
"""

import shutil

import pytest

from src.comum.grafo import gerar_grafo_aleatorio, grafo_exemplo_4_paginas
from src.comum.metricas import erro_l1
from src.sequencial.pagerank import pagerank

# sem pyspark ou sem Java (ex.: Windows) os testes são pulados, como em test_pagerank_spark.py
pytest.importorskip("pyspark")
pytestmark = pytest.mark.skipif(shutil.which("java") is None, reason="Java não encontrado")

# imports depois do importorskip de propósito ("noqa: E402" cala o aviso de estilo)
from src.spark.pagerank import criar_sessao  # noqa: E402
from src.spark.pagerank_alterado import (  # noqa: E402
    pagerank_rdd,
    pagerank_tabelas,
    para_numpy,
    preparar_rdds,
    preparar_tabelas,
    rdd_para_numpy,
)

# as quatro versões alteradas (explicadas em src/spark/pagerank_alterado.py)
ALTERACOES = ["1", "2", "3", "otimizada"]
# em quantos pedaços (partições) cada tabela é dividida entre os executores
PARTICOES = 4


# Mesma fixture de test_pagerank_spark.py: abrir o Spark demora, então scope="module"
# abre uma sessão só, usada por todos os testes deste arquivo, e fecha no fim.
@pytest.fixture(scope="module")
def spark():
    sessao = criar_sessao("local[2]")
    yield sessao
    sessao.stop()


def rodar_alterado(spark, alteracao, grafo, **opcoes):
    """Faz o mesmo que o main do pagerank_alterado, sem precisar abrir outro Spark."""
    # a alteração 3 usa RDD (outro formato de dados do Spark), por isso tem funções próprias
    if alteracao == "3":
        tabelas = preparar_rdds(spark, grafo, PARTICOES)
        ranks, iteracoes = pagerank_rdd(*tabelas, grafo.n_vertices, PARTICOES, **opcoes)
        return rdd_para_numpy(ranks), iteracoes

    # as outras usam tabelas (DataFrames) e ligam as opções de cada alteração: 2 e otimizada
    # dividem os links por destino e usam broadcast; 1 e otimizada somam a nota das páginas
    # sem saída junto com o cálculo da tabela nova (observar_massa)
    por_destino = alteracao in ("2", "otimizada")
    tabelas = preparar_tabelas(spark, grafo, PARTICOES, por_destino)
    ranks, iteracoes = pagerank_tabelas(*tabelas, grafo.n_vertices, **opcoes,
                                        observar_massa=alteracao in ("1", "otimizada"),
                                        broadcast=por_destino)
    return para_numpy(ranks), iteracoes


# parametrize roda este teste uma vez para cada alteração. Com tolerancia=0 e 12 iterações
# fixas, Spark e sequencial fazem as mesmas rodadas, então o resultado tem que bater quase
# até a última casa (erro_l1 abaixo de 1e-12). Uma alteração pode mudar o tempo, não o resultado.
@pytest.mark.parametrize("alteracao", ALTERACOES)
def test_alteracao_igual_ao_sequencial_com_iteracoes_fixas(spark, alteracao):
    grafo = gerar_grafo_aleatorio(2000, 6, semente=11)
    esperado, _ = pagerank(grafo, tolerancia=0, max_iteracoes=12)
    obtido, iteracoes = rodar_alterado(spark, alteracao, grafo, tolerancia=0, max_iteracoes=12)
    assert iteracoes == 12
    assert erro_l1(obtido, esperado) < 1e-12


# Em cada alteração: página sem saída (exemplo de 4 páginas) e parada por tolerância. Como
# cada versão decide sozinha quando parar, a margem aceita é maior (1e-8) que no teste acima.
@pytest.mark.parametrize("alteracao", ALTERACOES)
def test_alteracao_com_pagina_sem_saida_e_parada_por_tolerancia(spark, alteracao):
    grafo = grafo_exemplo_4_paginas()
    esperado, _ = pagerank(grafo, tolerancia=1e-10)
    obtido, _ = rodar_alterado(spark, alteracao, grafo, tolerancia=1e-10)
    assert erro_l1(obtido, esperado) < 1e-8
