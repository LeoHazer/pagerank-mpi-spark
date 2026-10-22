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

# sem pyspark ou sem Java (ex.: Windows) os testes são pulados em vez de falhar;
# o Spark roda em Java, por isso o skipif procura o programa java
pytest.importorskip("pyspark")
pytestmark = pytest.mark.skipif(shutil.which("java") is None, reason="Java não encontrado")

# este import só pode vir depois do importorskip (sem pyspark ele daria erro);
# o "noqa: E402" avisa o verificador de estilo que o import fora do topo é de propósito
from src.spark.pagerank import criar_sessao, pagerank_spark, para_numpy, preparar_tabelas  # noqa: E402


# Fixture: algo que o pytest prepara e entrega a cada teste que tem um parâmetro com o
# mesmo nome (aqui, "spark"). Com scope="module" a sessão é aberta uma vez só para o
# arquivo inteiro. O que vem antes do yield roda no começo; o que vem depois, no fim.
@pytest.fixture(scope="module")
def spark():
    # local[2]: Spark rodando nesta máquina, com 2 núcleos trabalhando em paralelo
    sessao = criar_sessao("local[2]")
    yield sessao
    sessao.stop()


# Faz o mesmo que o main do Spark, reaproveitando a sessão já aberta: monta as tabelas
# divididas em 4 partições, roda o PageRank e traz o resultado como vetor do NumPy.
def rodar_spark(spark, grafo, **opcoes):
    tabelas = preparar_tabelas(spark, grafo, particoes=4)
    ranks, iteracoes = pagerank_spark(*tabelas, grafo.n_vertices, **opcoes)
    return para_numpy(ranks), iteracoes


# Com tolerancia=0 e 15 iterações fixas, Spark e sequencial fazem exatamente as mesmas 15
# rodadas. Por isso a comparação pode ser bem rígida (erro_l1 abaixo de 1e-12): sobra só a
# diferença da ordem das somas, que mexe nas últimas casas decimais. É o modo das medições.
def test_spark_igual_ao_sequencial_com_iteracoes_fixas(spark):
    grafo = gerar_grafo_aleatorio(2000, 6, semente=11)
    esperado, _ = pagerank(grafo, tolerancia=0, max_iteracoes=15)
    obtido, iteracoes = rodar_spark(spark, grafo, tolerancia=0, max_iteracoes=15)
    # confere que o Spark não parou antes da hora
    assert iteracoes == 15
    assert erro_l1(obtido, esperado) < 1e-12


# Cobre o que o teste anterior não passa: a página sem saída (exemplo de 4 páginas) e a
# parada por tolerância, em que o Spark mede quanto o vetor mudou a cada rodada. Como cada
# versão decide sozinha quando parar, a margem aceita é maior (1e-9) que no teste anterior.
def test_spark_com_pagina_sem_saida_e_parada_por_tolerancia(spark):
    grafo = grafo_exemplo_4_paginas()
    esperado, _ = pagerank(grafo, tolerancia=1e-12)
    obtido, _ = rodar_spark(spark, grafo, tolerancia=1e-12)
    assert erro_l1(obtido, esperado) < 1e-9
