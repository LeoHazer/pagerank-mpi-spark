import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.comum.grafo import gerar_grafo_aleatorio, salvar_grafo
from src.comum.resultados import COLUNAS, registrar
from src.medicao.analisar import resumir
from src.medicao.rodar import montar_comando

RAIZ_DO_PROJETO = Path(__file__).resolve().parent.parent


def ler_csv(caminho):
    with open(caminho, encoding="utf-8") as arquivo:
        return list(csv.DictReader(arquivo))


def test_registrar_escreve_cabecalho_uma_vez(tmp_path):
    caminho = str(tmp_path / "sub" / "medicoes.csv")
    registrar(caminho, versao="mpi", processos=1, tempo_calculo=2.5)
    registrar(caminho, versao="mpi", processos=2, tempo_calculo=1.5)
    linhas = ler_csv(caminho)
    assert len(linhas) == 2
    assert list(linhas[0].keys()) == COLUNAS
    assert linhas[1]["processos"] == "2"


def test_resumo_calcula_speedup_e_eficiencia():
    caso = dict(versao="mpi", variante="base", vertices=10, arestas=20, iteracoes=5)
    medicoes = pd.DataFrame([
        {**caso, "processos": 1, "tempo_calculo": 10.0},
        {**caso, "processos": 1, "tempo_calculo": 12.0},
        {**caso, "processos": 2, "tempo_calculo": 5.5},
        {**caso, "processos": 4, "tempo_calculo": 5.5},
    ])
    resumo = resumir(medicoes).set_index("processos")
    assert resumo.loc[1, "tempo_medio"] == pytest.approx(11.0)
    assert resumo.loc[1, "rodadas"] == 2
    assert resumo.loc[2, "speedup"] == pytest.approx(2.0)
    assert resumo.loc[2, "eficiencia"] == pytest.approx(1.0)
    assert resumo.loc[4, "eficiencia"] == pytest.approx(0.5)


def test_graficos_saem_um_por_versao_e_grafo(tmp_path):
    pytest.importorskip("matplotlib")
    from src.medicao.graficos import gerar

    linhas = []
    for versao, variante, processos, tempo in [
        ("mpi", "base", 1, 8.0), ("mpi", "base", 2, 5.0), ("mpi", "otimizada", 1, 6.0),
        ("mpi", "otimizada", 2, 3.0), ("spark", "base", 1, 50.0), ("spark", "alteracao3", 1, 300.0),
    ]:
        linhas.append(dict(versao=versao, variante=variante, vertices=1000, arestas=9000,
                           iteracoes=50, processos=processos, tempo_calculo=tempo))
    arquivos = gerar(pd.DataFrame(linhas), str(tmp_path / "graficos"))
    assert sorted(Path(a).name for a in arquivos) == ["mpi_1000_paginas.png", "spark_1000_paginas.png"]
    assert all(Path(a).stat().st_size > 10_000 for a in arquivos)


def test_resumo_sem_medicao_com_um_processo_deixa_speedup_vazio():
    caso = dict(versao="spark", variante="base", vertices=10, arestas=20, iteracoes=5)
    resumo = resumir(pd.DataFrame([{**caso, "processos": 2, "tempo_calculo": 3.0}]))
    assert resumo["speedup"].isna().all()


def opcoes(**mudancas):
    padrao = dict(versao="mpi", grafo="g.npz", iteracoes=50, csv="r.csv", variante="base",
                  extra="", mpi_extra="", master="local", alteracao=None)
    return argparse.Namespace(**{**padrao, **mudancas})


def test_comando_com_alteracao_usa_o_arquivo_alterado():
    comando = montar_comando(opcoes(alteracao="otimizada", variante="otimizada"), 2)
    assert "src.mpi.pagerank_alterado" in comando
    assert comando[comando.index("--alteracao") + 1] == "otimizada"
    assert "src.mpi.pagerank" not in comando


def test_comando_mpi_repassa_opcoes_do_mpirun():
    comando = montar_comando(opcoes(mpi_extra="--hostfile hosts --oversubscribe"), 8)
    assert comando[:6] == ["mpirun", "-n", "8", "--hostfile", "hosts", "--oversubscribe"]
    assert "src.mpi.pagerank" in comando and "--csv" in comando


def test_comando_spark_local_e_cluster():
    local = montar_comando(opcoes(versao="spark"), 4)
    assert "local[4]" in local and "--nucleos" not in local
    cluster = montar_comando(opcoes(versao="spark", master="spark://10.0.0.1:7077"), 4)
    assert cluster[cluster.index("--nucleos") + 1] == "4"


def rodar_medicao(pasta, versao, *argumentos):
    grafo = str(pasta / "g.npz")
    salvar_grafo(gerar_grafo_aleatorio(500, 4, semente=1), grafo)
    planilha = str(pasta / "medicoes.csv")
    comando = [sys.executable, "-m", "src.medicao.rodar", "--versao", versao, "--grafo", grafo,
               "--iteracoes", "5", "--csv", planilha, *argumentos]
    subprocess.run(comando, cwd=RAIZ_DO_PROJETO, check=True, capture_output=True, timeout=300)
    return ler_csv(planilha)


def test_rodador_sequencial_de_ponta_a_ponta(tmp_path):
    linhas = rodar_medicao(tmp_path, "sequencial", "--repeticoes", "2")
    assert len(linhas) == 2
    assert {linha["iteracoes"] for linha in linhas} == {"5"}


@pytest.mark.skipif(shutil.which("mpirun") is None, reason="mpirun não encontrado")
def test_rodador_mpi_de_ponta_a_ponta(tmp_path):
    linhas = rodar_medicao(tmp_path, "mpi", "--processos", "1", "2", "--repeticoes", "1")
    assert [linha["processos"] for linha in linhas] == ["1", "2"]


@pytest.mark.skipif(shutil.which("mpirun") is None, reason="mpirun não encontrado")
def test_rodador_anota_o_nome_da_alteracao(tmp_path):
    linhas = rodar_medicao(tmp_path, "mpi", "--alteracao", "2", "--processos", "2",
                           "--repeticoes", "1")
    assert [linha["variante"] for linha in linhas] == ["alteracao2"]
