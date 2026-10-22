"""
Testes das versões alteradas do MPI. Uma alteração só vale se continuar
dando exatamente o mesmo resultado da versão sequencial.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.comum.grafo import gerar_grafo_aleatorio, grafo_exemplo_4_paginas, salvar_grafo
from src.comum.metricas import erro_l1
from src.sequencial.pagerank import pagerank

pytest.importorskip("mpi4py")
pytestmark = pytest.mark.skipif(shutil.which("mpirun") is None, reason="mpirun não encontrado")

RAIZ_DO_PROJETO = Path(__file__).resolve().parent.parent
ALTERACOES = ["1", "2", "3", "otimizada"]


def rodar_alterado(alteracao, processos, grafo, pasta):
    caminho_grafo = str(pasta / "grafo.npz")
    caminho_ranks = str(pasta / "ranks.npy")
    salvar_grafo(grafo, caminho_grafo)
    comando = [
        "mpirun", "-n", str(processos), "--oversubscribe",
        sys.executable, "-m", "mpi4py", "-m", "src.mpi.pagerank_alterado",
        "--alteracao", alteracao,
        "--grafo", caminho_grafo,
        "--tolerancia", "1e-13",
        "--salvar-ranks", caminho_ranks,
    ]
    subprocess.run(comando, cwd=RAIZ_DO_PROJETO, check=True, capture_output=True, text=True,
                   timeout=120)
    return np.load(caminho_ranks)


@pytest.mark.parametrize("processos", [1, 3, 4])
@pytest.mark.parametrize("alteracao", ALTERACOES)
def test_alteracao_igual_ao_sequencial(tmp_path, alteracao, processos):
    grafo = gerar_grafo_aleatorio(3000, 8, semente=11)
    esperado, _ = pagerank(grafo, tolerancia=1e-13)
    assert erro_l1(rodar_alterado(alteracao, processos, grafo, tmp_path), esperado) < 1e-9


@pytest.mark.parametrize("alteracao", ALTERACOES)
def test_alteracao_com_mais_processos_que_arestas(tmp_path, alteracao):
    # 6 arestas e 4 páginas para 8 processos: vários ficam sem links e sem páginas
    grafo = grafo_exemplo_4_paginas()
    esperado, _ = pagerank(grafo, tolerancia=1e-13)
    assert erro_l1(rodar_alterado(alteracao, 8, grafo, tmp_path), esperado) < 1e-9
