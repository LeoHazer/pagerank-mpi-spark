"""
Testes da versão MPI. Cada teste chama o mpirun de verdade e compara o
resultado com a versão sequencial.
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

# no Windows sem MPI instalado os testes são pulados em vez de falhar
pytest.importorskip("mpi4py")
pytestmark = pytest.mark.skipif(shutil.which("mpirun") is None, reason="mpirun não encontrado")

RAIZ_DO_PROJETO = Path(__file__).resolve().parent.parent


def rodar_mpi(processos, grafo, pasta):
    caminho_grafo = str(pasta / "grafo.npz")
    caminho_ranks = str(pasta / "ranks.npy")
    salvar_grafo(grafo, caminho_grafo)

    comando = [
        "mpirun", "-n", str(processos),
        # deixa rodar mais processos do que núcleos disponíveis
        "--oversubscribe",
        # "-m mpi4py" derruba todos os processos se um deles der erro
        sys.executable, "-m", "mpi4py", "-m", "src.mpi.pagerank",
        "--grafo", caminho_grafo,
        "--tolerancia", "1e-13",
        "--salvar-ranks", caminho_ranks,
    ]
    # timeout: se algo travar, o teste falha em vez de ficar parado
    subprocess.run(comando, cwd=RAIZ_DO_PROJETO, check=True, capture_output=True, text=True,
                   timeout=120)
    return np.load(caminho_ranks)


@pytest.mark.parametrize("processos", [1, 2, 3, 4])
def test_mpi_igual_ao_sequencial(tmp_path, processos):
    grafo = gerar_grafo_aleatorio(3000, 8, semente=11)
    esperado, _ = pagerank(grafo, tolerancia=1e-13)
    assert erro_l1(rodar_mpi(processos, grafo, tmp_path), esperado) < 1e-9


def test_mpi_com_mais_processos_que_arestas(tmp_path):
    # 6 arestas para 8 processos: dois deles ficam sem nenhuma aresta
    grafo = grafo_exemplo_4_paginas()
    esperado, _ = pagerank(grafo, tolerancia=1e-13)
    assert erro_l1(rodar_mpi(8, grafo, tmp_path), esperado) < 1e-9
