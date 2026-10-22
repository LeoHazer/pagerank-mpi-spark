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

# sem mpi4py ou sem o programa mpirun (ex.: Windows) os testes são pulados em vez de
# falhar, como em test_pagerank_mpi.py
pytest.importorskip("mpi4py")
pytestmark = pytest.mark.skipif(shutil.which("mpirun") is None, reason="mpirun não encontrado")

RAIZ_DO_PROJETO = Path(__file__).resolve().parent.parent
# as quatro versões alteradas (explicadas em src/mpi/pagerank_alterado.py)
ALTERACOES = ["1", "2", "3", "otimizada"]


# Igual ao rodar_mpi de test_pagerank_mpi.py (mpirun de verdade, --oversubscribe,
# "-m mpi4py" e timeout), mas roda o arquivo das versões alteradas com a alteração escolhida.
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


# Dois parametrize juntos rodam todas as combinações: 4 alterações x 3 quantidades de
# processos = 12 execuções. Uma alteração só serve se mudar o tempo sem mudar o resultado,
# então cada uma tem que bater com a versão sequencial (erro_l1 abaixo de 1e-9).
@pytest.mark.parametrize("processos", [1, 3, 4])
@pytest.mark.parametrize("alteracao", ALTERACOES)
def test_alteracao_igual_ao_sequencial(tmp_path, alteracao, processos):
    grafo = gerar_grafo_aleatorio(3000, 8, semente=11)
    esperado, _ = pagerank(grafo, tolerancia=1e-13)
    assert erro_l1(rodar_alterado(alteracao, processos, grafo, tmp_path), esperado) < 1e-9


# Mesmo caso extremo do teste do MPI original, que já achou um bug real (processo sem
# links mandando inteiros no Allreduce e travando o programa). Na alteração 2 ele pesa
# ainda mais: ela divide as PÁGINAS entre os processos, e com 4 páginas e 8 processos
# alguns ficam com a faixa vazia e entram no Allgatherv sem nada para mandar.
@pytest.mark.parametrize("alteracao", ALTERACOES)
def test_alteracao_com_mais_processos_que_arestas(tmp_path, alteracao):
    # 6 arestas e 4 páginas para 8 processos: vários ficam sem links e sem páginas
    grafo = grafo_exemplo_4_paginas()
    esperado, _ = pagerank(grafo, tolerancia=1e-13)
    assert erro_l1(rodar_alterado(alteracao, 8, grafo, tmp_path), esperado) < 1e-9
