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

# no Windows sem MPI instalado os testes são pulados em vez de falhar:
# importorskip pula o arquivo inteiro se a biblioteca mpi4py não estiver instalada, e o
# pytestmark com skipif vale para todos os testes daqui e pula se não achar o programa mpirun
pytest.importorskip("mpi4py")
pytestmark = pytest.mark.skipif(shutil.which("mpirun") is None, reason="mpirun não encontrado")

# pasta principal do projeto; o mpirun roda a partir dela para encontrar o pacote src
RAIZ_DO_PROJETO = Path(__file__).resolve().parent.parent


# Roda a versão MPI de verdade, como se fosse no terminal: salva o grafo num arquivo,
# chama o mpirun com o número de processos pedido e lê o vetor de ranks que o programa salvou.
def rodar_mpi(processos, grafo, pasta):
    caminho_grafo = str(pasta / "grafo.npz")
    caminho_ranks = str(pasta / "ranks.npy")
    salvar_grafo(grafo, caminho_grafo)

    comando = [
        "mpirun", "-n", str(processos),
        # --oversubscribe deixa rodar mais processos do que núcleos disponíveis
        # (um dos testes usa 8 processos, mais do que muitas máquinas têm)
        "--oversubscribe",
        # "-m mpi4py" derruba todos os processos se um deles der erro; sem ele os outros
        # ficariam esperando para sempre e o teste travaria
        sys.executable, "-m", "mpi4py", "-m", "src.mpi.pagerank",
        "--grafo", caminho_grafo,
        # tolerância bem pequena: roda até o vetor quase não mudar, como o esperado
        "--tolerancia", "1e-13",
        "--salvar-ranks", caminho_ranks,
    ]
    # check=True: se o programa terminar com erro, o teste falha na hora;
    # capture_output guarda as mensagens do programa em vez de jogá-las na tela;
    # timeout (em segundos): se algo travar, o teste falha em vez de ficar parado
    subprocess.run(comando, cwd=RAIZ_DO_PROJETO, check=True, capture_output=True, text=True,
                   timeout=120)
    return np.load(caminho_ranks)


# parametrize roda o mesmo teste 4 vezes, com 1, 2, 3 e 4 processos. Para qualquer número
# de processos, o resultado do MPI tem que ser igual ao da versão sequencial, que por sua
# vez já foi conferida com o networkx (biblioteca usada como gabarito).
# erro_l1 = soma das diferenças absolutas; abaixo de 1e-9 significa praticamente igual.
@pytest.mark.parametrize("processos", [1, 2, 3, 4])
def test_mpi_igual_ao_sequencial(tmp_path, processos):
    grafo = gerar_grafo_aleatorio(3000, 8, semente=11)
    esperado, _ = pagerank(grafo, tolerancia=1e-13)
    assert erro_l1(rodar_mpi(processos, grafo, tmp_path), esperado) < 1e-9


# Caso extremo que já achou um bug de verdade: processos que ficam sem nenhum link. O
# np.bincount de um processo sem links devolvia números inteiros em vez de float (número
# com vírgula); no Allreduce todos precisam mandar o mesmo tipo de dado, e o programa
# travava. A correção (forçar float64) está em somar_contribuicoes, na versão sequencial.
def test_mpi_com_mais_processos_que_arestas(tmp_path):
    # 6 arestas para 8 processos: dois deles ficam sem nenhuma aresta
    grafo = grafo_exemplo_4_paginas()
    esperado, _ = pagerank(grafo, tolerancia=1e-13)
    assert erro_l1(rodar_mpi(8, grafo, tmp_path), esperado) < 1e-9
