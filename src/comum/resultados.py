"""
Guarda o resultado de cada execução numa planilha CSV (uma linha por execução).

CSV é um arquivo de texto em que cada linha é uma linha da planilha e as
colunas são separadas por vírgula. Abre direto no Excel ou no pandas.
"""

import csv
import os
import socket
from datetime import datetime

COLUNAS = [
    "data_hora",
    "maquina",
    "versao",               # sequencial, mpi ou spark
    "variante",             # nome da alteração de código testada ("base" = sem alteração)
    "processos",            # processos MPI ou núcleos do Spark
    "particoes",            # pedaços das tabelas no Spark (vazio nas outras versões)
    "vertices",
    "arestas",
    "iteracoes",
    "tempo_leitura",        # ler o arquivo do grafo
    "tempo_distribuicao",   # repartir o grafo entre os processos
    "tempo_calculo",        # só as iterações do PageRank: é o tempo usado no speedup
    "tempo_total",          # do início do programa até o fim do cálculo
]


def registrar(caminho, **valores):
    """Acrescenta uma linha ao CSV; cria o arquivo (com o cabeçalho) se ele ainda não existir."""
    pasta = os.path.dirname(caminho)
    if pasta:
        os.makedirs(pasta, exist_ok=True)
    arquivo_novo = not os.path.exists(caminho)

    linha = {
        "data_hora": datetime.now().isoformat(timespec="seconds"),
        "maquina": socket.gethostname(),
        **valores,
    }
    # "a" abre o arquivo para acrescentar no final, sem apagar o que já existe
    with open(caminho, "a", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=COLUNAS)
        if arquivo_novo:
            escritor.writeheader()
        escritor.writerow(linha)
