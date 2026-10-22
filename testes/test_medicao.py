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

# pasta principal do projeto; os programas são rodados a partir dela para achar o pacote src
RAIZ_DO_PROJETO = Path(__file__).resolve().parent.parent


# lê a planilha como uma lista de dicionários (nome da coluna -> valor), tudo como texto
def ler_csv(caminho):
    with open(caminho, encoding="utf-8") as arquivo:
        return list(csv.DictReader(arquivo))


# A planilha de medições recebe uma linha por execução e o cabeçalho só pode aparecer uma
# vez, senão ele seria lido como se fosse uma medição. Também confere que a pasta "sub" é
# criada quando não existe e que as colunas saem na ordem de COLUNAS.
def test_registrar_escreve_cabecalho_uma_vez(tmp_path):
    caminho = str(tmp_path / "sub" / "medicoes.csv")
    registrar(caminho, versao="mpi", processos=1, tempo_calculo=2.5)
    registrar(caminho, versao="mpi", processos=2, tempo_calculo=1.5)
    linhas = ler_csv(caminho)
    assert len(linhas) == 2
    assert list(linhas[0].keys()) == COLUNAS
    # o CSV guarda tudo como texto, por isso "2" entre aspas
    assert linhas[1]["processos"] == "2"


# Confere as contas do speedup e da eficiência com números fáceis de fazer de cabeça:
# com 1 processo a média é (10 + 12) / 2 = 11 s; com 2 processos, speedup = 11 / 5,5 = 2 e
# eficiência = 2 / 2 = 1 (100%); com 4 processos e o mesmo tempo, eficiência = 2 / 4 = 0,5.
def test_resumo_calcula_speedup_e_eficiencia():
    caso = dict(versao="mpi", variante="base", vertices=10, arestas=20, iteracoes=5)
    medicoes = pd.DataFrame([
        {**caso, "processos": 1, "tempo_calculo": 10.0},
        {**caso, "processos": 1, "tempo_calculo": 12.0},
        {**caso, "processos": 2, "tempo_calculo": 5.5},
        {**caso, "processos": 4, "tempo_calculo": 5.5},
    ])
    resumo = resumir(medicoes).set_index("processos")
    # pytest.approx compara números com vírgula aceitando um erro mínimo de arredondamento
    assert resumo.loc[1, "tempo_medio"] == pytest.approx(11.0)
    assert resumo.loc[1, "rodadas"] == 2
    assert resumo.loc[2, "speedup"] == pytest.approx(2.0)
    assert resumo.loc[2, "eficiencia"] == pytest.approx(1.0)
    assert resumo.loc[4, "eficiencia"] == pytest.approx(0.5)


# Gera os gráficos a partir de medições inventadas e confere que sai uma figura por versão
# paralela e tamanho de grafo, com todas as variantes no mesmo gráfico. A sequencial não
# ganha figura própria (seria um ponto só): ela entra como referência nas outras.
def test_graficos_saem_um_por_versao_e_grafo(tmp_path):
    # sem a biblioteca de gráficos (matplotlib) o teste é pulado; por isso o import vem depois
    pytest.importorskip("matplotlib")
    from src.medicao.graficos import gerar

    linhas = []
    for versao, variante, processos, tempo in [
        ("mpi", "base", 1, 8.0), ("mpi", "base", 2, 5.0), ("mpi", "otimizada", 1, 6.0),
        ("mpi", "otimizada", 2, 3.0), ("spark", "base", 1, 50.0), ("spark", "alteracao3", 1, 300.0),
        ("sequencial", "base", 1, 7.0),
    ]:
        linhas.append(dict(versao=versao, variante=variante, vertices=1000, arestas=9000,
                           iteracoes=50, processos=processos, tempo_calculo=tempo))
    arquivos = gerar(pd.DataFrame(linhas), str(tmp_path / "graficos"))
    assert sorted(Path(a).name for a in arquivos) == ["mpi_1000_paginas.png", "spark_1000_paginas.png"]
    # mais de 10 KB: a imagem foi desenhada de verdade, não é um arquivo vazio
    assert all(Path(a).stat().st_size > 10_000 for a in arquivos)


# O speedup é sempre calculado em relação ao tempo com 1 processo. Sem essa medição, ele
# fica vazio (NaN, "não é um número") em vez de dar erro ou inventar um valor.
def test_resumo_sem_medicao_com_um_processo_deixa_speedup_vazio():
    caso = dict(versao="spark", variante="base", vertices=10, arestas=20, iteracoes=5)
    resumo = resumir(pd.DataFrame([{**caso, "processos": 2, "tempo_calculo": 3.0}]))
    assert resumo["speedup"].isna().all()


# Imita os argumentos que o rodar.py receberia do terminal (argparse.Namespace), com valores
# padrão; cada teste muda só o que precisa. Assim dá para testar o comando montado sem rodar nada.
def opcoes(**mudancas):
    padrao = dict(versao="mpi", grafo="g.npz", iteracoes=50, csv="r.csv", variante="base",
                  extra="", mpi_extra="", master="local", alteracao=None)
    return argparse.Namespace(**{**padrao, **mudancas})


# Com --alteracao, o rodador tem que chamar o arquivo das versões alteradas (e não o
# original) e repassar qual alteração usar; senão a medição seria da versão errada.
def test_comando_com_alteracao_usa_o_arquivo_alterado():
    comando = montar_comando(opcoes(alteracao="otimizada", variante="otimizada"), 2)
    assert "src.mpi.pagerank_alterado" in comando
    assert comando[comando.index("--alteracao") + 1] == "otimizada"
    # "in" numa lista compara itens inteiros: confere que o módulo original não foi chamado
    assert "src.mpi.pagerank" not in comando


# As opções a mais do mpirun (como --hostfile, a lista de máquinas do cluster) têm que vir
# logo depois de "mpirun -n 8", antes do python: o que vem depois do programa vai para ele,
# e não para o mpirun.
def test_comando_mpi_repassa_opcoes_do_mpirun():
    comando = montar_comando(opcoes(mpi_extra="--hostfile hosts --oversubscribe"), 8)
    assert comando[:6] == ["mpirun", "-n", "8", "--hostfile", "hosts", "--oversubscribe"]
    assert "src.mpi.pagerank" in comando and "--csv" in comando


# No Spark local o número de núcleos vai dentro do próprio endereço (local[4]); no cluster
# vai numa opção separada, --nucleos.
def test_comando_spark_local_e_cluster():
    local = montar_comando(opcoes(versao="spark"), 4)
    assert "local[4]" in local and "--nucleos" not in local
    cluster = montar_comando(opcoes(versao="spark", master="spark://10.0.0.1:7077"), 4)
    assert cluster[cluster.index("--nucleos") + 1] == "4"


# Roda o rodar.py de verdade, como um programa separado (subprocess), num grafo pequeno, e
# devolve as linhas que ele anotou na planilha. sys.executable é o mesmo Python que está
# rodando os testes; check=True faz o teste falhar se o programa der erro; timeout=300
# (5 minutos) evita que um travamento prenda os testes para sempre.
def rodar_medicao(pasta, versao, *argumentos):
    grafo = str(pasta / "g.npz")
    salvar_grafo(gerar_grafo_aleatorio(500, 4, semente=1), grafo)
    planilha = str(pasta / "medicoes.csv")
    comando = [sys.executable, "-m", "src.medicao.rodar", "--versao", versao, "--grafo", grafo,
               "--iteracoes", "5", "--csv", planilha, *argumentos]
    subprocess.run(comando, cwd=RAIZ_DO_PROJETO, check=True, capture_output=True, timeout=300)
    return ler_csv(planilha)


# Teste de ponta a ponta (do comando até a planilha): com 2 repetições, a planilha precisa
# ganhar 2 linhas, todas com as 5 iterações pedidas.
def test_rodador_sequencial_de_ponta_a_ponta(tmp_path):
    linhas = rodar_medicao(tmp_path, "sequencial", "--repeticoes", "2")
    assert len(linhas) == 2
    assert {linha["iteracoes"] for linha in linhas} == {"5"}


# O mesmo de ponta a ponta com MPI, com 1 e 2 processos; o skipif pula o teste se não
# houver mpirun (ex.: Windows). Quando um caso falha, o rodar.py só avisa e segue, sem dar
# erro, então é a planilha que mostra se as duas execuções deram certo.
@pytest.mark.skipif(shutil.which("mpirun") is None, reason="mpirun não encontrado")
def test_rodador_mpi_de_ponta_a_ponta(tmp_path):
    linhas = rodar_medicao(tmp_path, "mpi", "--processos", "1", "2", "--repeticoes", "1")
    assert [linha["processos"] for linha in linhas] == ["1", "2"]


# Com --alteracao 2 e sem --variante, o rodador anota "alteracao2" na planilha. É esse nome
# que separa as medições da versão alterada das da original nas tabelas e nos gráficos.
@pytest.mark.skipif(shutil.which("mpirun") is None, reason="mpirun não encontrado")
def test_rodador_anota_o_nome_da_alteracao(tmp_path):
    linhas = rodar_medicao(tmp_path, "mpi", "--alteracao", "2", "--processos", "2",
                           "--repeticoes", "1")
    assert [linha["variante"] for linha in linhas] == ["alteracao2"]
