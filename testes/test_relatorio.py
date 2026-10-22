import numpy as np

from src.comum.relatorio import linhas_do_top, numero, porcentagem


def test_porcentagem_usa_virgula():
    assert porcentagem(0.01601534) == "1,6015%"
    assert porcentagem(1.0) == "100,0000%"


def test_numero_com_separador_brasileiro():
    assert numero(1601.5) == "1.602"
    assert numero(1.25) == "1,2"


def test_top_vem_do_maior_para_o_menor():
    linhas = linhas_do_top(np.array([0.2, 0.5, 0.3]), quantidade=2)
    assert len(linhas) == 3  # título + 2 páginas
    assert "página 1" in linhas[1] and "50,0000%" in linhas[1] and "1,5 vezes" in linhas[1]
    assert "página 2" in linhas[2]


def test_top_maior_que_o_grafo_mostra_todas():
    linhas = linhas_do_top(np.array([0.5, 0.5]), quantidade=5)
    assert linhas[0].startswith("Top 2 páginas")
