import numpy as np

from src.comum.relatorio import linhas_do_top, numero, porcentagem


# O relatório mostra os números no padrão brasileiro, com vírgula decimal. Aqui a chance
# vira porcentagem com 4 casas: 0,01601534 x 100 = 1,6015%.
def test_porcentagem_usa_virgula():
    assert porcentagem(0.01601534) == "1,6015%"
    assert porcentagem(1.0) == "100,0000%"


# De 10 para cima o número sai inteiro, com ponto separando os milhares (1601,5 -> 1.602);
# abaixo de 10 sai com uma casa decimal. Em 1,25 -> "1,2" o Python arredonda o 5 para o
# algarismo par mais próximo, e não sempre para cima.
def test_numero_com_separador_brasileiro():
    assert numero(1601.5) == "1.602"
    assert numero(1.25) == "1,2"


# O ranking precisa vir da maior nota para a menor. Com notas [0,2; 0,5; 0,3] a primeira é
# a página 1 (50%) e a segunda, a página 2. O "1,5 vezes" vem de 0,5 x 3 páginas: a página 1
# tem 1,5 vezes a nota que teria se todas fossem iguais (1/3 cada).
def test_top_vem_do_maior_para_o_menor():
    linhas = linhas_do_top(np.array([0.2, 0.5, 0.3]), quantidade=2)
    assert len(linhas) == 3  # título + 2 páginas
    assert "página 1" in linhas[1] and "50,0000%" in linhas[1] and "1,5 vezes" in linhas[1]
    assert "página 2" in linhas[2]


# Pedir um top 5 num grafo de só 2 páginas não pode dar erro: mostra as 2 que existem e o
# título diz "Top 2".
def test_top_maior_que_o_grafo_mostra_todas():
    linhas = linhas_do_top(np.array([0.5, 0.5]), quantidade=5)
    assert linhas[0].startswith("Top 2 páginas")
