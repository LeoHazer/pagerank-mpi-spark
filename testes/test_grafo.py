import numpy as np

from src.comum.grafo import (
    carregar_grafo,
    criar_grafo,
    gerar_grafo_aleatorio,
    grafo_exemplo_3_paginas,
    salvar_grafo,
)


# Garante as duas regras de limpeza do PageRank: link de uma página para ela mesma é
# ignorado e link repetido conta uma vez só. Sem isso, a nota seria dividida errado.
def test_remove_autolinks_e_repetidos():
    # 0->0 (auto-link) e 0->1 repetido devem ser descartados
    grafo = criar_grafo(3, [0, 0, 0, 1], [0, 1, 1, 2])
    pares = set(zip(grafo.origens.tolist(), grafo.destinos.tolist()))
    assert pares == {(0, 1), (1, 2)}


# Grau de saída = quantos links saem de cada página. No exemplo dos slides a página 0
# aponta para 1 e 2, e as páginas 1 e 2 apontam para uma só. Cada página divide a sua
# nota por esse número, então ele precisa estar certo.
def test_grau_saida_do_exemplo():
    grafo = grafo_exemplo_3_paginas()
    assert grafo.grau_saida().tolist() == [2, 1, 1]


# Com a mesma semente, o gerador precisa criar exatamente o mesmo grafo. É isso que
# permite rodar sequencial, MPI e Spark sobre o mesmo grafo e repetir as medições.
def test_gerador_e_reprodutivel():
    g1 = gerar_grafo_aleatorio(1000, 5, semente=7)
    g2 = gerar_grafo_aleatorio(1000, 5, semente=7)
    assert np.array_equal(g1.origens, g2.origens)
    assert np.array_equal(g1.destinos, g2.destinos)


# Todo link precisa sair de uma página que existe e chegar numa que existe (0 a N-1), e
# nenhum pode ligar uma página a ela mesma. Um número fora disso daria erro no cálculo.
def test_gerador_respeita_limites():
    grafo = gerar_grafo_aleatorio(500, 8, semente=1)
    assert grafo.origens.min() >= 0 and grafo.destinos.min() >= 0
    assert grafo.origens.max() < 500 and grafo.destinos.max() < 500
    assert not np.any(grafo.origens == grafo.destinos)


# Salva o grafo num arquivo .npz e lê de volta: o lido tem que ser idêntico ao original.
# Os programas MPI e Spark recebem o grafo por arquivo, então esse caminho precisa ser fiel.
# tmp_path é uma pasta temporária que o pytest cria só para este teste (fora do projeto).
def test_salvar_e_carregar(tmp_path):
    original = gerar_grafo_aleatorio(300, 4, semente=3)
    caminho = str(tmp_path / "g.npz")
    salvar_grafo(original, caminho)
    lido = carregar_grafo(caminho)
    assert lido.n_vertices == original.n_vertices
    assert np.array_equal(lido.origens, original.origens)
    assert np.array_equal(lido.destinos, original.destinos)