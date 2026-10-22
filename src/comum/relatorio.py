"""Funções para mostrar os resultados no terminal de um jeito fácil de ler."""

import numpy as np


def porcentagem(valor, casas=4):
    """0.01601534 -> '1,6015%' (vírgula como separador decimal, padrão brasileiro)."""
    # o ".{casas}f" dentro da f-string diz "escreva com 'casas' casas decimais";
    # o Python usa ponto como separador decimal, então trocamos por vírgula
    return f"{valor * 100:.{casas}f}".replace(".", ",") + "%"


def decimal(valor, casas=2):
    """6.5291 -> '6,53' (vírgula como separador decimal)."""
    return f"{valor:.{casas}f}".replace(".", ",")


def numero(valor):
    """1601.5 -> '1.602' e 1.25 -> '1,2' (abaixo de 10 mostra uma casa decimal)."""
    # números pequenos perderiam informação sem a casa decimal (1,2 viraria 1)
    if valor < 10:
        return f"{valor:.1f}".replace(".", ",")
    # o "," do formato manda separar milhares e o ".0f" tira as casas decimais
    # (arredondando); o Python separa milhares com vírgula, então trocamos pelo
    # ponto brasileiro. Ex.: 1601.5 -> '1,602' -> '1.602'
    return f"{valor:,.0f}".replace(",", ".")


def linhas_do_top(rank, quantidade=5):
    """Monta as linhas do ranking das páginas mais importantes."""
    n_vertices = len(rank)
    # argsort não devolve as notas ordenadas, e sim os NÚMEROS DAS PÁGINAS em
    # ordem da menor nota para a maior. Ex.: rank [0,2; 0,5; 0,3] -> [0, 2, 1].
    # [::-1] inverte (maior primeiro: [1, 2, 0]) e [:quantidade] pega só os primeiros
    melhores = np.argsort(rank)[::-1][:quantidade]

    linhas = [f"Top {len(melhores)} páginas (chance de o navegador aleatório estar em cada uma):"]
    # enumerate(..., start=1) entrega a colocação (1, 2, 3...) junto com a página
    for posicao, pagina in enumerate(melhores, start=1):
        # se todas fossem iguais, cada uma teria 1/N; rank * N diz quantas vezes ela supera isso
        vezes_a_media = rank[pagina] * n_vertices
        # "<8" alinha o número da página à esquerda em 8 espaços e ">9" alinha a
        # porcentagem à direita em 9 espaços, para as colunas ficarem retas
        linhas.append(
            f"  {posicao}. página {pagina:<8} {porcentagem(rank[pagina]):>9}"
            f"   ({numero(vezes_a_media)} vezes a média)"
        )
    return linhas
