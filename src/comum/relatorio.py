"""Funções para mostrar os resultados no terminal de um jeito fácil de ler."""

import numpy as np


def porcentagem(valor, casas=4):
    """0.01601534 -> '1,6015%' (vírgula como separador decimal, padrão brasileiro)."""
    return f"{valor * 100:.{casas}f}".replace(".", ",") + "%"


def decimal(valor, casas=2):
    """6.5291 -> '6,53' (vírgula como separador decimal)."""
    return f"{valor:.{casas}f}".replace(".", ",")


def numero(valor):
    """1601.5 -> '1.602' e 1.25 -> '1,2' (abaixo de 10 mostra uma casa decimal)."""
    if valor < 10:
        return f"{valor:.1f}".replace(".", ",")
    # o Python separa milhares com vírgula; trocamos pelo ponto brasileiro
    return f"{valor:,.0f}".replace(",", ".")


def linhas_do_top(rank, quantidade=5):
    """Monta as linhas do ranking das páginas mais importantes."""
    n_vertices = len(rank)
    # argsort ordena do menor para o maior; [::-1] inverte para o maior primeiro
    melhores = np.argsort(rank)[::-1][:quantidade]

    linhas = [f"Top {len(melhores)} páginas (chance de o navegador aleatório estar em cada uma):"]
    for posicao, pagina in enumerate(melhores, start=1):
        # se todas fossem iguais, cada uma teria 1/N; rank * N diz quantas vezes ela supera isso
        vezes_a_media = rank[pagina] * n_vertices
        linhas.append(
            f"  {posicao}. página {pagina:<8} {porcentagem(rank[pagina]):>9}"
            f"   ({numero(vezes_a_media)} vezes a média)"
        )
    return linhas
