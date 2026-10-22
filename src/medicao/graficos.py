"""
Gera os gráficos do relatório a partir da planilha de medições.

Para cada versão (mpi, spark) e cada tamanho de grafo sai uma figura com
três painéis lado a lado, todos em função do número de processos:
  - tempo médio de cálculo (com barrinhas mostrando o desvio)
  - speedup, com a linha do speedup ideal (p processos = p vezes mais rápido)
  - eficiência, com a referência de 100%

Cada variante (original, alteração 1, ...) tem sempre a mesma cor e o mesmo
marcador em todas as figuras, para dar para comparar uma figura com a outra.

Uso (a partir da raiz do projeto):
    python -m src.medicao.graficos --csv resultados/medicoes.csv --pasta resultados/graficos
"""

import argparse
import os

import matplotlib

# "Agg" gera os arquivos de imagem sem precisar abrir nenhuma janela
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import (  # noqa: E402
    FixedLocator,
    FuncFormatter,
    LogLocator,
    NullFormatter,
    NullLocator,
)

from src.comum.relatorio import decimal, numero  # noqa: E402
from src.medicao.analisar import resumir  # noqa: E402

# cor, marcador e nome de cada variante; cores testadas para daltonismo, e o
# marcador diferente garante que a identificação não dependa só da cor
ESTILOS = {
    "base": ("#2a78d6", "o", "original"),
    "alteracao1": ("#eb6834", "s", "alteração 1"),
    "alteracao2": ("#1baf7a", "^", "alteração 2"),
    "alteracao3": ("#eda100", "D", "alteração 3"),
    "otimizada": ("#e87ba4", "v", "otimizada"),
}
# variantes com outro nome usam estas cores, sempre na mesma ordem
ESTILOS_EXTRAS = [("#008300", "P"), ("#6250d6", "X"), ("#e34948", "*")]

FUNDO = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
CINZA = "#898781"
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"


def estilo_da(variante, extras_usados):
    if variante in ESTILOS:
        return ESTILOS[variante]
    if variante not in extras_usados:
        extras_usados.append(variante)
    cor, marcador = ESTILOS_EXTRAS[extras_usados.index(variante) % len(ESTILOS_EXTRAS)]
    return cor, marcador, variante


def arrumar_eixos(eixo, processos, titulo, rotulo_x, rotulo_y):
    eixo.set_facecolor(FUNDO)
    eixo.set_title(titulo, loc="left", fontsize=11, color=TINTA)
    eixo.set_xlabel(rotulo_x, color=TINTA_SECUNDARIA)
    eixo.set_ylabel(rotulo_y, color=TINTA_SECUNDARIA)
    # eixo x em escala de potências de 2: 1, 2, 4, 8... ficam igualmente espaçados
    eixo.set_xscale("log", base=2)
    eixo.xaxis.set_major_locator(FixedLocator(processos))
    eixo.xaxis.set_minor_locator(NullLocator())
    eixo.xaxis.set_major_formatter(FuncFormatter(lambda valor, _: f"{valor:g}"))
    # grade clara só na horizontal, para não competir com as linhas de dados
    eixo.grid(axis="y", color=GRADE, linewidth=0.8)
    eixo.set_axisbelow(True)
    for lado in ("top", "right"):
        eixo.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        eixo.spines[lado].set_color(EIXO)
    eixo.tick_params(colors=TINTA_SECUNDARIA, labelsize=9)


def desenhar_caso(resumo_do_caso, caminho):
    """Uma figura com os três painéis para um caso (versão + grafo)."""
    versao = resumo_do_caso["versao"].iloc[0]
    vertices = resumo_do_caso["vertices"].iloc[0]
    arestas = resumo_do_caso["arestas"].iloc[0]
    iteracoes = resumo_do_caso["iteracoes"].iloc[0]
    processos = sorted(resumo_do_caso["processos"].unique())

    figura, (tempo, speedup, eficiencia) = plt.subplots(1, 3, figsize=(15, 4.6))
    figura.patch.set_facecolor(FUNDO)
    figura.suptitle(
        f"{versao.upper()}: {numero(vertices)} páginas, {numero(arestas)} links, "
        f"{iteracoes} iterações",
        x=0.01, ha="left", fontsize=13, color=TINTA,
    )

    # no Spark o que variamos é o número de núcleos; no MPI, o de processos
    unidade = "núcleos" if versao == "spark" else "processos"
    arrumar_eixos(tempo, processos, "Tempo médio de cálculo", unidade, "segundos")
    arrumar_eixos(speedup, processos, "Speedup", unidade, "vezes mais rápido que com 1")
    arrumar_eixos(eficiencia, processos, "Eficiência", unidade, f"speedup ÷ {unidade}")

    # referências: speedup ideal (y = p) e eficiência de 100%, em cinza tracejado
    speedup.plot(processos, processos, linestyle="--", color=CINZA, linewidth=1.2, label="ideal")
    eficiencia.axhline(1.0, linestyle="--", color=CINZA, linewidth=1.2)

    extras_usados = []
    variantes = sorted(resumo_do_caso["variante"].unique(),
                       key=lambda v: list(ESTILOS).index(v) if v in ESTILOS else len(ESTILOS))
    for variante in variantes:
        linhas = resumo_do_caso[resumo_do_caso["variante"] == variante].sort_values("processos")
        cor, marcador, nome = estilo_da(variante, extras_usados)
        traco = dict(color=cor, marker=marcador, markersize=7, linewidth=2,
                     markeredgecolor=FUNDO, markeredgewidth=1, label=nome)

        # desvio vazio (uma rodada só) vira zero: a barrinha simplesmente não aparece
        tempo.errorbar(linhas["processos"], linhas["tempo_medio"], yerr=linhas["desvio"].fillna(0),
                       capsize=3, elinewidth=1, **traco)
        speedup.plot(linhas["processos"], linhas["speedup"], **traco)
        eficiencia.plot(linhas["processos"], linhas["eficiencia"], **traco)

    # tempos muito diferentes entre variantes (ex.: RDD) ficam legíveis em escala logarítmica
    medias = resumo_do_caso["tempo_medio"]
    if medias.max() / medias.min() > 5:
        tempo.set_yscale("log")
        tempo.set_ylabel("segundos (escala logarítmica)", color=TINTA_SECUNDARIA)
        # números em 1, 2 e 5 de cada potência de 10 (ex.: 20, 50, 100, 200, 500)
        tempo.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        tempo.yaxis.set_minor_formatter(NullFormatter())
        tempo.yaxis.set_major_formatter(FuncFormatter(lambda valor, _: f"{valor:g}".replace(".", ",")))
    else:
        tempo.set_ylim(bottom=0)
        tempo.yaxis.set_major_formatter(FuncFormatter(lambda valor, _: decimal(valor, 1)))
    speedup.set_ylim(bottom=0)
    speedup.yaxis.set_major_formatter(FuncFormatter(lambda valor, _: decimal(valor, 1)))
    eficiencia.set_ylim(0, max(1.1, resumo_do_caso["eficiencia"].max() * 1.1))
    eficiencia.yaxis.set_major_formatter(FuncFormatter(lambda valor, _: f"{valor * 100:.0f}%"))

    # uma legenda só, embaixo dos três painéis
    marcas, nomes = speedup.get_legend_handles_labels()
    figura.legend(marcas, nomes, loc="lower center", ncol=len(nomes), frameon=False,
                  fontsize=10, labelcolor=TINTA)
    figura.tight_layout(rect=(0, 0.08, 1, 0.94))
    figura.savefig(caminho, dpi=200, facecolor=FUNDO)
    plt.close(figura)


def gerar(medicoes, pasta):
    """Gera uma figura por caso e devolve a lista de arquivos criados."""
    os.makedirs(pasta, exist_ok=True)
    resumo = resumir(medicoes)
    arquivos = []
    # o caso aqui não separa por variante: todas as variantes vão no mesmo gráfico
    for (versao, vertices), linhas in resumo.groupby(["versao", "vertices"]):
        caminho = os.path.join(pasta, f"{versao}_{vertices}_paginas.png")
        desenhar_caso(linhas, caminho)
        arquivos.append(caminho)
    return arquivos


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera os gráficos de tempo, speedup e eficiência")
    parser.add_argument("--csv", type=str, default="resultados/medicoes.csv")
    parser.add_argument("--pasta", type=str, default="resultados/graficos")
    args = parser.parse_args()

    for caminho in gerar(pd.read_csv(args.csv), args.pasta):
        print(f"Gráfico salvo em {caminho}")


if __name__ == "__main__":
    main()
