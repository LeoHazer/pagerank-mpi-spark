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

# "Agg" gera os arquivos de imagem sem precisar abrir nenhuma janela. É o que funciona
# no cluster, que não tem tela; precisa vir antes do import do pyplot, por isso os
# imports abaixo ficam depois desta linha (o "noqa: E402" cala o aviso do verificador)
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
# marcador diferente garante que a identificação não dependa só da cor (também ajuda
# quem imprime o relatório em preto e branco). Cor fixa por variante = a "alteração 1"
# tem a mesma cara em todas as figuras. Marcadores: o bola, s quadrado, ^ triângulo,
# D losango, v triângulo para baixo
ESTILOS = {
    "base": ("#2a78d6", "o", "original"),
    "alteracao1": ("#eb6834", "s", "alteração 1"),
    "alteracao2": ("#1baf7a", "^", "alteração 2"),
    "alteracao3": ("#eda100", "D", "alteração 3"),
    "otimizada": ("#e87ba4", "v", "otimizada"),
}
# variantes com outro nome usam estas cores, sempre na mesma ordem
ESTILOS_EXTRAS = [("#008300", "P"), ("#6250d6", "X"), ("#e34948", "*")]

# cores da "moldura" (fundo, textos, grade e eixos), em tons neutros para os dados se destacarem
FUNDO = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
CINZA = "#898781"
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"


def estilo_da(variante, extras_usados):
    if variante in ESTILOS:
        return ESTILOS[variante]
    # variante desconhecida: ganha o próximo estilo extra livre; a lista extras_usados
    # lembra quem já pegou qual, e o % volta ao começo se as cores extras acabarem
    if variante not in extras_usados:
        extras_usados.append(variante)
    cor, marcador = ESTILOS_EXTRAS[extras_usados.index(variante) % len(ESTILOS_EXTRAS)]
    return cor, marcador, variante


def arrumar_eixos(eixo, processos, titulo, rotulo_x, rotulo_y):
    eixo.set_facecolor(FUNDO)
    eixo.set_title(titulo, loc="left", fontsize=11, color=TINTA)
    eixo.set_xlabel(rotulo_x, color=TINTA_SECUNDARIA)
    eixo.set_ylabel(rotulo_y, color=TINTA_SECUNDARIA)
    # eixo x em escala de potências de 2: 1, 2, 4, 8... ficam igualmente espaçados.
    # Numa escala comum, de 1 a 128, os valores 1, 2, 4 e 8 ficariam espremidos no canto.
    eixo.set_xscale("log", base=2)
    # marcas do eixo x só nos valores medidos, sem marquinhas intermediárias, escritas
    # como número normal ("8", e não "2³")
    eixo.xaxis.set_major_locator(FixedLocator(processos))
    eixo.xaxis.set_minor_locator(NullLocator())
    eixo.xaxis.set_major_formatter(FuncFormatter(lambda valor, _: f"{valor:g}"))
    # grade clara só na horizontal, para não competir com as linhas de dados
    eixo.grid(axis="y", color=GRADE, linewidth=0.8)
    eixo.set_axisbelow(True)
    # tira as bordas de cima e da direita, que não carregam informação
    for lado in ("top", "right"):
        eixo.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        eixo.spines[lado].set_color(EIXO)
    eixo.tick_params(colors=TINTA_SECUNDARIA, labelsize=9)


def desenhar_caso(resumo_do_caso, caminho, tempo_sequencial=None):
    """
    Uma figura com os três painéis para um caso (versão + grafo).
    tempo_sequencial, se existir, vira uma linha de referência no painel de tempo.
    """
    versao = resumo_do_caso["versao"].iloc[0]
    vertices = resumo_do_caso["vertices"].iloc[0]
    arestas = resumo_do_caso["arestas"].iloc[0]
    iteracoes = resumo_do_caso["iteracoes"].iloc[0]
    processos = sorted(resumo_do_caso["processos"].unique())

    # uma figura com 1 linha e 3 colunas de painéis; figsize é o tamanho em polegadas
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

    # referências: speedup ideal (y = p) e eficiência de 100%, em cinza tracejado.
    # Ideal = com p processos fica p vezes mais rápido (ex.: 8 processos, speedup 8);
    # quanto mais perto dessa linha a curva medida, melhor a paralelização
    speedup.plot(processos, processos, linestyle="--", color=CINZA, linewidth=1.2, label="ideal")
    eficiencia.axhline(1.0, linestyle="--", color=CINZA, linewidth=1.2)
    referencias = []
    if tempo_sequencial is not None:
        # tempo da versão sequencial (sem dividir nada): abaixo da linha = mais rápido que ela
        referencias.append(tempo.axhline(tempo_sequencial, linestyle=":", color=TINTA_SECUNDARIA,
                                         linewidth=1.5, label="sequencial"))

    extras_usados = []
    # desenha as variantes na ordem de ESTILOS (original primeiro); as desconhecidas, no fim
    variantes = sorted(resumo_do_caso["variante"].unique(),
                       key=lambda v: list(ESTILOS).index(v) if v in ESTILOS else len(ESTILOS))
    for variante in variantes:
        linhas = resumo_do_caso[resumo_do_caso["variante"] == variante].sort_values("processos")
        cor, marcador, nome = estilo_da(variante, extras_usados)
        # variante medida só em alguns valores (com buracos na lista): só os pontos, sem
        # linha, para não sugerir que os valores intermediários foram medidos.
        # Compara os valores dela com todos os do gráfico até o maior que ela tem.
        # Ex.: gráfico com 1, 2, 4, ..., 32 e variante só com 1 e 32 -> tem buracos
        tem_buracos = list(linhas["processos"]) != [p for p in processos if p <= linhas["processos"].max()]
        # as mesmas opções de desenho servem para os três painéis
        traco = dict(color=cor, marker=marcador, markersize=7, linewidth=2,
                     linestyle="none" if tem_buracos else "-",
                     markeredgecolor=FUNDO, markeredgewidth=1, label=nome)

        # errorbar desenha o ponto com uma barrinha vertical de +/- um desvio padrão.
        # desvio vazio (uma rodada só) vira zero: a barrinha simplesmente não aparece
        tempo.errorbar(linhas["processos"], linhas["tempo_medio"], yerr=linhas["desvio"].fillna(0),
                       capsize=3, elinewidth=1, **traco)
        speedup.plot(linhas["processos"], linhas["speedup"], **traco)
        eficiencia.plot(linhas["processos"], linhas["eficiencia"], **traco)

    # tempos muito diferentes entre variantes (ex.: RDD) ficam legíveis em escala logarítmica.
    # Na escala comum, um tempo 20 vezes maior achataria os outros no rodapé; na
    # logarítmica, cada degrau multiplica (10, 100, 1000), e todos aparecem bem.
    # Só liga quando o maior tempo passa de 5 vezes o menor (contando a sequencial).
    medias = resumo_do_caso["tempo_medio"]
    if tempo_sequencial is not None:
        medias = pd.concat([medias, pd.Series([tempo_sequencial])])
    if medias.max() / medias.min() > 5:
        tempo.set_yscale("log")
        tempo.set_ylabel("segundos (escala logarítmica)", color=TINTA_SECUNDARIA)
        # números em 1, 2 e 5 de cada potência de 10 (ex.: 20, 50, 100, 200, 500)
        tempo.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        tempo.yaxis.set_minor_formatter(NullFormatter())
        # escreve o número com vírgula decimal, como no português (0,5 em vez de 0.5)
        tempo.yaxis.set_major_formatter(FuncFormatter(lambda valor, _: f"{valor:g}".replace(".", ",")))
    else:
        # escala comum começando do zero, para a altura das curvas ser proporcional ao tempo
        tempo.set_ylim(bottom=0)
        tempo.yaxis.set_major_formatter(FuncFormatter(lambda valor, _: decimal(valor, 1)))
    # a altura acompanha o maior speedup medido; a linha do ideal pode sair por cima,
    # senão ela (que chega a 128) achataria as curvas reais no rodapé
    speedup.set_ylim(0, max(2.0, resumo_do_caso["speedup"].max() * 1.4))
    speedup.yaxis.set_major_formatter(FuncFormatter(lambda valor, _: decimal(valor, 1)))
    eficiencia.set_ylim(0, max(1.1, resumo_do_caso["eficiencia"].max() * 1.1))
    eficiencia.yaxis.set_major_formatter(FuncFormatter(lambda valor, _: f"{valor * 100:.0f}%"))

    # uma legenda só, embaixo dos três painéis. Os nomes vêm do painel de speedup (que já
    # tem as variantes e o "ideal"), mais a linha da sequencial, que só existe no de tempo
    marcas, nomes = speedup.get_legend_handles_labels()
    marcas += referencias
    nomes += [linha.get_label() for linha in referencias]
    figura.legend(marcas, nomes, loc="lower center", ncol=len(nomes), frameon=False,
                  fontsize=10, labelcolor=TINTA)
    # tight_layout ajeita os painéis sem sobreposição, deixando 8% embaixo para a legenda
    # e 6% em cima para o título; dpi=200 dá uma imagem nítida para o relatório
    figura.tight_layout(rect=(0, 0.08, 1, 0.94))
    figura.savefig(caminho, dpi=200, facecolor=FUNDO)
    # fecha a figura para liberar memória (o laço em gerar() cria várias)
    plt.close(figura)


def gerar(medicoes, pasta):
    """Gera uma figura por caso e devolve a lista de arquivos criados."""
    os.makedirs(pasta, exist_ok=True)
    resumo = resumir(medicoes)
    # a sequencial não ganha figura própria (seria um ponto só): vira referência nas outras
    sequencial = resumo[resumo["versao"] == "sequencial"]
    paralelas = resumo[resumo["versao"] != "sequencial"]

    arquivos = []
    # o caso aqui não separa por variante: todas as variantes vão no mesmo gráfico
    for (versao, vertices), linhas in paralelas.groupby(["versao", "vertices"]):
        # procura a sequencial com o mesmo grafo e o mesmo número de iterações, senão a
        # comparação seria injusta; se não houver, a figura sai sem essa linha
        mesmo_grafo = sequencial[(sequencial["vertices"] == vertices)
                                 & (sequencial["iteracoes"] == linhas["iteracoes"].iloc[0])]
        tempo_sequencial = mesmo_grafo["tempo_medio"].iloc[0] if len(mesmo_grafo) else None
        caminho = os.path.join(pasta, f"{versao}_{vertices}_paginas.png")
        desenhar_caso(linhas, caminho, tempo_sequencial)
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
