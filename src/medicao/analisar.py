"""
Lê a planilha de medições e calcula, para cada caso:
  - tempo médio de cálculo e desvio padrão (o quanto as rodadas variaram)
  - speedup    = tempo com 1 processo / tempo com p processos
  - eficiência = speedup / p   (100% = cada processo rende o máximo possível)

Uso (a partir da raiz do projeto):
    python -m src.medicao.analisar --csv resultados/medicoes.csv
"""

import argparse

import pandas as pd

from src.comum.relatorio import decimal, numero, porcentagem

# linhas com os mesmos valores nestas colunas são o mesmo experimento
CASO = ["versao", "variante", "vertices", "arestas", "iteracoes"]


def resumir(medicoes):
    """Uma linha por (caso, processos) com média, desvio, speedup e eficiência."""
    resumo = medicoes.groupby(CASO + ["processos"], as_index=False).agg(
        rodadas=("tempo_calculo", "size"),
        tempo_medio=("tempo_calculo", "mean"),
        desvio=("tempo_calculo", "std"),
    )

    # para cada caso, pega o tempo médio com 1 processo e coloca ao lado de todas as linhas dele
    com_um = resumo[resumo["processos"] == 1].set_index(CASO)["tempo_medio"].rename("tempo_com_1")
    resumo = resumo.join(com_um, on=CASO)

    resumo["speedup"] = resumo["tempo_com_1"] / resumo["tempo_medio"]
    resumo["eficiencia"] = resumo["speedup"] / resumo["processos"]
    return resumo.drop(columns="tempo_com_1").sort_values(CASO + ["processos"]).reset_index(drop=True)


def imprimir(resumo):
    # groupby separa a tabela em um bloco por experimento
    for caso, linhas in resumo.groupby(CASO, sort=False):
        versao, variante, vertices, arestas, iteracoes = caso
        print(f"\n{versao} | variante {variante} | {numero(vertices)} vértices, "
              f"{numero(arestas)} arestas, {iteracoes} iterações")
        print(f"  {'processos':>9}  {'rodadas':>7}  {'tempo médio':>11}  {'desvio':>8}  "
              f"{'speedup':>7}  {'eficiência':>10}")
        for _, linha in linhas.iterrows():
            # desvio só existe com 2 rodadas ou mais; speedup só se houver medição com 1 processo
            desvio = "-" if pd.isna(linha.desvio) else f"{decimal(linha.desvio, 3)} s"
            speedup = "-" if pd.isna(linha.speedup) else decimal(linha.speedup, 2)
            eficiencia = "-" if pd.isna(linha.eficiencia) else porcentagem(linha.eficiencia, 1)
            print(f"  {linha.processos:>9}  {linha.rodadas:>7}  "
                  f"{decimal(linha.tempo_medio, 3) + ' s':>11}  {desvio:>8}  "
                  f"{speedup:>7}  {eficiencia:>10}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calcula speedup e eficiência das medições")
    parser.add_argument("--csv", type=str, default="resultados/medicoes.csv")
    parser.add_argument("--saida", type=str, default="resultados/resumo.csv",
                        help="onde salvar a tabela-resumo")
    args = parser.parse_args()

    resumo = resumir(pd.read_csv(args.csv))
    imprimir(resumo)
    resumo.to_csv(args.saida, index=False)
    print(f"\nResumo salvo em {args.saida}")


if __name__ == "__main__":
    main()
