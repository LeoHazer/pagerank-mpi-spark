"""
Roda uma versão do PageRank várias vezes, com quantidades diferentes de
processos, e anota os tempos numa planilha CSV.

Cada caso é repetido algumas vezes (o artigo de referência usa 3) porque o
tempo varia um pouco de uma execução para outra; depois usamos a média.
Todas as execuções usam um número fixo de iterações (tolerância 0) para
que o trabalho feito seja sempre o mesmo.

Exemplos (a partir da raiz do projeto):
    python -m src.medicao.rodar --versao mpi --grafo dados/grafo_1m.npz --processos 1 2 4
    python -m src.medicao.rodar --versao spark --grafo dados/grafo_100k.npz --processos 1 2 4

No cluster, os argumentos a mais do mpirun vão em --mpi-extra:
    python -m src.medicao.rodar --versao mpi --grafo dados/grafo_1m.npz \
        --processos 1 2 4 8 16 32 64 128 --mpi-extra "--hostfile hosts --oversubscribe"
"""

import argparse
import shlex
import subprocess
import sys
import time


def montar_comando(args, processos):
    """Monta a linha de comando de uma execução, como se fosse digitada no terminal."""
    opcoes = [
        "--grafo", args.grafo,
        "--tolerancia", "0",
        "--max-iteracoes", str(args.iteracoes),
        "--top", "1",
        "--csv", args.csv,
        "--variante", args.variante,
        # shlex.split quebra um texto em argumentos respeitando as aspas
        *shlex.split(args.extra),
    ]

    if args.versao == "sequencial":
        return [sys.executable, "-m", "src.sequencial.pagerank", *opcoes]

    # com --alteracao, roda o arquivo das versões alteradas em vez do original
    modulo = f"src.{args.versao}.pagerank"
    if args.alteracao:
        modulo += "_alterado"
        opcoes += ["--alteracao", args.alteracao]

    if args.versao == "mpi":
        return [
            "mpirun", "-n", str(processos), *shlex.split(args.mpi_extra),
            sys.executable, "-m", "mpi4py", "-m", modulo, *opcoes,
        ]

    # Spark: no modo local o número de núcleos vai dentro do próprio master, ex.: local[4]
    if args.master.startswith("local"):
        return [sys.executable, "-m", modulo, "--master", f"local[{processos}]", *opcoes]
    return [sys.executable, "-m", modulo, "--master", args.master,
            "--nucleos", str(processos), *opcoes]


def main() -> None:
    parser = argparse.ArgumentParser(description="Mede o PageRank com 1, 2, 4, ... processos")
    parser.add_argument("--versao", choices=["sequencial", "mpi", "spark"], required=True)
    parser.add_argument("--grafo", type=str, required=True, help="arquivo .npz do gerar_grafo")
    parser.add_argument("--processos", type=int, nargs="+", default=[1, 2, 4],
                        help="lista de quantidades de processos, ex.: 1 2 4 8")
    parser.add_argument("--repeticoes", type=int, default=3, help="quantas vezes rodar cada caso")
    parser.add_argument("--iteracoes", type=int, default=50, help="iterações fixas do PageRank")
    parser.add_argument("--csv", type=str, default="resultados/medicoes.csv")
    parser.add_argument("--alteracao", choices=["1", "2", "3", "otimizada"], default=None,
                        help="mede uma versão alterada; sem esta opção mede a original")
    parser.add_argument("--variante", type=str, default=None,
                        help="rótulo na planilha; o padrão é 'base' ou o nome da alteração")
    parser.add_argument("--extra", type=str, default="",
                        help="opções a mais repassadas ao programa do PageRank")
    parser.add_argument("--mpi-extra", type=str, default="",
                        help='opções a mais do mpirun, ex.: "--hostfile hosts --oversubscribe"')
    parser.add_argument("--master", type=str, default="local",
                        help="Spark: 'local' (esta máquina) ou spark://ip:7077 (cluster)")
    parser.add_argument("--limite", type=int, default=1800,
                        help="segundos máximos por execução; acima disso o caso é abandonado")
    args = parser.parse_args()
    if args.alteracao and args.versao == "sequencial":
        parser.error("as alterações existem só para mpi e spark")
    if args.variante is None:
        args.variante = "otimizada" if args.alteracao == "otimizada" else (
            f"alteracao{args.alteracao}" if args.alteracao else "base")

    # a versão sequencial só existe com 1 processo
    lista_de_processos = [1] if args.versao == "sequencial" else args.processos

    for processos in lista_de_processos:
        for rodada in range(1, args.repeticoes + 1):
            print(f"[{args.versao}/{args.variante}] {processos} processo(s), "
                  f"rodada {rodada}/{args.repeticoes} ... ", end="", flush=True)
            inicio = time.perf_counter()
            try:
                subprocess.run(montar_comando(args, processos), check=True,
                               capture_output=True, text=True, timeout=args.limite)
            except subprocess.TimeoutExpired:
                print(f"passou de {args.limite} s; pulando as outras rodadas deste caso")
                break
            except subprocess.CalledProcessError as erro:
                # mostra só o final da mensagem de erro, que é onde fica a causa
                print("falhou:\n" + erro.stderr[-2000:])
                break
            print(f"ok ({time.perf_counter() - inicio:.1f} s)")

    print(f"Tempos anotados em {args.csv}")


if __name__ == "__main__":
    main()
