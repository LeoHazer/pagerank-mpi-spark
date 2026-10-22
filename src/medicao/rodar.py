"""
Roda uma versão do PageRank várias vezes, com quantidades diferentes de
processos, e anota os tempos numa planilha CSV.

Cada caso é repetido algumas vezes (o artigo de referência usa 3) porque o
tempo varia um pouco de uma execução para outra; depois usamos a média.
Todas as execuções usam um número fixo de iterações (tolerância 0) para
que o trabalho feito seja sempre o mesmo. O padrão é 10: cada iteração
leva sempre o mesmo tempo, então o speedup não depende de quantas rodamos
(a correção do resultado é conferida pelos testes, que rodam até convergir).

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
    # O comando é uma lista em que cada item é uma "palavra" digitada no terminal.
    # Ex.: ["--grafo", "dados/g.npz"] é o mesmo que escrever  --grafo dados/g.npz
    # Como cada item já é um argumento separado, espaços num caminho não atrapalham.
    # Estas opções valem para as três versões do PageRank.
    opcoes = [
        "--grafo", args.grafo,
        # tolerância 0 nunca é alcançada (a variação nunca fica menor que zero), então
        # o programa roda sempre exatamente --max-iteracoes iterações
        "--tolerancia", "0",
        "--max-iteracoes", str(args.iteracoes),
        # mostra só a 1ª página do ranking, para não encher a saída
        "--top", "1",
        # o próprio programa do PageRank anota o tempo dele nesta planilha
        "--csv", args.csv,
        "--variante", args.variante,
        # shlex.split quebra um texto em argumentos respeitando as aspas, como o terminal
        # faz. Ex.: '--memoria 8g' vira ["--memoria", "8g"]
        # o * "despeja" os itens dessa lista dentro da lista opcoes
        *shlex.split(args.extra),
    ]

    # sys.executable é o caminho do Python que está rodando este script (o do venv),
    # para o programa filho usar as mesmas bibliotecas instaladas
    if args.versao == "sequencial":
        return [sys.executable, "-m", "src.sequencial.pagerank", *opcoes]

    # com --alteracao, roda o arquivo das versões alteradas em vez do original
    modulo = f"src.{args.versao}.pagerank"
    if args.alteracao:
        modulo += "_alterado"
        opcoes += ["--alteracao", args.alteracao]

    # MPI: o mpirun abre "-n" cópias do programa ao mesmo tempo, uma por processo.
    # Ex.: mpirun -n 4 python -m mpi4py -m src.mpi.pagerank --grafo ...
    # O mpi_extra entra logo depois do -n porque são opções do próprio mpirun (ex.:
    # --oversubscribe, que deixa abrir mais processos do que o computador tem de núcleos).
    # "python -m mpi4py -m modulo" roda o módulo pelo mpi4py: se um processo der erro,
    # ele derruba todos, em vez de deixar os outros travados esperando para sempre.
    if args.versao == "mpi":
        return [
            "mpirun", "-n", str(processos), *shlex.split(args.mpi_extra),
            sys.executable, "-m", "mpi4py", "-m", modulo, *opcoes,
        ]

    # Spark: não há mpirun; o próprio Spark divide o trabalho entre núcleos.
    # No modo local (tudo nesta máquina) o número de núcleos vai dentro do master,
    # ex.: local[4] = usar 4 núcleos.
    if args.master.startswith("local"):
        return [sys.executable, "-m", modulo, "--master", f"local[{processos}]", *opcoes]
    # Num cluster Spark de verdade (spark://ip:7077) o master é o endereço do cluster,
    # e o limite de núcleos vai à parte, em --nucleos (vira spark.cores.max)
    return [sys.executable, "-m", modulo, "--master", args.master,
            "--nucleos", str(processos), *opcoes]


def main() -> None:
    parser = argparse.ArgumentParser(description="Mede o PageRank com 1, 2, 4, ... processos")
    parser.add_argument("--versao", choices=["sequencial", "mpi", "spark"], required=True)
    parser.add_argument("--grafo", type=str, required=True, help="arquivo .npz do gerar_grafo")
    # nargs="+" aceita uma ou mais quantidades seguidas: --processos 1 2 4 8
    parser.add_argument("--processos", type=int, nargs="+", default=[1, 2, 4],
                        help="lista de quantidades de processos, ex.: 1 2 4 8")
    parser.add_argument("--repeticoes", type=int, default=3, help="quantas vezes rodar cada caso")
    # 10 iterações fixas bastam: todas levam o mesmo tempo, então o speedup sai igual
    # com 10 ou com 100, e a medição fica bem mais curta
    parser.add_argument("--iteracoes", type=int, default=10, help="iterações fixas do PageRank")
    parser.add_argument("--csv", type=str, default="resultados/medicoes.csv")
    parser.add_argument("--alteracao", choices=["1", "2", "3", "otimizada"], default=None,
                        help="mede uma versão alterada; sem esta opção mede a original")
    parser.add_argument("--variante", type=str, default=None,
                        help="rótulo na planilha; o padrão é 'base' ou o nome da alteração")
    parser.add_argument("--extra", type=str, default="",
                        help="opções a mais repassadas ao programa do PageRank")
    # cuidado: um valor que começa com "--" e não tem espaço (ex.: --oversubscribe) é
    # confundido com uma opção deste script; escreva junto com "=": --mpi-extra=--oversubscribe
    parser.add_argument("--mpi-extra", type=str, default="",
                        help='opções a mais do mpirun, ex.: "--hostfile hosts --oversubscribe"')
    parser.add_argument("--master", type=str, default="local",
                        help="Spark: 'local' (esta máquina) ou spark://ip:7077 (cluster)")
    parser.add_argument("--limite", type=int, default=1800,
                        help="segundos máximos por execução; acima disso o caso é abandonado")
    args = parser.parse_args()
    if args.alteracao and args.versao == "sequencial":
        parser.error("as alterações existem só para mpi e spark")
    # rótulo automático da planilha: "base" (original), "alteracao1", ..., "otimizada"
    if args.variante is None:
        args.variante = "otimizada" if args.alteracao == "otimizada" else (
            f"alteracao{args.alteracao}" if args.alteracao else "base")

    # a versão sequencial só existe com 1 processo
    lista_de_processos = [1] if args.versao == "sequencial" else args.processos

    # para cada quantidade de processos (1, 2, 4, ...), repete a execução algumas vezes
    for processos in lista_de_processos:
        for rodada in range(1, args.repeticoes + 1):
            # end="" deixa o "ok" sair na mesma linha; flush=True mostra o texto na hora
            print(f"[{args.versao}/{args.variante}] {processos} processo(s), "
                  f"rodada {rodada}/{args.repeticoes} ... ", end="", flush=True)
            # este relógio é só para o log; o tempo que vai para a planilha é medido
            # dentro do próprio programa do PageRank (só a parte do cálculo)
            inicio = time.perf_counter()
            # subprocess.run executa o comando como outro programa e espera ele terminar:
            #   check=True: se o programa terminar com erro, levanta CalledProcessError
            #   capture_output=True: guarda o que ele escreveria na tela (não polui o log)
            #   text=True: devolve essa saída como texto, e não como bytes
            #   timeout: passou desse tempo, o programa é encerrado e vem TimeoutExpired
            try:
                subprocess.run(montar_comando(args, processos), check=True,
                               capture_output=True, text=True, timeout=args.limite)
            # se uma rodada estourou o tempo ou deu erro, as próximas do mesmo caso
            # fariam o mesmo; o break pula para a próxima quantidade de processos,
            # para não gastar o tempo de cluster à toa
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
