# PageRank: MPI (mpi4py) vs Spark (PySpark)

Trabalho 2 de Sistemas Distribuídos (UFF). Comparação de desempenho entre
duas implementações paralelas do PageRank.

## Ambiente
WSL2 (Ubuntu 24.04), Python 3, Open MPI, Java 17.

## Como preparar
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

## Como testar
Todos os comandos são rodados a partir da raiz do projeto.

    python -m pytest -q

## Como rodar
1. Gerar um grafo sintético (fica em `dados/`, fora do git):

       python -m src.comum.gerar_grafo --vertices 100000 --grau-medio 10 --saida dados/grafo_100k.npz

2. PageRank sequencial (versão de referência):

       python -m src.sequencial.pagerank --grafo dados/grafo_100k.npz

   Com `--tolerancia 0 --max-iteracoes 10` ele roda sempre 10 iterações,
   o que deixa as medições de tempo comparáveis entre as versões.

3. PageRank com MPI (4 processos):

       mpirun -n 4 python -m mpi4py -m src.mpi.pagerank --grafo dados/grafo_100k.npz

   O `-m mpi4py` encerra todos os processos se um deles der erro, em vez
   de deixar os outros travados esperando.

4. PageRank com Spark (4 núcleos desta máquina):

       python -m src.spark.pagerank --grafo dados/grafo_100k.npz --master "local[4]"

   No cluster, troque o `--master` pelo endereço do mestre do Spark
   (ex.: `spark://<ip-do-mestre>:7077`).

## Como medir
O rodador executa uma versão várias vezes (3 por padrão), com 1, 2, 4...
processos e 10 iterações fixas, e anota cada execução em
`resultados/medicoes.csv`. Cada iteração leva sempre o mesmo tempo, então
o speedup não depende de quantas iterações rodamos; a correção do
resultado é conferida pelos testes, que rodam até convergir (o grafo de
1 milhão de páginas converge em 21 iterações, com tolerância 1e-10).

    python -m src.medicao.rodar --versao sequencial --grafo dados/grafo_1m.npz
    python -m src.medicao.rodar --versao mpi --grafo dados/grafo_1m.npz --processos 1 2 4 8
    python -m src.medicao.rodar --versao spark --grafo dados/grafo_1m.npz --processos 1 2 4 8

Para rodar mais processos MPI do que núcleos físicos, acrescente
`--mpi-extra=--oversubscribe`.

Depois, o analisador calcula tempo médio, speedup e eficiência:

    python -m src.medicao.analisar

- speedup = tempo com 1 processo ÷ tempo com p processos
- eficiência = speedup ÷ p (100% = cada processo rende o máximo possível)

E os gráficos (tempo, speedup e eficiência; uma figura por versão e
tamanho de grafo, com todas as variantes juntas):

    python -m src.medicao.graficos

As figuras ficam em `resultados/graficos/`.

## No cluster
As medições foram feitas num nó só (n-11, 32 núcleos), com o MPI dentro do
nó e o Spark em modo local. A pasta pessoal do cluster é compartilhada
entre grupos, então tudo fica em `~/grupo_pagerank`:

    mkdir -p ~/grupo_pagerank && cd ~/grupo_pagerank
    git clone -b feat/pagerank https://github.com/DiasCaio/pagerank-mpi-spark.git
    cd pagerank-mpi-spark
    bash cluster/preparar.sh

`preparar.sh` cria o ambiente Python, instala as dependências, gera os
grafos e roda os testes. `calibrar.sh` faz uma execução curta de cada
versão para estimar a duração das medições:

    nohup bash cluster/calibrar.sh > calibracao.log 2>&1 &
    tail -f calibracao.log

O `nohup ... &` mantém o script rodando mesmo se a conexão cair.

As medições oficiais (cerca de 50 minutos) ficam em `medir_tudo.sh` e são
anotadas em `resultados/medicoes_cluster.csv`; no fim, o próprio script
gera o resumo e os gráficos:

    nohup bash cluster/medir_tudo.sh > medicao.log 2>&1 &
    tail -f medicao.log

Para interromper antes do fim (o que já terminou continua na planilha):

    pkill -f cluster/medir_tudo.sh; pkill -f src.medicao.rodar

Atenção: a conta do cluster define `SPARK_HOME` apontando para o Spark
4.0.1 instalado lá, e o nosso pyspark (4.2.0) tentaria usá-lo. Os scripts
já fazem `unset SPARK_HOME`; ao rodar comandos à mão, faça o mesmo antes
(vale só para o terminal aberto; não altere o `.bashrc`, que é compartilhado).

## Alterações de desempenho
As versões originais (`src/mpi/pagerank.py` e `src/spark/pagerank.py`)
não mudam. As alterações ficam em `pagerank_alterado.py`, escolhidas com
`--alteracao`. Cada uma é medida sozinha contra a original:

| | Área | MPI | Spark |
|---|---|---|---|
| 1 | trabalho repetido em cada rodada | peso de cada link calculado uma vez só | soma das páginas sem saída anotada durante a conta principal (`observe`), sem uma ida extra ao driver |
| 2 | divisão dos dados | cada processo é dono de uma faixa de páginas de destino (Allgatherv) | links divididos por destino e tabela de notas enviada inteira (broadcast), sem reembaralhar os links |
| 3 | formato dos dados | `allreduce` com o vetor serializado no lugar do `Allreduce` | RDD de tuplas do Python no lugar de DataFrame |
| otimizada | | 1 e 2 juntas | 1 e 2 juntas |

    mpirun -n 4 python -m mpi4py -m src.mpi.pagerank_alterado --alteracao 2 --grafo dados/grafo_1m.npz
    python -m src.spark.pagerank_alterado --alteracao 2 --grafo dados/grafo_100k.npz --master "local[4]"
    python -m src.medicao.rodar --versao mpi --alteracao 2 --grafo dados/grafo_1m.npz --processos 1 2 4

As duas versões do Spark usam 4 GB de memória por padrão (o padrão do
próprio Spark, 1 GB, não comporta o grafo de 1 milhão de páginas). Para
mudar: `--memoria 8g`, ou `--extra "--memoria 8g"` no rodador.