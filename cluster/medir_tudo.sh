#!/bin/bash
# Medições oficiais no nó do cluster (n-11), em cerca de 50 minutos. Rodar de
# dentro da pasta do projeto:
#     nohup bash cluster/medir_tudo.sh > medicao.log 2>&1 &
#     tail -f medicao.log
# nohup deixa o script rodando mesmo se a conexão SSH cair ou o terminal fechar;
# "> medicao.log 2>&1" manda tudo o que seria mostrado na tela (inclusive os erros)
# para o arquivo; o "&" no fim solta o terminal. tail -f acompanha o arquivo ao vivo.
#
# A ordem vai do mais rápido e mais importante para o mais lento. Se o tempo
# acabar, dá para interromper (veja o README) e tudo o que já terminou
# continua anotado na planilha.

# ativa o ambiente Python criado pelo preparar.sh (python passa a ser o do venv)
PASTA_VENV="${PASTA_VENV:-$HOME/grupo_pagerank/venv}"
source "$PASTA_VENV/bin/activate"
# a conta do cluster aponta SPARK_HOME para outra versão do Spark (instalada pelo
# professor); sem isto o nosso pyspark tentaria usá-la e as versões não se entendem
unset SPARK_HOME

CSV=resultados/medicoes_cluster.csv
GRAFO=dados/grafo_1m.npz
# cada iteração leva sempre o mesmo tempo, então o speedup não depende de quantas
# rodamos; 10 basta para medir (a correção do resultado é garantida pelos testes)
ITERACOES=10
# cada caso é repetido para tirar a média; no Spark são menos rodadas para o plano
# todo caber no tempo previsto
REPETICOES_MPI=3
REPETICOES_SPARK=2
# o enunciado pede 1, 2, 4, ... 128; o nó tem 16 núcleos físicos (32 com hyper-threading).
# Hyper-threading: cada núcleo físico finge ser dois. Com 64 e 128 processos há mais
# processos que núcleos, e eles se revezam (por isso o --oversubscribe mais abaixo)
TODOS="1 2 4 8 16 32 64 128"
# no Spark, acima de 32 só há mais tarefas disputando os mesmos núcleos
ATE_32="1 2 4 8 16 32"
# uma execução que passar disso é abandonada (o rodador avisa no log)
LIMITE=900
# --oversubscribe deixa o mpirun abrir mais processos que núcleos; o "=" junta tudo numa
# palavra só, senão o rodador acharia que --oversubscribe é uma opção dele (e daria erro)
MPI_EXTRA="--mpi-extra=--oversubscribe"

# atalho para não repetir as opções comuns: "$@" acrescenta o que vier depois de
# "medir" na chamada. Ex.: medir --versao sequencial vira o comando completo
medir() {
  python -m src.medicao.rodar --grafo "$GRAFO" --iteracoes "$ITERACOES" --limite "$LIMITE" \
    --csv "$CSV" "$@"
}

# o mesmo para o Spark, já com 8 GB de memória: o nó tem 31 GB e o padrão de 4 GB fica
# apertado com muitos núcleos (o --extra repassa a opção ao programa do Spark)
medir_spark() {
  medir --versao spark --extra "--memoria 8g" "$@"
}

# data e uptime vão para o log: uptime mostra a carga do nó (se outro grupo estiver
# usando a máquina, os tempos saem piores)
echo "== início: $(date)"
uptime

# primeiro a sequencial: é a base de comparação de todos os gráficos
echo "== 1. sequencial (referência)"
medir --versao sequencial --repeticoes "$REPETICOES_MPI"

# depois o MPI, que é rápido: todas as variantes, de 1 a 128 processos.
# $TODOS vai sem aspas de propósito: assim o bash separa "1 2 4 ..." em vários
# argumentos, um por quantidade, como o --processos espera
echo "== 2. MPI: original, alterações e otimizada, de 1 a 128 processos"
medir --versao mpi --processos $TODOS --repeticoes "$REPETICOES_MPI" $MPI_EXTRA
for alteracao in 1 2 3 otimizada; do
  medir --versao mpi --alteracao $alteracao --processos $TODOS --repeticoes "$REPETICOES_MPI" $MPI_EXTRA
done

# Spark por último, do mais importante (original e otimizada, curva completa) para o
# mais lento (alterações só nos extremos 1 e 32, e a RDD só com 32)
echo "== 3. Spark original: de 1 a 32 núcleos, e 64 uma vez para registrar que não compensa"
medir_spark --processos $ATE_32 --repeticoes "$REPETICOES_SPARK"
medir_spark --processos 64 --repeticoes 1

echo "== 4. Spark otimizada: de 1 a 32 núcleos"
medir_spark --alteracao otimizada --processos $ATE_32 --repeticoes "$REPETICOES_SPARK"

echo "== 5. Spark alterações 1 e 2: com 1 e 32 núcleos"
for alteracao in 1 2; do
  medir_spark --alteracao $alteracao --processos 1 32 --repeticoes "$REPETICOES_SPARK"
done

echo "== 6. Spark alteração 3 (RDD): só com 32 núcleos, porque é muito lenta"
medir_spark --alteracao 3 --processos 32 --repeticoes 1

# com tudo medido, gera a tabela de speedup/eficiência e as figuras do relatório
echo "== resumo e gráficos"
python -m src.medicao.analisar --csv "$CSV" --saida resultados/resumo_cluster.csv
python -m src.medicao.graficos --csv "$CSV" --pasta resultados/graficos
echo "== fim: $(date)"
