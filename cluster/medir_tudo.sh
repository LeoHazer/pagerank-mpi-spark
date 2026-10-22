#!/bin/bash
# Medições oficiais no nó do cluster (n-11), em cerca de 50 minutos. Rodar de
# dentro da pasta do projeto:
#     nohup bash cluster/medir_tudo.sh > medicao.log 2>&1 &
#     tail -f medicao.log
#
# A ordem vai do mais rápido e mais importante para o mais lento. Se o tempo
# acabar, dá para interromper (veja o README) e tudo o que já terminou
# continua anotado na planilha.

PASTA_VENV="${PASTA_VENV:-$HOME/grupo_pagerank/venv}"
source "$PASTA_VENV/bin/activate"
# a conta do cluster aponta SPARK_HOME para outra versão do Spark; sem isto não funciona
unset SPARK_HOME

CSV=resultados/medicoes_cluster.csv
GRAFO=dados/grafo_1m.npz
# cada iteração leva sempre o mesmo tempo, então o speedup não depende de quantas
# rodamos; 10 basta para medir (a correção do resultado é garantida pelos testes)
ITERACOES=10
REPETICOES_MPI=3
REPETICOES_SPARK=2
# o enunciado pede 1, 2, 4, ... 128; o nó tem 16 núcleos físicos (32 com hyper-threading)
TODOS="1 2 4 8 16 32 64 128"
# no Spark, acima de 32 só há mais tarefas disputando os mesmos núcleos
ATE_32="1 2 4 8 16 32"
# uma execução que passar disso é abandonada (o rodador avisa no log)
LIMITE=900
MPI_EXTRA="--mpi-extra=--oversubscribe"

medir() {
  python -m src.medicao.rodar --grafo "$GRAFO" --iteracoes "$ITERACOES" --limite "$LIMITE" \
    --csv "$CSV" "$@"
}

medir_spark() {
  medir --versao spark --extra "--memoria 8g" "$@"
}

echo "== início: $(date)"
uptime

echo "== 1. sequencial (referência)"
medir --versao sequencial --repeticoes "$REPETICOES_MPI"

echo "== 2. MPI: original, alterações e otimizada, de 1 a 128 processos"
medir --versao mpi --processos $TODOS --repeticoes "$REPETICOES_MPI" $MPI_EXTRA
for alteracao in 1 2 3 otimizada; do
  medir --versao mpi --alteracao $alteracao --processos $TODOS --repeticoes "$REPETICOES_MPI" $MPI_EXTRA
done

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

echo "== resumo e gráficos"
python -m src.medicao.analisar --csv "$CSV" --saida resultados/resumo_cluster.csv
python -m src.medicao.graficos --csv "$CSV" --pasta resultados/graficos
echo "== fim: $(date)"
