#!/bin/bash
# Medições oficiais no nó do cluster (n-11). Rodar de dentro da pasta do projeto:
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
ITERACOES=25
REPETICOES=3
# o enunciado pede 1, 2, 4, ... 128; acima de 32 os processos dividem núcleos
TODOS="1 2 4 8 16 32 64 128"
# lista menor para as alterações do Spark, que são lentas
ALGUNS="1 4 16 32"
# uma execução que passar disso é abandonada (o rodador avisa no log)
LIMITE=2400
MPI_EXTRA="--mpi-extra=--oversubscribe"

medir() {
  python -m src.medicao.rodar --grafo "$GRAFO" --iteracoes "$ITERACOES" --limite "$LIMITE" \
    --csv "$CSV" "$@"
}

echo "== início: $(date)"
uptime

echo "== 1. sequencial (referência)"
medir --versao sequencial --repeticoes "$REPETICOES"

echo "== 2. MPI: original, alterações e otimizada, lista completa"
medir --versao mpi --processos $TODOS --repeticoes "$REPETICOES" $MPI_EXTRA
for alteracao in 1 2 3 otimizada; do
  medir --versao mpi --alteracao $alteracao --processos $TODOS --repeticoes "$REPETICOES" $MPI_EXTRA
done

echo "== 3. Spark: original e otimizada, lista completa"
medir --versao spark --processos $TODOS --repeticoes "$REPETICOES" --extra "--memoria 8g"
medir --versao spark --alteracao otimizada --processos $TODOS --repeticoes "$REPETICOES" --extra "--memoria 8g"

echo "== 4. Spark: alterações 1 e 2, lista menor"
for alteracao in 1 2; do
  medir --versao spark --alteracao $alteracao --processos $ALGUNS --repeticoes "$REPETICOES" --extra "--memoria 8g"
done

echo "== 5. Spark: alteração 3 (RDD), só alguns pontos e uma vez, porque é muito lenta"
medir --versao spark --alteracao 3 --processos 4 16 32 --repeticoes 1 --extra "--memoria 8g"

echo "== resumo e gráficos"
python -m src.medicao.analisar --csv "$CSV" --saida resultados/resumo_cluster.csv
python -m src.medicao.graficos --csv "$CSV" --pasta resultados/graficos
echo "== fim: $(date)"
