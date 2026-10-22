#!/bin/bash
# Mede quanto tempo cada versão leva neste nó, com 1 e com 32 processos, para
# planejar as medições oficiais (que são bem mais longas). Rodar de dentro da
# pasta do projeto, depois do preparar.sh:
#     nohup bash cluster/calibrar.sh > calibracao.log 2>&1 &
#     tail -f calibracao.log
# nohup deixa o script rodando mesmo se a conexão SSH cair ou o terminal fechar;
# "> calibracao.log 2>&1" manda tudo o que seria mostrado na tela (inclusive os erros)
# para o arquivo; o "&" no fim solta o terminal. tail -f acompanha o arquivo ao vivo.
#
# Os tempos vão para resultados/calibracao.csv, separados das medições oficiais.

# ativa o ambiente Python criado pelo preparar.sh (python passa a ser o do venv)
PASTA_VENV="${PASTA_VENV:-$HOME/grupo_pagerank/venv}"
source "$PASTA_VENV/bin/activate"
# a conta do cluster aponta SPARK_HOME para outra versão do Spark (instalada pelo
# professor); sem isto o nosso pyspark tentaria usá-la e as versões não se entendem
unset SPARK_HOME

CSV=resultados/calibracao.csv
GRAFO=dados/grafo_1m.npz
# cada execução que passar disso é abandonada (o rodador avisa no log)
LIMITE=1500
# o nó tem 31 GB; o padrão de 4 GB do Spark fica apertado com muitos núcleos.
# Este texto é repassado ao programa do Spark pela opção --extra do rodador
MEMORIA_SPARK="--memoria 8g"

# atalho para não repetir as opções comuns: "$@" acrescenta o que vier depois de
# "medir" na chamada. Ex.: medir --versao sequencial. Na calibração basta 1 rodada por caso
medir() {
  python -m src.medicao.rodar --grafo "$GRAFO" --repeticoes 1 --limite "$LIMITE" --csv "$CSV" "$@"
}

# data e uptime vão para o log: uptime mostra a carga do nó (se outro grupo estiver
# usando a máquina, os tempos saem piores)
echo "== início: $(date)"
uptime
medir --versao sequencial
# só 1 e 32 processos: os dois extremos bastam para estimar quanto o plano todo vai levar.
# --oversubscribe deixa rodar mais processos que núcleos físicos (o "=" é necessário
# para o rodador não confundir o valor com uma opção dele)
medir --versao mpi --processos 1 32 --mpi-extra=--oversubscribe
medir --versao mpi --alteracao otimizada --processos 1 32 --mpi-extra=--oversubscribe
medir --versao spark --processos 1 32 --extra "$MEMORIA_SPARK"
medir --versao spark --alteracao otimizada --processos 1 32 --extra "$MEMORIA_SPARK"
# a versão RDD é a mais lenta: só com 32 núcleos, para saber se cabe no plano
medir --versao spark --alteracao 3 --processos 32 --extra "$MEMORIA_SPARK"

# tabela com tempo médio, speedup e eficiência da calibração
python -m src.medicao.analisar --csv "$CSV" --saida resultados/resumo_calibracao.csv
echo "== fim: $(date)"
