#!/bin/bash
# Prepara o ambiente no nó do cluster. Rodar uma vez, de dentro da pasta do projeto:
#     bash cluster/preparar.sh
#
# Tudo fica dentro de ~/grupo_pagerank, porque a pasta pessoal do cluster é
# compartilhada com outros grupos.

# set -e: para o script no primeiro comando que der erro, em vez de seguir adiante
# com o ambiente pela metade
set -e

# venv = uma pasta com um Python e as bibliotecas só deste projeto, sem mexer no do sistema.
# ${PASTA_VENV:-...} usa a variável PASTA_VENV se alguém a definiu antes; senão, o caminho padrão
PASTA_VENV="${PASTA_VENV:-$HOME/grupo_pagerank/venv}"
# a conta do cluster aponta SPARK_HOME para outra versão do Spark (instalada pelo
# professor); sem isto o nosso pyspark tentaria usá-la e as versões não se entendem
unset SPARK_HOME

echo "== processador (para o relatório)"
# lscpu descreve o processador; o grep filtra só modelo, total de CPUs, threads por
# núcleo, núcleos por soquete e soquetes. free -h mostra a memória RAM
lscpu | grep -E "Model name|^CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)"
free -h | head -2

echo "== ambiente Python em $PASTA_VENV"
# só cria o venv se a pasta ainda não existir (rodar o script de novo não refaz tudo)
if [ ! -d "$PASTA_VENV" ]; then
  # alguns sistemas não trazem o pip junto do venv; nesse caso instalamos o pip à parte
  if ! python3 -m venv "$PASTA_VENV" 2>/dev/null; then
    rm -rf "$PASTA_VENV"
    python3 -m venv --without-pip "$PASTA_VENV"
    curl -sS https://bootstrap.pypa.io/get-pip.py | "$PASTA_VENV/bin/python"
  fi
fi
# source "ativa" o venv neste terminal: daqui em diante, python e pip são os do venv
source "$PASTA_VENV/bin/activate"
# instala as bibliotecas listadas em requirements.txt (-q = quieto, com menos mensagens)
pip install -q --upgrade pip
pip install -q -r requirements.txt
# confere se as três bibliotecas principais carregam e anota as versões para o relatório
python -c "import numpy, mpi4py, pyspark; print('numpy', numpy.__version__, '| mpi4py', mpi4py.__version__, '| pyspark', pyspark.__version__)"

echo "== grafos"
mkdir -p dados
# só gera se ainda não existir (gerar de novo daria o mesmo grafo, por causa da semente).
# "[ -f arquivo ] || comando" = se o arquivo não existe, roda o comando.
# Dois grafos: 100 mil páginas (testes rápidos) e 1 milhão (medições), ~10 links por página
[ -f dados/grafo_100k.npz ] || python -m src.comum.gerar_grafo --vertices 100000 --grau-medio 10 --saida dados/grafo_100k.npz
[ -f dados/grafo_1m.npz ] || python -m src.comum.gerar_grafo --vertices 1000000 --grau-medio 10 --saida dados/grafo_1m.npz

echo "== testes"
# roda os testes automáticos: confirma que as três versões dão o resultado certo
# neste nó antes de gastar tempo medindo
python -m pytest -q

echo "Pronto. Para usar o ambiente depois: source $PASTA_VENV/bin/activate"
