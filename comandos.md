# Comandos do projeto, com exemplos

Todos os comandos rodam **a partir da pasta do projeto** (`pagerank-mpi-spark/`),
num terminal Linux (WSL ou cluster), com o ambiente Python ativado.

Os exemplos de teste gravam em `/tmp` de propósito, para não misturar com os
resultados oficiais do cluster que estão em `resultados/`.

## 1. Preparar o terminal

Toda vez que abrir um terminal novo:

```bash
cd ~/caminho/para/pagerank-mpi-spark
source ~/.venvs/pagerank/bin/activate      # no cluster: source ~/grupo_pagerank/venv/bin/activate
```

No cluster, rode também `unset SPARK_HOME`. A conta do cluster aponta essa
variável para o Spark 4.0.1 instalado lá, e o nosso pyspark é o 4.2.0. O
`unset` vale só para o terminal aberto; não altere o `.bashrc`, que é
compartilhado com outros grupos.

Instalar as dependências (uma vez só):

```bash
pip install -r requirements.txt
```

## 2. Testes

```bash
# todos os testes (60 no Linux; leva uns 3 minutos por causa do Spark)
python -m pytest -q

# só um arquivo
python -m pytest -q testes/test_pagerank_mpi.py

# só os testes cujo nome contém uma palavra
python -m pytest -q -k sequencial
```

No Windows, os testes de MPI e Spark são pulados (não há MPI nem Java).

## 3. Gerar grafos

```bash
# o grafo usado nas medições: 1 milhão de páginas, 10 links por página em média
python -m src.comum.gerar_grafo --vertices 1000000 --grau-medio 10 --saida dados/grafo_1m.npz

# um grafo menor, bom para testes rápidos (100 mil páginas)
python -m src.comum.gerar_grafo --vertices 100000 --grau-medio 10 --saida dados/grafo_100k.npz

# um grafo bem pequeno, com outra semente (sai um sorteio diferente)
python -m src.comum.gerar_grafo --vertices 5000 --grau-medio 5 --semente 7 --saida /tmp/grafo_5k.npz
```

Com a mesma semente (o padrão é 42), o grafo sai sempre igual. O de 1 milhão de
páginas tem 9.993.543 links e 46 páginas sem links de saída.

## 4. PageRank sequencial

```bash
# até convergir (para quando a mudança total fica abaixo de 1e-10)
python -m src.sequencial.pagerank --grafo dados/grafo_100k.npz

# 10 iterações fixas, como nas medições, mostrando as 10 páginas mais importantes
python -m src.sequencial.pagerank --grafo dados/grafo_1m.npz --tolerancia 0 --max-iteracoes 10 --top 10

# sem amortecimento (d = 1), só para comparar
python -m src.sequencial.pagerank --grafo dados/grafo_100k.npz --amortecimento 1.0
```

Saída esperada do primeiro comando: 21 iterações, soma das chances 100,0000% e a
página 0 em primeiro lugar, com 1,6015%.

## 5. PageRank com MPI

```bash
# 4 processos
mpirun -n 4 python -m mpi4py -m src.mpi.pagerank --grafo dados/grafo_100k.npz

# 8 processos com 10 iterações fixas
mpirun -n 8 python -m mpi4py -m src.mpi.pagerank --grafo dados/grafo_1m.npz --tolerancia 0 --max-iteracoes 10

# mais processos do que núcleos: precisa de --oversubscribe
mpirun -n 64 --oversubscribe python -m mpi4py -m src.mpi.pagerank --grafo dados/grafo_1m.npz --tolerancia 0 --max-iteracoes 10

# guardar o vetor final para comparar com outra versão
mpirun -n 4 python -m mpi4py -m src.mpi.pagerank --grafo dados/grafo_100k.npz --salvar-ranks /tmp/ranks_mpi.npy
```

O `-m mpi4py` derruba todos os processos se um deles der erro, em vez de deixar
os outros travados esperando.

### Versões alteradas do MPI

```bash
# alteração 1: peso de cada link calculado uma vez só
mpirun -n 4 python -m mpi4py -m src.mpi.pagerank_alterado --alteracao 1 --grafo dados/grafo_1m.npz --tolerancia 0 --max-iteracoes 10

# alteração 2: cada processo é dono de uma faixa de páginas de destino
mpirun -n 8 python -m mpi4py -m src.mpi.pagerank_alterado --alteracao 2 --grafo dados/grafo_1m.npz --tolerancia 0 --max-iteracoes 10

# alteração 3: allreduce minúsculo (o vetor é serializado antes de enviar)
mpirun -n 4 python -m mpi4py -m src.mpi.pagerank_alterado --alteracao 3 --grafo dados/grafo_1m.npz --tolerancia 0 --max-iteracoes 10

# otimizada: alterações 1 e 2 juntas
mpirun -n 8 python -m mpi4py -m src.mpi.pagerank_alterado --alteracao otimizada --grafo dados/grafo_1m.npz --tolerancia 0 --max-iteracoes 10
```

## 6. PageRank com Spark

```bash
# 4 núcleos desta máquina
python -m src.spark.pagerank --grafo dados/grafo_100k.npz --master "local[4]"

# 32 núcleos, 10 iterações fixas e 8 GB de memória
python -m src.spark.pagerank --grafo dados/grafo_1m.npz --master "local[32]" --tolerancia 0 --max-iteracoes 10 --memoria 8g

# escolhendo o número de partições (o padrão é igual ao número de núcleos)
python -m src.spark.pagerank --grafo dados/grafo_100k.npz --master "local[4]" --particoes 8
```

`local[4]` quer dizer "rode nesta máquina, com 4 núcleos". Abrir o Spark leva
alguns segundos; isso não entra no tempo de cálculo.

### Versões alteradas do Spark

```bash
# alteração 1: soma das páginas sem saída anotada durante a conta (observe)
python -m src.spark.pagerank_alterado --alteracao 1 --grafo dados/grafo_100k.npz --master "local[4]"

# alteração 2: links divididos por destino e tabela de notas em broadcast
python -m src.spark.pagerank_alterado --alteracao 2 --grafo dados/grafo_1m.npz --master "local[4]" --tolerancia 0 --max-iteracoes 10

# alteração 3: RDD (bem mais lenta; prefira o grafo pequeno)
python -m src.spark.pagerank_alterado --alteracao 3 --grafo dados/grafo_100k.npz --master "local[4]" --tolerancia 0 --max-iteracoes 10

# otimizada: alterações 1 e 2 juntas
python -m src.spark.pagerank_alterado --alteracao otimizada --grafo dados/grafo_1m.npz --master "local[4]" --tolerancia 0 --max-iteracoes 10
```

## 7. Medir (várias execuções seguidas)

O rodador repete cada caso (3 vezes por padrão), com 10 iterações fixas, e anota
uma linha por execução numa planilha CSV.

```bash
# sequencial, 3 vezes
python -m src.medicao.rodar --versao sequencial --grafo dados/grafo_1m.npz --csv /tmp/teste.csv

# MPI original com 1, 2, 4 e 8 processos
python -m src.medicao.rodar --versao mpi --grafo dados/grafo_1m.npz --processos 1 2 4 8 --csv /tmp/teste.csv

# MPI otimizada com 1 a 64 processos (64 passa dos núcleos: --oversubscribe)
python -m src.medicao.rodar --versao mpi --alteracao otimizada --grafo dados/grafo_1m.npz --processos 1 2 4 8 16 32 64 --mpi-extra=--oversubscribe --csv /tmp/teste.csv

# Spark com 1 e 4 núcleos, 1 repetição só, 8 GB de memória
python -m src.medicao.rodar --versao spark --grafo dados/grafo_100k.npz --processos 1 4 --repeticoes 1 --extra "--memoria 8g" --csv /tmp/teste.csv

# desistir de uma execução que passar de 10 minutos
python -m src.medicao.rodar --versao spark --alteracao 3 --grafo dados/grafo_1m.npz --processos 32 --repeticoes 1 --limite 600 --csv /tmp/teste.csv
```

O `=` em `--mpi-extra=--oversubscribe` é necessário: sem ele, o rodador acha que
`--oversubscribe` é uma opção dele e dá erro.

## 8. Analisar e gerar gráficos

```bash
# tabela com tempo médio, desvio, speedup e eficiência
python -m src.medicao.analisar --csv /tmp/teste.csv --saida /tmp/resumo.csv

# a mesma tabela para os resultados oficiais do cluster
python -m src.medicao.analisar --csv resultados/medicoes_cluster.csv --saida /tmp/resumo_cluster.csv

# gráficos de tempo, speedup e eficiência (um arquivo por versão)
python -m src.medicao.graficos --csv /tmp/teste.csv --pasta /tmp/graficos
```

Cuidado: sem `--pasta`, os gráficos vão para `resultados/graficos/` e substituem
os gráficos oficiais do cluster.

## 9. No cluster (nó n-11)

Entrar no nó (o atalho `n11` fica no arquivo `~/.ssh/config` do seu computador,
com o endereço, a porta e o usuário do slide do professor; a senha é pedida na
hora e não vai nesse arquivo):

```bash
ssh n11
```

Preparar o ambiente, rodar as medições oficiais (cerca de 50 minutos) e
acompanhar o andamento:

```bash
mkdir -p ~/grupo_pagerank && cd ~/grupo_pagerank
git clone https://github.com/LeoHazer/pagerank-mpi-spark.git
cd pagerank-mpi-spark
bash cluster/preparar.sh

nohup bash cluster/medir_tudo.sh > medicao.log 2>&1 &
tail -f medicao.log          # Ctrl+C só para de mostrar; a medição continua
```

Conferir se o nó está livre antes de medir (o primeiro número depois de
"load average" deve estar perto de 0):

```bash
uptime
```

Interromper as medições (o que já terminou continua na planilha):

```bash
pkill -f cluster/medir_tudo.sh; pkill -f src.medicao.rodar
```

Trazer a planilha para o seu computador (rodar no seu computador, não no
cluster):

```bash
scp n11:~/grupo_pagerank/pagerank-mpi-spark/resultados/medicoes_cluster.csv /tmp/
```
