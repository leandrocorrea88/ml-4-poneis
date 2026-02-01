
# %%

# Resolução do Case de Star Wars

import pandas as pd

df = pd.read_parquet('../../data/dados_clones.parquet')
df

# %%

# Primeiro rodamos uma análise desciritiva para tentar entender ONDE pode
# estar o problema

# Entender as médias de alguma caracteristicas, pra ver se há diferença significativa
df.groupby(['Status'])[['Estatura(cm)' , 'Massa(em kilos)' , 'Tempo de existência(em meses)']].mean()

# Deu ruim com o nome das colunas. Verificar os nomes pra entender se há espaços
df.columns

# Realmente há espaços, então vamos remover
# Index(['p2o_master_id', 'Massa(em kilos)', 'General Jedi encarregado',
#        'Estatura(cm)', 'Distância Ombro a ombro', 'Tamanho do crânio',
#        'Tamanho dos pés', 'Tempo de existência(em meses)', 'Status '],
#       dtype='str')
novas_colunas = [
    i.strip(" ")
    for i
    in df.columns.to_list()
]

df.columns = novas_colunas

# %%

###############################################################
# MODELO 1 - USANDO ANÁLISES UNIVARIADAS (UMA VARIAVEL POR VEZ
###############################################################

# %% 

# Testar as características intrinsecas dos clones
# Hipótese : Lotes defeituosos

df.groupby(['Status'])[['Estatura(cm)' , 'Massa(em kilos)' , 'Tempo de existência(em meses)']].mean()

'''
Os valores de Estatura e Massa estão muito proximos (180cm e 83kg), 
mas o Tempo de Existencia tem uma diferença um pouco maior 
(Apto 22.9 vs Def 31.45)
'''

# Agora vamos testar as PROPORÇÕES de defeituosos considerando as outras colunas que trazem
# variáveis categóricas

# Para possibilitar as proporções, vamos converter a coluna de alvo para um valor binario
df['Status_bol'] = df['Status'] == 'Apto'

# Agora vamos calcular as proporções de aptos e defeituosos pelas colunas do tipo texto
df.dtypes

df.groupby(["Distância Ombro a ombro"])['Status_bol'].mean()
df.groupby(["Tamanho do crânio"])['Status_bol'].mean()
df.groupby(["Tamanho dos pés"])['Status_bol'].mean()

'''
Em todos os 3 testes anteriores vemos que a proporção de defeituosos é muito proxima entre
todos os grupos analisados (~89% de aptos), o que não nos permite afirmar que temos caracteristicas
dos clones que expliquem esse fenômeno

CONCLUSAO : Até então, nenhuma caracteristica do clone explica os defeituosos

'''
# %%

# Testar os desempenhos de cada general Jedi
# Hipótese : Problemas em lideranças/métodos

df.groupby(['General Jedi encarregado'])['Status_bol'].mean()

'''
Resultado: 2 generais tem uma proporção menor de APTOS

General Jedi encarregado
Aayla Secura      1.000000
Mace Windu        1.000000
Obi-Wan Kenobi    1.000000
Shaak Ti          0.765018  <<
Yoda              0.764945  <<

Com esse novo recorte temos uma hipótese que poderia ser melhor trabalhada abrindo por general

CONCLUSÃO : Há um problema maior com defeituosos nas divisões de Yoda e Shaak Ti

'''

# %%
# %%

###################################################################
# MODELO 2 - USANDO ARVORES DE DECISÃO PARA ANÁLISES MULTIVARIADAS
###################################################################

# Testar as características intrinsecas dos clones
# Hipótese : Lotes defeituosos

features = [
    'Massa(em kilos)', 
    'Estatura(cm)', 
    'Distância Ombro a ombro', 
    'Tamanho do crânio', 
    'Tamanho dos pés' ,
    'Tempo de existência(em meses)'
]

# ATUALIZAÇÃO: Vamos tentar usar todas as variáveis para tentar extrapolar a Hipótese 1
# features = [
#     'Massa(em kilos)', 
#     'Estatura(cm)', 
#     'Distância Ombro a ombro', 
#     'Tamanho do crânio', 
#     'Tamanho dos pés' ,
#     'Tempo de existência(em meses)' ,
#     'General Jedi encarregado'
# ]


X = df[features]
X

# Temos 3 variaveis que estão em TEXTO (Tamanhos e Distancias)

# %%

# Transformando variáveis categóricas

# Vamos converte-las para numeros usando recursos das bibliotecas
# pip install feature-engine

from feature_engine import encoding

# Segmentamos as variaveis CATEGORICAS
cat_features = [
    'Distância Ombro a ombro', 
    'Tamanho do crânio', 
    'Tamanho dos pés'
]

# ATUALIZAÇÃO: Vamos tentar usar todas as variáveis para tentar extrapolar a Hipótese 1
# cat_features = [
#     'Distância Ombro a ombro', 
#     'Tamanho do crânio', 
#     'Tamanho dos pés' ,
#     'General Jedi encarregado'
# ]

# Passamos a lista para o Encoder ...
onehot = encoding.OneHotEncoder(variables=cat_features)
# ... invocamos o método fit para que ele APRENDA as CLASSES em cada variavel ...
onehot.fit(X)
# ... e transformamos cada nivel das categorias em uma coluna binária ...
X = onehot.transform(X)
# ... mas preservando as colunas que não foram passadas para o OneHotEnconder
X

# %%

# Aplicando Arvore de Decisão para análise

from sklearn import tree
import matplotlib.pyplot as plt

# Vamos começar com uma árvore de 3 níveis para testar se já seria o suficiente para
# chegar em nós puros e/ou enxergar padrões
arvore = tree.DecisionTreeClassifier(max_depth=4)
arvore.fit(X=X , y=df['Status'])

plt.figure(dpi=1200)

tree.plot_tree(arvore ,
               class_names=arvore.classes_ ,
               feature_names=X.columns ,    # <- vamos usar X.columns em vez de features , pois 
                                            # na versão extrapolada deu estouro na lista usada como
                                            # parâmetro
               filled=True)

'''
A árvore de 3 níveis retornou um padrão onde o primeiro nó divide grupos em [MASSA <= 83,405] e 
deriva um nó puro de Aptos para os casos False, ou seja, todos os clones ACIMA DESSA MASSA SÃO APTOS!

O próximo nó faz um corte em [ESTURA <= 180.555] derivando novamente outro nó puro de APTOS, com a 
segunda conclusão de que AINDA QUE TENHAM MASSA <= 83,405, CLONES COM ESTATURA > 180.555 SÃO TODOS APTOS!

Um detalhe é que considerando apenas esses dois cortes já temos 843k dos 1.05 clones.

Agora temos um novo corte, AINDA EM ESTATURA mas com [ESTATURA <= 180.245] onde os VERDADEIROS desse corte
são TODOS APTOS. No outro sentido temos um ainda não puro, mas analisando a proporção de defeituosos nesse
nó (Total: 152.070 , Aptos 41.544 [27,4%] e Defeituosos 110.526 [72,6%]) é possivel chegar em outra conclusão
que HÁ UM POTENCIAL PROBLEMA EM CLONES COM MASSA <= 83,405 E ESTATURA ENTRE 180.245 E 180.555.

Ao testar a árvore com 4 níves o nó não puro anterior foi dividido em [DISTANCIA OMBRO A OMBRO <= 16,85]
porém ainda resultando em 2 nós não puros. Com isso temos:

CONCLUSÕES: 

1. Há um potencial problema em clones com massa de até 83,405kg e com estatura entre 180,245 e 180,555

2. Mesmo que estivéssemos analisando 6 caracteristicas físicas, apenas 2 foram suficientes para entender
potenciais focos de problema

3. O modelo de árvore de 3 níveis foi suficiente para chegar nas conclusões, pois o modelo de 4 níveis
não apresentou nós puros em N4

ATUALIZAÇÃO:

Ao adicionar a coluna de General Jedi (e fazer as transformações do OneHotEncoder, aplicamos uma nova
versão da árvore com 4 níveis e chegamos a um novo step com nó puro de APTOS usando o corte de
[General Obi-Wan <= 0] em VERDADEIRO, ou seja, ainda que os clones tenham as especificações acima
o Obi-Wan está tendo aptos, enquanto que essas mesmas especificações nas mãos dos outros generais
estão com uma proporção alta de defeituosos (Total: 138.521 , Aptos 28.015 [20%] e Defeituosos 110.526 [80%])

CONCLUSÕES ADICIONAIS:

4. Ainda que tenhamos problemas de desempenho com os clones da especificação em (1) ainda assim o general
Obi-Wan consegue fazer bom proveito deles. O que ele tem feito de diferente?

'''

# %%

# Confrontando as decisões

'''

Pela análise univariada, a conclusão seria mandar Yoda e Shaak Ti pro RH pra fazer treinamento e
acompanha-los em ciclos de feecback

Pela análise multivariada temos duas possíveis conclusões

- Não comprar mais os clones nas especificações encontradas
- Se eles forem mais baratos podemos dividir entre os generais para reduzir as perdas
- Porém ao rodar a versão completa da árvore, vemos que podemos mandar esses clones pra divisão
do Obi-Wan porque ele está sabendo usar esses recursos

'''