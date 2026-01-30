
# Vamos fazer os primeiros exercicios baseado nos dados arquivo dado_frutas.xlsx
# para entender os primeiros conceitos em aprendizado de maquina

# %%

import pandas as pd

df = pd.read_excel(io="../../data/dados_frutas.xlsx")
df

# %%

# O DF representa uma base com algumas caracteristicas de frutas com o nome da fruta
# ao final. Baseado nisso, como podemos aplicar o método apresentado no slide para
# descobrir a fruta ao final?

# Ordem dos criterios no slide: (1) Redonda/Esfera (2) Suculenta (3) Vermelha (4) Doce

# 01. O DF total tem um total de 7 possiveis opções
df.shape

# 02. Aplicando o filtro de Arredondada = True vemos que reduzimos para 5 posíveis opções
filtro_redonda = df['Arredondada'] == 1
df[filtro_redonda]

# 03. Aplicando o segundo filtro de Suculenta a quantidade e opções cai para 4 possíveis opções
filtro_suculenta = df['Suculenta'] == 1
filtro_total = filtro_redonda & filtro_suculenta
df[filtro_total]

# 04. Filtro Vermelha -> 3 opçoes
filtro_vermelha = df['Vermelha'] == 1
filtro_total = filtro_total & filtro_vermelha
df[filtro_total]

# 05. Filtro Doce -> 2 opçoes
filtro_doce = df['Doce'] == 1
filtro_total = filtro_total & filtro_doce
df[filtro_total]

# Esse formato nos obrigou a inserir os filtros MANUALMENTE em uma ordem específica

# %%

# Agora vamos entender a maneira que a maquina aprende esses padrões aplcados

from sklearn import tree

# Tudo em ML funciona com a definição sobre quem são as variaveis (atributos e respostas)
# No nosso caso, as colunas iniciais do DF são as caracteristicas (ou FEATURES)
# e a coluna com o nome da fruta é o nosso alvo (TARGET)

features = ['Arredondada' , 'Suculenta' , 'Vermelha' , 'Doce']
target = ['Fruta']

# E separamos os DFs entre x (features) e y (target)
X = df[features]
y = df[target]

# Então podemos chamar o algoritmo usando nossas informações...
arvore = tree.DecisionTreeClassifier(random_state=42)
# e invocar o método que executa a rotina de aprendizado
arvore.fit(X=X , y=y)

# Ao executar com sucesso, temos o resultado na tela Interactive Window
# e o objeto ARVORE pode ser acessado

# %%

# Plotando os resultados em arvore

import matplotlib.pyplot as plt

# UDPATE : Ao rodar a imagem abaixo a qualidade ficou ruim, então vamos importar
# explicitamente o PyPlot para mexer nesse parametro
plt.figure(dpi=1200)

# Tambem podemos plotar a árvore para poder enxergar melhor o resultado
tree.plot_tree(decision_tree=arvore ,
               class_names=arvore.classes_,
               feature_names=features,
               filled=True)

plt.savefig('01a. Arvore Frutas.png')

# Os ramos coloridos são chamados de PUROS, ou seja, resultados obtidos de maneira unica
# no fluxo de aprendizado

# A leitura é sempre para a DIREITA temos as instruções FALSE, então no caso do primeiro ramo
# temos a leitura Arredondada <= 0.5 que significa =0 ou seja, não é arredondada, então o caminho
# falso indica =1, ou seja É ARREDONDADA

# Cada ramo tem a quantidade de elementos (samples) e o PRIMEIRO VALOR DA CLASSE, então temos 5
# ramos PUROS e 1 que ficou em branco, com 2 elementos que o algoritmo não encontrou features
# suficientes para segmentar (Cereja e Maçã tem a mesma combinação de features)

# %%

# Prevendo novos valores

# CLASSES retorna a lista com os valores alvo
arvore.classes_

# Para prever novos valores passamos uma lista com os parametros do mesmo tamanho de df_x
# features = ['Arredondada' , 'Suculenta' , 'Vermelha' , 'Doce']

# A instrução abaixo retorna : array(['Pera'], dtype=object)
arvore.predict(X=[[1 , 1 , 0 , 1]])

# A instrução abaixo retorna : array(['Cereja'], dtype=object)
# porém como vimos, esse ramo NÃO É PURO e contém 2 elementos ...
arvore.predict(X=[[1 , 1 , 1 , 1]])

# ... em outras palavras, mesmo que tenhamos 2 possíveis resultado para esse ramo, a resposta
# vai trazer APENAS o primeiro elemento. Como vemos isso?

# Ao passar essa instrução temo como retorno uma lista com o seguinte formato
# array([[0. , 0.5, 0. , 0.5, 0. , 0. , 0. ]]) que passa a LISTA DE PROBABILIDADES 
# para CADA CLASSE (ALVO) corresponder aos preditores passados
probas = arvore.predict_proba(X=[[1 , 1 , 1 , 1]])

# Para traduzir melhor podemos concatenar essa lista de probabilidades com as classes
# que a arvore já possui para visualizar melhor o resultado
pd.Series(data=probas[0] ,          # Primeiro elemento pois temos uma LISTA de LISTAS
          index=arvore.classes_)    # Classes

# E como resultado temos DUAS possíveis respostas, mas o retorno é SEMPRE UM ELEMENTO
#   Banana     0.0
#   Cereja     0.5  <
#   Limão      0.0
#   Maçã       0.5  <
#   Morango    0.0
#   Pera       0.0
#   Tomate     0.0

# Ou seja a PROBABILIDADE de cada classe baseada nos preditores

# Então, a resposta do modelo é sempre o alvo que tem maior probabilidade de corresponder
# às features passadas

# É possivel trabalhar com as SAÍDAS DO MODELO para refiná-lo posteriormente com os PROBAS

# %%
