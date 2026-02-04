'''
Nessa versão do código vamos trabalhar de maneira mais profissional com a modelagem
de ML, aplicando algumas boas práticas ao longo do processo

Roteiro Macro

    1. Importação da Base
    2. Split de Treino e Teste
    3. Tratamento dos dados (na base de TREINO) - Imputations/Tranformações
    4. Configuração do modelo base
    5. Tuning de parâmetros/hiperparâmetros (GridSearch, RandomSearch)
    6. Escolha do Modelo Final
    7. Predict sobre base de teste
    8. Avaliação e Diagnóstico com as bases

'''

# %%

################################
# Bibliotecas usadas no projeto
################################
import pandas as pd

from sklearn import metrics
from sklearn import model_selection
from sklearn import pipeline

from sklearn import tree
#from sklearn import linear_model
#from sklearn import naive_bayes
from sklearn import ensemble

from feature_engine import imputation

#import matplotlib.pyplot as plt
#import scikitplot as skplot

#%%

#############################
# Importação da base de dados
#############################

df = pd.read_csv('../../data/dados_pontos.csv' , sep=";")
df

# %%

#######################################
# Separação das bases de treino e teste
#######################################

features = df.columns.to_list()[3:-1]
target = 'flActive'

X_train , X_test , y_train , y_test = model_selection.train_test_split(
    df[features] , df[target] , test_size=0.2 , random_state=42 , stratify=df[target]
)

# Testar as taxas de resposta
print(f"Taxa de Resposta Treino : {y_train.mean()}")
print(f"Taxa de Resposta Teste : {y_test.mean()}")

# %%

#######################
# Tratamento dos dados 
#######################

# Procurar dados faltantes e definir os valores
X_train.isna().sum().T      # avgRecorrencia | 810 NAs <- Max(avgRecorrencia)

# (a) Imputar ZERO nas colunas que tenham NA
features_input_0 = [
    'qtdeRecencia',
    'freqDias',
    'freqTransacoes',
    'qtdListaPresença',
    'qtdChatMessage',
    'qtdTrocaPontos',
    'qtdResgatarPonei',
    'qtdPresençaStreak',
    'pctListaPresença',
    'pctChatMessage',
    'pctTrocaPontos',
    'pctResgatarPonei',
    'pctPresençaStreak',
    'qtdePontosGanhos',
    'qtdePontosGastos' ,
    'qtdePontosSaldo'
    ]

imputacao_0 = imputation.ArbitraryNumberImputer(variables=features_input_0 ,
                                                     arbitrary_number=0)

# (b) Imputers de MAX para a coluna de avgRecorrencia (Natureza do Dado)
max_avgRecorrencia = X_train['avgRecorrencia'].max()
imputacao_max = imputation.ArbitraryNumberImputer(variables=['avgRecorrencia'] , 
                                                  arbitrary_number=max_avgRecorrencia)

#%%

######################
# Aplicação do Modelo
######################

modelo = tree.DecisionTreeClassifier(max_depth=4 , min_samples_leaf=50 ,
                                     random_state=42)

# %%

############################
# Implementação de Pipeline
############################

'''
Agora apara garantir que ao rodar o predict na base de teste os mesmos tratamentos sejam feitos
vamos encapsular as etapas anteriores no método PIPE, que, nesse nosso caso, ao final vai retornar
um objeto do tipo MODELO.
'''
# %%

# Aqui, cada linha é uma TUPLA com um nome livre para a etapa e a variável que armazena a saida
meu_pipeline = pipeline.Pipeline(steps=[
    ('input_0' , imputacao_0) ,
    ('input_max' , imputacao_max) ,     
    ('modelo' , modelo)                 
])

# Agora podemos acessar o pipeline e aplicar os métodos de modelos
meu_pipeline.fit(X_train , y_train)

# %%

########################
# Diagnostico do Modelo
########################

# Referenciamos o PIPELINE para acessar os métodos e atributos do MODELO

y_train_predict = meu_pipeline.predict(X_train)
y_train_proba = meu_pipeline.predict_proba(X_train)[: , 1]

y_test_predict = meu_pipeline.predict(X_test)
y_test_proba = meu_pipeline.predict_proba(X_test)[: , 1]

# Calcular as métricas
acc_train = metrics.accuracy_score(y_train , y_train_predict)
acc_test = metrics.accuracy_score(y_test , y_test_predict)
print(f"Acuracia base Treino : {acc_train}")
print(f"Acuracia base Teste : {acc_test}")

auc_train = metrics.roc_auc_score(y_train , y_train_proba)
auc_test = metrics.roc_auc_score(y_test , y_test_proba)
print(f"AUC base Treino : {auc_train}")
print(f"AUC base Teste : {auc_test}")

# %%

'''
DecisionTreeClassifier
    Acuracia base Treino : 0.8109619686800895
    Acuracia base Teste : 0.8008948545861297
    AUC base Treino : 0.8531284015204619
    AUC base Teste : 0.8380512447094162

Os resultados do modelo podem ser alterados a medida em que mudamos os hipeparametros do modelo,
então fica a pergunta QUAL É A CONFIGURAÇÃO OTIMA DE HIPERPARAMETROS QUE MAXIMIZA A PERFORMANCE?

Para isso podemos usar o GridSearch que é um ORQUESTRADOR DE EXPERIMENTOS, que fatia a base de
em porções menores (por padrão, 3) e aplica uma validação cruzada (Modela em 1, teste em 2 e 3 em
sua possiveis cmbinações) e roda isso iterando em uma serie de hiperparametros que passamos. Vamos
usar esse recurso no estrato a partir da linha de codigo abaixo
'''

# %%

'''
Vamos manter as 5 primeiras etapas da primeira versão do fluxo, sendo elas

    - Importação dos dados
    - Separação das bases de treino e teste
    - Tratamento dos dados

Vamos a partir de agora refazer essa duas etapas para seguir

    - Aplicação do Modelo
    - Implementação de Pipeline

Vamos usar as mesmas variáveis criadas nessas etapas anteriores para incluir a etapa de Tuning
do modelo usando GridSearch
'''

# %%

#####################
# Aplicação do Modelo
#####################

# Retomando o modelo e pipeline criados

modelo = tree.DecisionTreeClassifier(random_state=42)

# %%

############################
# Implementação de Pipeline
############################

meu_pipeline = pipeline.Pipeline(steps=[
    ('input_0' , imputacao_0) ,
    ('input_max' , imputacao_max) ,     
    ('model' , modelo)                 
])

# %%

####################################
# Tuning do Modelo usando GridSeach
####################################

# Nessa versão vamos retirar os hiperparametros que queremos iterar para descobrir a melhor
# configuração: max_depth e min_samples_leaf e criamos um dicionario com a lista de testes,
# e como estamos acessando hiperparametros de um modelo que está encapsulado em um pipeline
# precisamos do prefixo e __ no dicionario

params = {
    "model__max_depth" : [4 , 5 , 6 , 7] ,
    "model__min_samples_leaf" : [ 10 , 30 , 50 , 50 , 100]
}

grid = model_selection.GridSearchCV(
    estimator=meu_pipeline ,    # Recebe o Pipe como entrada
    param_grid=params ,         # Acessa os parametros do dicionario (com prefixo do membro)
    n_jobs=-1 ,                 # Maximo de processadores
    cv = 3 ,                    # Estratégia de Cross Validation com 3 splits
    scoring='roc_auc'           # OBS.: Liste os métodos de scoring com metrics.get_scorer_names()
)

# E agora que faz o FIT é o GRID
grid.fit(X_train , y_train)

# %%

########################
# Diagnostico do Modelo
########################

# Para descobrir os parametros definimos podemos acessar
grid.best_params_

# Para listar todos os cenários testados com cv_results em um DataFrame
# como a métrica escolhida foi roc_auc o mean_test_score e std_test_score representam a média
# e variação da métrica nos splits definidos em CV
df_grid = pd.DataFrame(grid.cv_results_)
# podemos adicionar o coeficiente de variação para avaliar a variabilidade entre os modelos 
# testados
df_grid['coef_var_test_score'] = df_grid['mean_test_score'] / df_grid['std_test_score']
# Ao ordenar vemos que o melhor modelo tem maior média e tambem menor variabilidade de resultados
df_grid.sort_values(by='rank_test_score', ascending=False)

# Podemos repetir todo o processo de diagnostico, sabendo que devemos referenciar o GRID e não
# mais o PIPELINE para acessar

y_train_predict = grid.predict(X_train)
y_train_proba = grid.predict_proba(X_train)[: , 1]

y_test_predict = grid.predict(X_test)
y_test_proba = grid.predict_proba(X_test)[: , 1]

# Calcular as métricas
acc_train = metrics.accuracy_score(y_train , y_train_predict)
acc_test = metrics.accuracy_score(y_test , y_test_predict)
print(f"Acuracia base Treino : {acc_train}")
print(f"Acuracia base Teste : {acc_test}")

auc_train = metrics.roc_auc_score(y_train , y_train_proba)
auc_test = metrics.roc_auc_score(y_test , y_test_proba)
print(f"AUC base Treino : {auc_train}")
print(f"AUC base Teste : {auc_test}")

# %%

'''
Comparação dos Modelos
                            (1) Atual                   (2) Anterior
                            ----------------------      ---------------------
                            max_depth': 4               max_depth': 5               
                            min_samples_leaf': 50       min_samples_leaf': 50
                            ----------------------      ---------------------
    Acuracia base Treino    0.819910514541387           0.8109619686800895
    Acuracia base Teste     0.8120805369127517          0.8008948545861297
    AUC base Treino         0.8675601089626326          0.8531284015204619
    AUC base Teste          0.8354002639603151          0.8380512447094162

Nessa segunda abordagem temos uma leve melhora a ACC de Treino e Teste, com uma leve melhoria 
de AUC.

Aqui testamos 4 possíveis valores de max_depth e 5 possíveis valores de min_samples_leaf, o que
gera a execução e comparação de 60 modelos com estrategia de cross validation em 3 splits cada.

A desvantagem dessa abordagem é que o custo computacional fica alto, pois a cada iteração do GRID
o Pipe é disparado, executando todas as tarefas para cada nova combinação testada

             --------------------------------------
             |               meu_pipe             |     <- É chamado por grid.fit()
             --------------------------------------
Iteração 01 :   Imput_0     Imput_avg       Model01     <- Cada iteração executa os IMPUTS
Iteração 02 :   Imput_0     Imput_avg       Model02
Iteração 03 :   Imput_0     Imput_avg       Model03
...
Iteração 20 :   Imput_0     Imput_avg       Model20

Agora vamos testar uma nova abordagem onde vamos INSERIR O GRID NO PIPELINE, com isso forçando a
execução dos IMPUTS para acontecerem SOMENTE UMA VEZ e alimentarem os modelos do grid
'''
# %%

'''
Semelhante a arquitetura anterios, vamos manter as 5 primeiras etapas da primeira versão do fluxo, 
sendo elas

    - Importação dos dados
    - Separação das bases de treino e teste
    - Tratamento dos dados

Vamos a partir de agora refazer essa duas etapas para seguir

    - Aplicação do Modelo           <- Agora vamos usar RandomForest
    - Implementação de Pipeline     <- Inserindo o GRID dentro do pipeline

Nessa versão vamos usar RANDOM FOREST onde criaremos uma quantidade de arvores de decisão com seus
parametros e então

    - Cada arvore receberá uma amostra dos dados (levemente diferente, inclusive features)
    - Cada arvore é treinada respeitando os hiperparametros de construção
    - Cada arvore "vota" em uma classe
    - O resultado final é a MAIORIA
    - PROBA é a média das probabilidades das árvores

Comparando DecisionTree com RandomForest

                            Arvore          Floresta
                            ---------       -----------
    Variância	            Alta	        Baixa
    Interpretabilidade	    Alta	        Baixa
    Robustez	            Baixa	        Alta
    Tuning	                Simples	        Mais parâmetros
    Uso em produção	        Raro	        Muito comum

A ideia é que Random Forest não aprenda uma regra perfeita, mas aprenda MUITAS REGRAS IMPERFEITAS
e CONFIE NA MÉDIA
'''

# %%

#####################
# Aplicação do Modelo
#####################

# Retomando o modelo e pipeline criados

modelo = ensemble.RandomForestClassifier(random_state=42)

# Como agora o grid fará parte do Pipe e vai receber diretamente o modelo o dicionário não
# precisa conter a referencia ao membro do pipe. Os hiperparametros que estarão no dicionario
# não deve ser declarados na instanciação do modelo

params={
    #"max_depth" : [3,5,10,10,15,20] ,
    "n_estimators" : [100,150,250,500,1000] ,
    "min_samples_leaf" : [10,20,30,50,100] ,
}

grid = model_selection.GridSearchCV(
    estimator=modelo , param_grid=params , scoring='roc_auc' ,
    n_jobs=-1 
)

# %%

############################
# Implementação de Pipeline
############################

meu_pipeline = pipeline.Pipeline(steps=[
    ('input_0' , imputacao_0) ,
    ('input_max' , imputacao_max) ,     
    ('model' , grid)     # Agora o modelo recebe GRID como entrada
])

# Treinar o modelo com os dados de teste
meu_pipeline.fit(X_train , y_train)

# %%

########################
# Diagnostico do Modelo
########################

# Para descobrir os parametros definidos precisamos acessar o ultimo membro (GRID) que contem as
# propriedades
meu_pipeline[-1].best_params_
meu_pipeline[-1].best_estimator_
meu_pipeline[-1].classes_


# Para listar todos os cenários testados com cv_results em um DataFrame
# como a métrica escolhida foi roc_auc o mean_test_score e std_test_score representam a média
# e variação da métrica nos splits definidos em CV
df_pipe = pd.DataFrame(meu_pipeline[-1].cv_results_)
# podemos adicionar o coeficiente de variação para avaliar a variabilidade entre os modelos 
# testados
df_pipe['coef_var_test_score'] = df_pipe['mean_test_score'] / df_pipe['std_test_score']
# Ao ordenar vemos que o melhor modelo tem maior média e tambem menor variabilidade de resultados
df_pipe.sort_values(by='rank_test_score', ascending=False)

# Podemos repetir todo o processo de diagnostico, sabendo que devemos referenciar o GRID e não
# mais o PIPELINE para acessar

# Referenciamos o PIPELINE para acessar os métodos e atributos do MODELO

y_train_predict = meu_pipeline.predict(X_train)
y_train_proba = meu_pipeline.predict_proba(X_train)[: , 1]

y_test_predict = meu_pipeline.predict(X_test)
y_test_proba = meu_pipeline.predict_proba(X_test)[: , 1]

# Calcular as métricas
acc_train = metrics.accuracy_score(y_train , y_train_predict)
acc_test = metrics.accuracy_score(y_test , y_test_predict)
print(f"Acuracia base Treino : {acc_train}")
print(f"Acuracia base Teste : {acc_test}")

auc_train = metrics.roc_auc_score(y_train , y_train_proba)
auc_test = metrics.roc_auc_score(y_test , y_test_proba)
print(f"AUC base Treino : {auc_train}")
print(f"AUC base Teste : {auc_test}")

# %%
'''
Comparação dos Modelos
                            (1) Atual                   (2) Anterior
                            ----------------------      ---------------------
                            n_estimators: 250           max_depth: 4               
                            min_samples_leaf': 20       min_samples_leaf': 50
                            ----------------------      ---------------------
    Acuracia base Treino    0.8098434004474273          0.819910514541387
    Acuracia base Teste     0.8008948545861297          0.8120805369127517
    AUC base Treino         0.8729569506419792          0.8675601089626326
    AUC base Teste          0.8565284667546533          0.8354002639603151

Nessa terceira abordagem temos uma leve piora a ACC de Treino e Teste, com uma melhoria de AUC.

Também implementamos a nova versão do pipe onde agregamos o grid ao processo garantindo que os
imputs aconteçam somente uma unica vez e alimentem todos os modelos

             ----------------------------------------------
             |                    meu_pipe                |      <- É chamado por meu_pipe.fit()
             ----------------------------------------------
                                                 ---------
Iteração 01 :                                   | Model01 |     
Iteração 02 :  ------------------------         | Model02 |
Iteração 03 : |  Imput_0     Imput_avg | -->    | Model03 |     <- IMPUTS uma unica vez
...            ------------------------         |   ...   |
Iteração 20 :                                   | Model20 |
                                                 ---------
'''