'''
Vamos aplicar aqui mais uma métrica de avaliação de modelo que é a Métrica KS (Kolmogorov-Smirnoff)
onde vamos comparar duas curvas (o Gain cumulativo) das repostas do modelo e comparar a MAXIMA
distância entre elas, porque esperamos que as curvas sejam DIFERENTES.

O ideal é que esse numero seja um numero maior que um modelo naive de atribuição de probabilidades.

Vamos retomar nosso exemplo do modelo de churn para implementar a lógica manualmente e na sequencia
usar os métodos disponíveis em scikit
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
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import ensemble

from feature_engine import imputation

import matplotlib.pyplot as plt
import scikitplot as skplot

#%%

# Importação da base de dados
df = pd.read_csv('../../data/dados_pontos.csv' , sep=";")

# %%

# Separação das bases de treino e teste
features = df.columns.to_list()[3:-1]
target = 'flActive'

X_train , X_test , y_train , y_test = model_selection.train_test_split(
    df[features] , df[target] , test_size=0.2 , random_state=42 , stratify=df[target]
)

# Testar as taxas de resposta
print(f"Taxa de Resposta Treino : {y_train.mean()}")
print(f"Taxa de Resposta Teste : {y_test.mean()}")

# %%

# Tratamento dos dados 

# Procurar dados faltantes e definir os valores
X_train.isna().sum().T      # avgRecorrencia | 810 NAs <- Max(avgRecorrencia)

# (a) Imputar ZERO nas colunas que tenham NA
features_input_0 = [
    'qtdeRecencia', 'freqDias', 'freqTransacoes', 'qtdListaPresença',
    'qtdChatMessage', 'qtdTrocaPontos', 'qtdResgatarPonei' , 'qtdPresençaStreak', 
    'pctListaPresença', 'pctChatMessage', 'pctTrocaPontos', 'pctResgatarPonei',
    'pctPresençaStreak', 'qtdePontosGanhos', 'qtdePontosGastos' , 'qtdePontosSaldo'
    ]

imputacao_0 = imputation.ArbitraryNumberImputer(variables=features_input_0 ,
                                                     arbitrary_number=0)

# (b) Imputers de MAX para a coluna de avgRecorrencia (Natureza do Dado)
max_avgRecorrencia = X_train['avgRecorrencia'].max()
imputacao_max = imputation.ArbitraryNumberImputer(variables=['avgRecorrencia'] , 
                                                  arbitrary_number=max_avgRecorrencia)

# Aplicação do Modelo
modelo = ensemble.RandomForestClassifier(
    random_state=42 , n_estimators=250 , min_samples_leaf=20
)
# %%

# Implementação de Pipeline
meu_pipeline = pipeline.Pipeline(steps=[
    ('input_0' , imputacao_0) ,
    ('input_max' , imputacao_max) ,     
    ('model' , modelo)     # Agora o modelo recebe GRID como entrada
])

# Treinar o modelo com os dados de teste
meu_pipeline.fit(X_train , y_train)

# %%

# Diagnostico do Modelo

y_train_predict = meu_pipeline.predict(X_train)
y_train_proba = meu_pipeline.predict_proba(X_train)

y_test_predict = meu_pipeline.predict(X_test)
y_test_proba = meu_pipeline.predict_proba(X_test)

# %%

# Implementação

# Vamos montar um dataframe unindo os resultados reais e os PROBAS do nosso modelo
df_ks = pd.DataFrame({
    "flActive" : y_train ,
    "Proba_modelo" : y_train_proba[:, 1]
})

# Agora vamos ordenar nosso df com base no proba
df_ks = df_ks.sort_values(by='Proba_modelo' , ascending=True)

# E preparar a tabela para agrupar por classes de probabilidade

# Criar os binários das colunas
df_ks['is_1'] = (df_ks['flActive']==1).astype(int)
df_ks['is_0'] = (df_ks['flActive']==0).astype(int)

# Soma cumulativamente as colunas, ja que as linhas já estão ordenadas
df_ks['cum_is_1'] = df_ks['is_1'].cumsum()
df_ks['cum_is_0'] = df_ks['is_0'].cumsum()

# Agora vamos criar uma coluna para criar classses de probabilidades
n_classes = 50
steps = 100 / n_classes
df_ks['classe'] = (((df_ks['Proba_modelo'] * 100) // steps) * steps).astype(int)

# Agora agrupamos os valores do df transformado (absoluto)
df_ks_classes = df_ks.groupby('classe').agg(
    qtde_1 = ('is_1' , 'sum') ,
    qtde_0 = ('is_0' , 'sum')
).sort_index().reset_index()

# Calculamos os acumulados absolutos
df_ks_classes['cumsum_1'] = df_ks_classes['qtde_1'].cumsum()
df_ks_classes['cumsum_0'] = df_ks_classes['qtde_0'].cumsum()

# e relativos
df_ks_classes['cumsum%_1'] = df_ks_classes['cumsum_1'] / df_ks_classes['cumsum_1'].iloc[-1]
df_ks_classes['cumsum%_0'] = df_ks_classes['cumsum_0'] / df_ks_classes['qtde_0'].sum()

# Agora calculamos as diferenças (em modulo) em cada trecho
df_ks_classes['diff'] = abs(df_ks_classes['cumsum%_1'] - df_ks_classes['cumsum%_0'])

# Capturamos o maior valor dessa coluna
df_ks_classes[df_ks_classes['diff'] == df_ks_classes['diff'].max()]

# Com 20 classes o maior valor alcançado foi 0.58 na classe 30
# Com 50 classes o maior valor alcançado foi 0.58 na classe 35

# Ou seja, para encontrar o ponto da maxima distancia entre as curvas precisariamos aumentar
# muito o numero de classes para encontar esse ponto

# %%

###########################
# Grafico K-S do SciKitPlot
###########################

# É aqui que entra o KS do SCIKITPLOT de maneira muito simples
skplot.metrics.plot_ks_statistic(y_true=y_train , y_probas=y_train_proba)

# E é exibido o valor da estatistica 0.586 na posição X em 0.354

# Esse método retorna mais estatisiticas que geraram o grafico KS
skplot.metrics.binary_ks_curve(y_true=y_train , y_probas=y_train_proba[: , 1])

# Vamos supor um modelo naive que devolve probabilidades aleatoramente a cada ponto e comparar
# com nosso modelo usando K-S

import random as rnd

y_train_proba_naive = []

for i in y_train_proba:
    proba = rnd.random()
    y_train_proba_naive.append( [ 1 - proba , proba ] )

skplot.metrics.plot_ks_statistic(y_true=y_train , y_probas=y_train_proba_naive)

# E é exibido o valor da estatistica 0.017 na posição X em 0.472
