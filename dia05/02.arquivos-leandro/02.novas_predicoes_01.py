'''
Nesse arquivo vamos implementar o modelo validado e model_train.py e avançar em recursos de como
podemos exportar o modelo para uso em outro dataset
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

#####################
# Aplicação do Modelo
#####################

# Retomando o modelo validado no gridsearch

modelo = ensemble.RandomForestClassifier(
    random_state=42 , n_estimators=250 , min_samples_leaf=20
)
# %%

############################
# Implementação de Pipeline
############################

# Vamos mudar o nome do pipeline para algo mais genérico
model = pipeline.Pipeline(steps=[
    ('input_0' , imputacao_0) ,
    ('input_max' , imputacao_max) ,     
    ('model' , modelo)     # Agora o modelo recebe GRID como entrada
])

# Treinar o modelo com os dados de teste
model.fit(X_train , y_train)

# %%

########################
# Diagnostico do Modelo
########################

# Referenciamos o PIPELINE para acessar os métodos e atributos do MODELO

y_train_predict = model.predict(X_train)
y_train_proba = model.predict_proba(X_train)[: , 1]

y_test_predict = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)[: , 1]

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

###############################
# Exportando o modelo para uso
###############################

'''
Após definir o modelo campeão ele estará restrito a esse arquivo (memória RAM), então precisamos
persisti-lo em memória tornando-o logenvo. Para isso podemos SERIALIZAR esses objetos em um
arquivo do tipo PICKLE encapsulando toda essa estrutura em um arquivo binário.

Vamos fazer isso salvado os principais achados.
'''

# Armazendo os objetos em uma SERIES
model_s = pd.Series({
    "model" : model ,           # Pipeline do modelo que tem o método FIT
    "features" : features ,     # Lista de features (nomes de colunas em lista)
    "auc_train" : auc_train ,   # Score AUC do Treino para referencia
    "auc_test" : auc_test       # Socre AUC do Teste para referencia 
})

# Agora podemos salvar essa estrutura completa em um binário
model_s.to_pickle(path="model_rf.pkl")
