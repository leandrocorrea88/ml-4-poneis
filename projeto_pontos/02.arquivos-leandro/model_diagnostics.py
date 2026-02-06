'''
Nesse arquivo vamos implementar o modelo validado e model_train.py e avançar em recursos de como
podemos explorar os resultados e diagnosticar a performance do modelo escolhido
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

meu_pipeline = pipeline.Pipeline(steps=[
    ('input_0' , imputacao_0) ,
    ('input_max' , imputacao_max) ,     
    ('model' , modelo)     # Agora o modelo recebe GRID como entrada
])

# Treinar o modelo com os dados de teste
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
# Curva ROC
curva_roc_auc = metrics.roc_curve(y_train , y_train_proba)

# Plot da ROC
fig , ax = plt.subplots()
ax.plot(curva_roc_auc[0] , curva_roc_auc[1] , '-')
# Plot da diagonal NAIVE
ax.plot([0,1] , [0,1] , '--')
# Configuração
ax.set_title(f"Curva ROC | AUC : {auc_train:.4f}")
ax.set_xlabel("FPR | 1 - Especificidade")
ax.set_ylabel("TPR | Recall")
ax.grid(visible=True, linestyle='--', linewidth=1)
plt.show()

# Outros indicadores
conf_matrix_model = pd.DataFrame(metrics.confusion_matrix(y_train, y_train_predict))
precisao_model = metrics.precision_score(y_train , y_train_predict)
recall_model = metrics.recall_score(y_train , y_train_predict)
espec_model = float(conf_matrix_model.iloc[0][0] / conf_matrix_model.iloc[0].sum())

# Encontrando o ponto otimo (maxima distancia TPR-FPR)
id_ponto_otimo = (curva_roc_auc[1] - curva_roc_auc[0]).argmax()
cutoff_otimo = curva_roc_auc[2][id_ponto_otimo]

# Gerar predicts com cutoff otimo
y_train_otimo = y_train_proba >= cutoff_otimo
conf_matrix_otimo = pd.DataFrame(metrics.confusion_matrix(y_train , y_train_otimo))
precisao_otimo = metrics.precision_score(y_train , y_train_otimo)
recall_otimo = metrics.recall_score(y_train , y_train_otimo)
espec_otimo = float(conf_matrix_otimo[0][0] / conf_matrix_otimo.iloc[0].sum())
auc_otimo = metrics.roc_auc_score(y_train , y_train_otimo)


# Plot da ROC com ponto otimo
fig , ax = plt.subplots()
ax.plot(curva_roc_auc[0] , curva_roc_auc[1] , '-')
# Plot da diagonal NAIVE
ax.plot([0,1] , [0,1] , '--')
# Plot do ponto otimo
ax.scatter(curva_roc_auc[0][id_ponto_otimo] , curva_roc_auc[1][id_ponto_otimo] , 
           marker='o' , color='red')
texto = f"Otimo em\n{cutoff_otimo:.3f}"
ax.annotate(text=texto , textcoords='offset points' , xytext=(0,30) ,
            xy= (curva_roc_auc[0][id_ponto_otimo] , curva_roc_auc[1][id_ponto_otimo]))
# Configuração
ax.set_title(f"Curva ROC | AUC : {auc_train:.4f}")
ax.set_xlabel("FPR | 1 - Especificidade")
ax.set_ylabel("TPR | Recall")
ax.grid(visible=True, linestyle='--', linewidth=1)
plt.show()

# Quadro comparativo de cenários

# Via matriz de Confusão
conf_matrix_comp = pd.concat([conf_matrix_model , conf_matrix_otimo] , 
                             axis=1 , join='inner')
conf_matrix_comp.columns=['M_False' , 'M_True' , 'O_False' , 'O_True']
conf_matrix_comp.index=['Falso' , 'Verdadeiro']
conf_matrix_comp

# Comparando as metricas           
dic_indicadores = {
    "auc" : [auc_train , auc_otimo] ,
    "precisao" : [precisao_model , precisao_otimo] ,
    "recall" : [recall_model , recall_otimo] ,
    "espec" : [espec_model , espec_otimo]
}
metrics_comp = pd.DataFrame(dic_indicadores , index=['Modelo' , 'Otimo'])
metrics_comp

# %%

# Usando ferramentas do ScikitPlot - Avaliação técnica do modelo

y_train_proba_full = meu_pipeline.predict_proba(X_train)

# Curva ROC | É necessário o PROBA INTEIRO e não só o y=1
skplot.metrics.plot_roc(y_train, y_train_proba_full)
skplot.metrics.roc_curve(y_train , y_train_proba)   # Valores do grafico

# Precision - Recall Curve | É necessário o PROBA INTEIRO e não só o y=1
skplot.metrics.plot_precision_recall_curve(y_train , y_train_proba_full)
skplot.metrics.precision_recall_curve(y_train , y_train_proba)  # Valores do grafico

# Lift CURVE | É necessário o PROBA INTEIRO e não só o y=1
skplot.metrics.plot_lift_curve(y_train , y_train_proba_full)

'''
Essa visualização mostra as curvas preditivas de ambas as classes versus um modelo Naive de 
classificação. A tendência é que no fim (com 100% da amostra ordenada), ambos os modelos 
cubram todas as observações, porém o LIFT mostra a vantagem do modelo ao apontar a clase à medida
em que avançamos nas predições.

No nosso caso, com 20% das maiores predições o modelo estava com um desempenho 2,5x maior que o
acaso em casos y=1 e ~1,5x maior em casos y=o
'''

# Cumulative Gain | É necessário o PROBA INTEIRO e não só o y=1
skplot.metrics.plot_cumulative_gain(y_train , y_train_proba_full)
skplot.metrics.cumulative_gain_curve(y_train , y_train_proba_full)

'''
Essa visualização mostra o quanto de cada classe foi capturado a medida em que avançamos no vetor
de predição ordenado

No nosso caso, com 20% das maiores predições o modelo havia capturado 50% dos casos de y=1 e 25% 
dos casos de y=0 e com 40% das maiores predições o modelo já havia capturado 80% dos casos de y=1 
e 58% dos casos y=0
'''