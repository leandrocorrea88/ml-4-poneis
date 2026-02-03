'''
Aqui vamos aplicar as métricas de avaliação dos modelos.

Os códigos de implementação do modelo serão replicados, mas todo o racional teórico está no
arquivo anterior.

Nesse vamos focar em avaliar a eficácia de cada modelo apenas
'''

# %%

# Lendo os dados

import pandas as pd

df = pd.read_excel('../../data/dados_cerveja_nota.xlsx')
df

# %%

# Preparar os dados para os modelos

# criar a variavel binária
df['aprovado'] = df['nota'] >= 5

# Fatiar as variaveis | Matriz para Features
target ='aprovado'
features = ['cerveja']

# %%

###########################################
# Aplicando e avaliando REGRESSÃO LOGÍSTICA
###########################################

# %%

# PARTE 1 - Aplicando o Modelo

from sklearn import linear_model

# Ensinar o modelo
log = linear_model.LogisticRegression(penalty=None,
                                      fit_intercept=True)
log.fit(X=df[features] , y=df[target])

# %%

# PARTE 2 - Avaliando o Modelo

from sklearn import metrics

# Prever novos valores
reg_predict = log.predict(df[features])

# 1. MATRIZ DE CONFUSÃO : Detecta os TIPOS DE ERRO 
# (Falsos positivos e Falsos negativos)
reg_conf = metrics.confusion_matrix(y_true=df[target], y_pred=reg_predict)

# Vamos transformar a matriz em um DATAFRAME
reg_df_conf = pd.DataFrame(data = reg_conf ,
                           index=['False', 'True'] ,   # OBSERVADOS
                           columns=['False', 'True'])  # PREVISTOS

# Podemos transformar os absolutos da matriz de confusão em relativos
observacoes = df.shape[0]
reg_df_conf_perc = pd.DataFrame(data = reg_conf / observacoes ,
                                index=['False', 'True'] ,   # OBSERVADOS
                                columns=['False', 'True'])  # PREVISTOS

# ACURÁCIA : PROPORÇÃO de ACERTOS NO TOTAL
reg_acc = metrics.accuracy_score(y_true=df[target] ,  y_pred=reg_predict)

# PRECISÃO : QUANTO O MODELO ACERTOU NOS POSITIVOS
reg_prec = metrics.precision_score(y_true=df[target], y_pred=reg_predict)

# RECALL/SENSIBILIDADE : Taxa de captura dos POSITIVOS
reg_recall = metrics.recall_score(y_true=df[target], y_pred=reg_predict)

# ESPECIFICIDADE : Taxa de captura do NEGATIVOS. Nativamente não temos esse método
# então podemos calcular manualmente obtendo as parcelas da matriz de confusão e
# implementando a fórmula

# O método RAVEL() retorna as 4 variaveis da matriz de confusão em uma ordem especifica,
# sendo (1,1) Verdadeiro Negativo , (1,2) Falso Positivo , (2,1) Falso Negativo e 
# (2,2) Verdadeiro Positivo. O mesmo layout que exploramos linhas acima
vn , fp , fn , vp = reg_conf.ravel()
reg_espec = vn / ( vn + fp)

texto = "Desempenho da Regressão Linear"
texto = texto + "\n" + f"Acurácia : {reg_acc:.3f}"
texto = texto + "\n" + f"Precisão : {reg_prec:.3f}"
texto = texto + "\n" + f"Recall : {reg_recall:.3f}"
texto = texto + "\n" + f"Especificidade : {reg_espec:.3f}"
print(texto)

# %%

#########################################
# Aplicando e avaliando ARVORE DE DECISÃO
#########################################

# %%

# PARTE 1 - Aplicando o Modelo

from sklearn import tree

# Ensinar o modelo
arvore = tree.DecisionTreeClassifier(max_depth=2)
arvore.fit(X=df[features] , y=df[target])

# %%

# PARTE 2 - Avaliando o Modelo

# Prever novos valores 
# Se usamos o REGRESSOR, temos aqui as PROBABILIDADES
# Se usamos o CLASSIFIER, temos aqui os BINÁRIOS
arv_predict = arvore.predict(X=df[features])

# MATRIZ DE CONFUSÃO
arv_conf = metrics.confusion_matrix(y_true=df[target], y_pred=arv_predict)

arv_df_conf = pd.DataFrame(data=arv_conf,
                           index=['False', 'True'],
                           columns=['False', 'True'])

# ACURACIA
arv_acc = metrics.accuracy_score(y_true=df[target], y_pred=arv_predict)

# PRECISÃO
arv_prec = metrics.precision_score(y_true=df[target], y_pred=arv_predict)

# RECALL
arv_recall = metrics.recall_score(y_true=df[target], y_pred=arv_predict)

# ESPECIFICIDADE
vn , fp , fn , vp = arv_conf.ravel()
arv_espec = vn / ( vn + fp)

texto = "Desempenho da Árvore de Decisão"
texto = texto + "\n" + f"Acurácia : {arv_acc:.3f}"
texto = texto + "\n" + f"Precisão : {arv_prec:.3f}"
texto = texto + "\n" + f"Recall : {arv_recall:.3f}"
texto = texto + "\n" + f"Especificidade : {arv_espec:.3f}"

print(texto)

# %%

###################################
# Aplicando e avaliando NAIVE BAYES
###################################

# %%

# PARTE 1 - Aplicando o Modelo

from sklearn import naive_bayes
from sklearn import metrics

nb = naive_bayes.GaussianNB()
nb.fit(X = df[features] , y=df[target])

# %%

# PARTE 2 - Avaliando o Modelo

# Prever novos valores 
nb_predict = nb.predict(X=df[features])

# MATRIZ DE CONFUSÃO
nb_conf = metrics.confusion_matrix(y_true=df[target], y_pred=nb_predict)

# ACURACIA
nb_acc = metrics.accuracy_score(y_true=df[target] , y_pred=nb_predict)

# PRECISÃO
nb_prec = metrics.precision_score(y_true=df[target] , y_pred=nb_predict)

# RECALL
nb_recall = metrics.recall_score(y_true=df[target] , y_pred=nb_predict)

# ESPECIFICIDADE
vn , fp , fn , vp = n}b_conf.ravel()
nb_espec = vn / ( vn + fp)

texto = "Desempenho de Naive Bayes"
texto = texto + "\n" + f"Acurácia : {nb_acc:.3f}"
texto = texto + "\n" + f"Precisão : {nb_prec:.3f}"
texto = texto + "\n" + f"Recall : {nb_recall:.3f}"
texto = texto + "\n" + f"Especificidade : {nb_espec:.3f}"
print(texto)

# %%

###############################
# Validando curva ROC e ROC AUC
###############################

'''
Já vimos as métricas a partir dos dados gerados pelo modelo. Normalmente em Scikit o ponto de corte
para considerar um PROBA em POSITIVO gira em torno de 0.5, mas alterar esse indicador pode fazer
com que minha matriz de confusão e demais métricas sejam alteradas.

No fim, o que importa não é o quão bom é um modelo, mas o quanto de resultado ele pode gerar (receita, 
despesa, margem, risco, reputação...). Então podemos combinar algumas métricas em um visual que
compara a mudança de patamar das métricas a medida em que mudo meu ponto de corte, que é conhecida
como Curva ROC

Primeiro vamos construi-la manualmente e depois usar as funções disponíveis no scikit pra isso.

Para facilitar vamos fazer usando somente os resultados do modelo Naive Bayes
'''
# %%

# Modelo 1 - Construção manual

# Primeiro geramos os resultados dos PROBAS
nb_probas = nb.predict_proba(X=df[features])

# Como retorno temos um array de listas com as probabilidades de y=0 e y=1
# como pares. Como queremos a probabilidade de y=1, então pegamos somente a segunda coluna
# inteira
nb_probas = nb.predict_proba(X=df[features])[: , 1]

# Vamos começar com um valor baixo de linha de corte para simular a transformaçao
# dos PROBAS e PREDICTS
corte = 0.1
nb_predict = nb_probas >= corte

# E aplicar as metricas
nb_acc = metrics.accuracy_score(y_true=df[target] , y_pred=nb_predict)      # ACURACIA
nb_prec = metrics.precision_score(y_true=df[target] , y_pred=nb_predict)    # PRECISÃO
nb_recall = metrics.recall_score(y_true=df[target] , y_pred=nb_predict)     # RECALL
vn , fp , fn , vp = nb_conf.ravel()
nb_espec = vn / ( vn + fp)                                                  # ESPECIFICIDADE

# E plotar os resultados em um texto simples
texto = "Desempenho de Naive Bayes"
texto = texto + "\n" + f"Acurácia : {nb_acc:.3f}"
texto = texto + "\n" + f"Precisão : {nb_prec:.3f}"
texto = texto + "\n" + f"Recall : {nb_recall:.3f}"
texto = texto + "\n" + f"Especificidade : {nb_espec:.3f}"
print(texto)

# Agora vamos transformar isso em iterações para enxergar em um DF
lst_cenarios = []

for i in range(0 , 11 , 1):
    # Dividimos por 10 porque range não suporta float
    corte = i / 10
    
    # Recalcular a confusão
    nb_predict = nb_probas >= corte
    nb_conf = metrics.confusion_matrix(y_true=df[target], y_pred=nb_predict)
    vn , fp , fn , vp = nb_conf.ravel()

    # Adicionar os elementos dos cenarios
    item = (
        round(corte , 3) ,
        round( metrics.accuracy_score(y_true=df[target] , y_pred=nb_predict) , 3 ) ,
        round( metrics.precision_score(y_true=df[target] , y_pred=nb_predict) , 3 ) ,
        round( metrics.recall_score(y_true=df[target] , y_pred=nb_predict) , 3 ) ,
        round( (vn / ( vn + fp)) , 3 )
    )
    lst_cenarios.append(item)

# Criando e populando o DF
df_cenarios = pd.DataFrame(data=lst_cenarios ,
                           index=None ,
                           columns=['Corte', 'Acuracia' , 'Precisão' , 'Recall', 'Especificidade'])

# Vamos criar uma coluna com o complementar da especificidade
df_cenarios['tpr'] = df_cenarios['Recall']                  # True Positive Rate
df_cenarios['fpr'] = 1 - df_cenarios['Especificidade']      # False Positive Rate

# %%

# Plotando o grafico
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# A curva ROC mostra a relação entre TPR e FPR, para AVALIAÇÃO DO MODELO e definição
# dos tradeoffs (DIAGNÓSTICO)

# Adicionar os pontos do DF
ax.scatter(df_cenarios['fpr'] , df_cenarios['tpr'] , marker=None , zorder=3)
# Adicionar a linha de referencia onde Y = X 
# (que é um modelo simples, equivalente a jogar a moeda)
ax.plot([0,1] , [0, 1] , linestyle='--')
# Adicionar a linha da curva ROC (escadinha)
ax.plot(df_cenarios['fpr'] , df_cenarios['tpr'] , drawstyle='steps-post' , label='ROC' , zorder=2)
# Adicionar area abaixo da curva
ax.fill_between(df_cenarios['fpr'], df_cenarios['tpr'], step='post' , alpha=0.2, zorder=1)

ax.set_xlabel("FPR (1 - Especificidade)")
ax.set_ylabel("TPR (Recall)")
ax.set_title("ROC manual por cortes")
ax.grid(True , linestyle='--')
plt.show()

# %%

# Outra versão da curva pode mostrar a relação entre CUTOFF e alguma métrica que seja mais
# importante para o negócio [Precisão, Recall, Especificidade] (DECISÃO DE NEGÓCIO)

fig, ax = plt.subplots()

# Adicionar os pontos do DF
ax.scatter(df_cenarios['Corte'] , df_cenarios['Precisão'] , marker=None , zorder=3)
# Adicionar a linha de referencia onde Y = X 
# (que é um modelo simples, equivalente a jogar a moeda)
ax.plot([0,1] , [0, 1] , linestyle='--')
# Adicionar a linha da curva ROC (escadinha)
ax.plot(df_cenarios['Corte'] , df_cenarios['Precisão'] , drawstyle='steps-post' , label='ROC' , zorder=2)
# Adicionar area abaixo da curva
ax.fill_between(df_cenarios['Corte'], df_cenarios['Precisão'], step='post' , alpha=0.2, zorder=1)

ax.set_xlabel("Cortes (Cut-offs)")
ax.set_ylabel("Precisão")
ax.set_title("Curva Corte vs Precisão")
ax.grid(True , linestyle='--')
plt.show()

# %%

# Modelo 2 - usando as funções SciKit

'''
Uma grande diferença entre montar a tabela manual e usar o método ROC_CURVE é que o método vai
estimar os pontos de corte onde podemos ter diferença real em Recall, nos dando a possibilidade
de pesquisar em que ponto da curva nosso corte atual se encontra
'''

# Configuração do modelo base | Obter os PROBAS para y=1
nb_probas = nb.predict_proba(df[features])[:,1]

# Aplicação do método ROC_CURVE que retorna 3 resultados, sendo (1) a taxa de verdadeiros positivos 
# [TPR], (2) a Taxa de Falsos POsitivos [FPR] e os limiares que alteram esses valores [Thresholds]
roc_curve = metrics.roc_curve(y_true=df[target] , y_score=nb_probas)

# Obter AREA ABAIXO DA CURVA
roc_auc = metrics.roc_auc_score(y_true=df[target] , y_score=nb_probas)

# Agora podemos capturar esses elementos para traçar a curva usando TPR e FPR

fig, ax = plt.subplots()

# Linha de referência (Modelo básico)
ax.plot([0,1] , [0,1] , linestyle='--' , color='gray' , zorder=1)
# Linha TPR vs FPR (estilo escadinha. Sem o steps-post teriamos uma curva inclinada)
ax.plot(roc_curve[0] , roc_curve[1] , linestyle='-' , color='black' , 
        label='ROC' , drawstyle='steps-post' , zorder=2)
# Preencher a area abaixo da curva
ax.fill_between(roc_curve[0] , roc_curve[1] , step='post' , alpha=0.3 , zorder = 3 )
# Configurar rotulos
ax.set_title(f"Curva ROC usando Método ROC_Curve:{roc_auc:.3f}")
ax.set_xlabel("FPR (False Positive Rate | 1-Especificidade)")
ax.set_ylabel("TPR (True Positive Rate | Recall)")
ax.grid(True, linestyle='--')

plt.show()

# %%

# Descobrindo o melhor cutoff

'''
O método classico para calcular o melhor ponto de corte, do ponto de vista de DIAGNOSTICO DE
MODELO é o chamado ÍNDICE DE YOUDEN que estima

    - Qual o ponto mais distante da curva aleatória, que matematicamente falando é
    - O ponto onde há maior distância entre TPR e FPR

Existe o F1-Score que combina precisão e recall, mas é uma medida PONTUAL e exige um CORTE prévio,
enquanto a curva ROC dá uma ideia de DISTRIBUIÇÃO

O objetivo de todo modelo é Maximizar a área sob a curva, mas para calcular o ponto otimo podemos
capturar a primeira ocorrencia da maior diferença entre TPR e FPR

IMPORTANTE: Para discutir cutoff o modelo precisa estar pronto, ou seja

    - Acima da diagonal aleatória (modelo "bobo")
    - Área de pelo menos 0.7 (0.6 a 0.7 ainda pode ser considerado fraco)
    - Quando não se tem modelo nenhum, qualquer modelo melhor que a sorte já é algo a considerar

'''

# Procurando a posição no vetor de cortes a partir de TPR e FPR : Posição 4
id_ponto_otimo = (roc_curve[1] - roc_curve[0]).argmax()

# aplicando o valor para descobrir : 0,922
roc_curve[2][id_ponto_otimo]

# Caso tivessemos um corte determinado e quisessemos saber a posição dele na nossa curva
# precisamos encontrar um ponto MAIS PROXIMO POSSIVEL desse valor, ou seja, encontramos
# o ponto onde há menor distancia (em módulo) entre o nosso corte e os niveis definidos na ROC
corte_atual = 0.3
id_ponto_atual = abs(roc_curve[2] - corte_atual).argmin()

# Agora devolvemos esse índice ao vetor : 0,626
roc_curve[2][id_ponto_atual]

# Tambem podemos plotar esses valores no grafico como um ponto

fig, ax = plt.subplots()

# Linha de referência (Modelo básico)
ax.plot([0,1] , [0,1] , linestyle='--' , color='gray' , zorder=1)
# Linha FPR vs TPR (estilo escadinha. Sem o steps-post teriamos uma curva inclinada)
ax.plot(roc_curve[0] , roc_curve[1] , linestyle='-' , color='black' , 
        label='ROC' , drawstyle='steps-post' , zorder=2)
# Preencher a area abaixo da curva
ax.fill_between(roc_curve[0] , roc_curve[1] , step='post' , alpha=0.3 , zorder = 3 )

# Plotar o PONTO OTIMO e anotar o valor
ax.scatter(roc_curve[0][id_ponto_otimo] , roc_curve[1][id_ponto_otimo] , marker='o' , color='red')
ax.annotate(text=f"Ótimo em\n{roc_curve[2][id_ponto_otimo]:.3f}" , 
            xy=(roc_curve[0][id_ponto_otimo] , roc_curve[1][id_ponto_otimo]) ,
            textcoords='offset points' , xytext=(0,10))

# Plotar o PONTO ATUAL e anotar o valor
ax.scatter(roc_curve[0][id_ponto_atual] , roc_curve[1][id_ponto_atual] , marker='o' , color='red')
ax.annotate(text=f"Atual em\n{roc_curve[2][id_ponto_atual]:.3f}" , 
            xy=(roc_curve[0][id_ponto_atual] , roc_curve[1][id_ponto_atual]) ,
            textcoords='offset points' , xytext=(0,-30))

# Configurar rotulos
ax.set_title("Curva ROC usando Método ROC_Curve")
ax.set_xlabel("FPR (False Positive Rate | 1-Especificidade)")
ax.set_ylabel("TPR (True Positive Rate | Recall)")
ax.grid(True, linestyle='--')

plt.show()

# %%
