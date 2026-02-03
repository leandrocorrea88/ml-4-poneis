''''
Nesse arquivo vamos TREINAR o nosso modelo para posterior aplicação

Resumo da ABT : cada linha representa um usuário em um momento específico do tempo, com estastísticas
considerando uma janela de 15 dias

Nosso alvo será a coluna flActive que representa o status de atividade 15 dias APÓS a data da
transação do usuário

'''

# %%

import pandas as pd

df = pd.read_csv('../../data/dados_pontos.csv', sep=";")
df

# %%

###############################################################
# A primeira coisa a fazer é DIVIDR a base entre TREINO e TESTE
###############################################################

from sklearn import model_selection

# Para as features vamos usar TODAS a partir de qtdeRecencia POS=3
features = df.columns[3:-1]
target = 'flActive'

# %%

# Separar as bases
X_train , X_test , y_train , y_test = model_selection.train_test_split(df[features] ,   # X
                                                                       df[target] ,     # y
                                                                       test_size=0.2,   # 20% para teste
                                                                       random_state=42) # Replicabilidade

# Vamos validar a proporção de verdadeiros em ambas as bases (Taxa de Resposta)
print("Tx de Ativos Treino:" , y_train.mean())
print("Tx de Ativos Teste:" , y_test.mean())

# Com média superior a 30% não podemos dizer que a base está BALANCEADA (isso seria algo proximo a
# 50%), mas podemos dizer que a resposta NÃO É UM EVENTO RARO

# %%

# Caso quisessemos manter uma media mais equilibrada poderiamos voltar ao split e adicionar um
# novo argumento STRATIFY, passando a variavel alvo como argumento

X_train , X_test , y_train , y_test = model_selection.train_test_split(df[features] ,       # X
                                                                       df[target] ,         # y
                                                                       test_size=0.2,       # 20% para teste
                                                                       random_state=42,     # Replicabilidade
                                                                       stratify=df[target]) # Equilibrar target

# Avaliando novamente
print("Tx de Ativos Treino:" , y_train.mean())
print("Tx de Ativos Teste:" , y_test.mean())

# %%

#########################################################################
# Agora podemos explorar os dados, SEMPRE OLHANDO PARA A BASE DE TREINO!
# A base de teste HERDARÁ TODAS AS DESCOBERTAS FEITAS NA BASE DE TREINO!
#########################################################################

# %%

# Procurar MISSINGS
X_train.isna().sum().T

'''
Aqui vemos que temos 810 ocorrências de MISSING na coluna avgRecorrencia.

O scikit não suporta valores faltantes para rodar os algoritmos.

INTERPRETAÇÃO do dado (natureza da variável): não conseguimos calcular a recorrência nos casos em 
que uma pessoa veio na live e não voltou mais. Imputar ZERO implica em dizer que o usuário tem 
acessado a live TODOS os dias, enquanto imputar um valor negativo faz o modelo pressupor que 
ELA NUNCA SAIU.

Esse é o unico dado faltante, então baseado na NATUREZA DO DADO 

    - Se não conseguimos calcular a recorência significa que ela não voltou
    - Recorrência Média alta significa que a pessoa DEMOROU demais a voltar
    - NOVOS usuários terão recorrencia NULA, mas tambem precisam de resposta do modelo

Algumas opções são

    - Imputar o MAIOR VALOR OBSERVADO
    - Imputar um valor MUITO ALTO (ex.: 9999)

'''

# Testando o maior valor e devolvendo a base original e imputando o MESMO valor para o TESTE
input_avgRecorrencia = X_train['avgRecorrencia'].max()
X_train['avgRecorrencia'] = X_train['avgRecorrencia'].fillna(input_avgRecorrencia)
X_test['avgRecorrencia'] = X_test['avgRecorrencia'].fillna(input_avgRecorrencia)

# %%

# Exploratória
X_train.describe().T

# Como separamos os X do y, vamos concatena-los para fazer uma exploratória pela resposta
df_train = pd.concat([y_train , X_train] , axis=1)
df_train

# Medias por classe
df_train.groupby('flActive').describe().T
df_train.groupby('flActive')[df_train.columns].mean().T
# %%

####################################
# Implementando os Testes de Modelo
####################################

# %%

# Testar ARVORE DE DECISÃO

from sklearn import tree
from sklearn import metrics

# Treinando o modelo
arvore = tree.DecisionTreeClassifier(max_depth=7,           # Profundidade (Quantas perguntas?)
                                     random_state=42,       # Replicabilidade
                                     min_samples_leaf=50    # Amostras em cada nó (quantos exemplos?)
                                     )
arvore.fit(X_train , y_train)

# Prevendo na propria base
tree_pred_train = arvore.predict(X_train)
tree_acc_train = metrics.accuracy_score(y_train , tree_pred_train)
print(f"Árvore Train ACC: {tree_acc_train:.3f}")

# Prevendo na base de teste
tree_pred_test = arvore.predict(X_test)
tree_acc_test = metrics.accuracy_score(y_test , tree_pred_test)
print(f"Árvore Test ACC: {tree_acc_test:.3f}")

'''
O Modelo aprendeu com os dados? 

    - Sim, dadas as ACURÁCIAS de ambos (Treino = 0.832 e Teste=0.803)

O modelo pode ser generalizado para uso (overfit)?
    
    (a) Diferença baixa entre as acurácias de Treino [0.832] e Teste [0.803]
    
    (b) Se o modelo assume que NINGUEM volta, significa que a resposta será sempre y=0 e isso 
    significa que acertaremos 67,3% das vezes (considerando que na base de TESTE a média de 
    resposta [Ativos] é de 32,6%) - Esse cenário é uma visão inicial, desconsiderando ainda 
    todas as probabilidades condicionais das covariáveis (aqui "chutamos com o que sabemos", em
    outras palavras uma abordagem "a priori")

    (c) O modelo ATUAL, na base de teste, tem acurácia de 0.814. Isso é maior que a acurácia do
    modelo a priori, nos dando boa segurança sobre a GENERALIZAÇÃO do modelo


'''
# %%

# Testando os PROBAS

# Prevendo na propria base
tree_proba_train = arvore.predict_proba(X_train)[: , 1]
tree_auc_train = metrics.roc_auc_score(y_train , tree_proba_train)
print(f"Árvore Train AUC: {tree_auc_train:.3f}")

# Prevendo na base de teste
tree_proba_test = arvore.predict_proba(X_test)[: , 1]
tree_auc_test = metrics.roc_auc_score(y_test , tree_proba_test)
print(f"Árvore Test AUC: {tree_auc_test:.3f}")

'''
Ao testar as distribuições de probabilidade os valores continuam muito proximos, o que indica que
ainda temos boa generalização do modelo

Se testarmos o modelo com outros parametros vemos um movimento de melhorar o desempenho em Treino,
mas uma tendência a perda de performance em Teste

    [max_depth = 5]
                ACC         AUC
    Train       0.832       0.871
    Test        0.803       0.848

    [max_depth = 6]
                ACC         AUC
    Train       0.848       0.894   <- Aumenta
    Test        0.792       0.805   <- Reduz

    [max_depth = 6 , min_samples_leaf = 50]
                ACC         AUC
    Train       0.814       0.862   <- Reduz
    Test        0.801       0.838   <- Mantem

    [max_depth = 7 , min_samples_leaf = 50]
                ACC         AUC
    Train       0.814       0.863   <- Aumenta
    Test        0.801       0.840   <- Reduz

'''

# %%
