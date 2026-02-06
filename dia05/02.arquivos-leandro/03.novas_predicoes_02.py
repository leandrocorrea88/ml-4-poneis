'''
Aqui vamos carregar o modelo salvo em um dataset. Para fins didáticos vamos assumir que os
dados em dados_pontos.csv são novos dados para aplicarmos o modelo
As bibiotecas usadas no treinamento do modelo não precisam ser carregadas aqui, pois serão
referenciadas e carregadas automaticamente
'''
# %%

# Carregando referências
import pandas as pd

# %%

# Acessando o modelo
model = pd.read_pickle(filepath_or_buffer='model_rf.pkl')

# Agora o modelo está pronto para ser usado
model

# %%

# Acessando o novo dataset
df_novosdados = pd.read_csv('../../data/dados_pontos.csv' , 
                            sep=";")

# %%

# E chamando o modelo para prever usando os métodos em cada posição
X = df_novosdados[model['features']]
y = df_novosdados['flActive']

predict_proba = model['model'].predict_proba(X)

# Capturar o probra de y=1
probas = predict_proba[: , 1]

# E agregando na base de dados carregada
df_novosdados['prob_active'] = probas

# Para deixar a tabela mais compacta podemos segregar somente os usários com
# seus scores, ordenados
df_novosdados[['Name' , 'prob_active']].sort_values(by='prob_active' , ascending=False)

# %%
