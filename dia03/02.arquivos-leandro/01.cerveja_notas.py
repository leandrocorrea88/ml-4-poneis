'''
Aplicação dos algoritmo de regressão linear sobre a base que contém dados de vários alunos apontando
duas caracteristicas:

- A quantidade de cervejas ingeridas
- A nota final no exame

Queremos prever a nota final no exame, então sabemos que

Target : Nota
Features : Qtde de cervejas

'''

# %%

# Importando os dados

import pandas as pd

df = pd.read_excel('../../data/dados_cerveja_nota.xlsx')
df

# %%

################################
# BLOCO 1 - Enxergando os dados
################################

'''
Temos uma série cujas notas podem variar no intervalo de 0 a 10 e a quantidade de cervejas
observadas variando de 1 a 9
'''

df['cerveja'].describe()
df['nota'].describe()

# Vamos criar um gráfico para visualizar o comportamento combinado das variáveis

import matplotlib.pyplot as plt

# Criamos uma figura vazia para configuração. No caso desse método temos uma tupla com dois
# retornos, sendo a primeira a FIGURA (como sendo a folha de papel) e AXES (que retorna o conteudo
# que estamos desenhando, ou os eixos/plots). Cada um deles tem seus proprios métodos, então podemos
# desempacotar nos dois objetos
fig , ax = plt.subplots()

## Agora vamos configurar o plot efetivamente

# Eixos e marcadores
ax.plot(df['cerveja'] , df['nota'] ,'o') # ou marker='o' , linestyle=None
# Linhas de grade
ax.grid(visible=True,linestyle='--')
# Configuração do eixo
ax.set_title("Relação Nota vs Cervejas",loc='left')
ax.set_xlabel("Qtde de cervejas")
ax.set_ylabel("Nota final")
ax.set_xlim(left=0 , right=11)
ax.set_ylim(bottom=0 , top=11)

# Exibir o resultado
plt.show()

# %%

######################################
# BLOCO 2 - Implementando Linear Model
######################################

from sklearn import linear_model

X_observado = df[['cerveja']]   # O argumento é uma MATRIZ/DATAFRAME
y_observado = df['nota']

# Implementar o modelo LinearRegression e passando os parametros
reg = linear_model.LinearRegression()
reg.fit(X=X_observado , y=y_observado)

# Obtendo o INTERCEPTO (Numero)
reg.intercept_

# Obtendo os COEFICIENTES (Lista)
reg.coef_

# Para prever os valores usamos o PREDICT, no nosso exercicio, com os valores
# da propria base de Qtde de Cervejas
X_deduplicado = X_observado.drop_duplicates()
y_previsto = reg.predict(X=X_deduplicado)

# %%

# Agora vamos visualizar os resultados

fig , ax = plt.subplots()

## Agora vamos configurar o plot efetivamente

# Eixos e marcadores (Plot dos DADOS OBSERVADOS)
ax.plot(df['cerveja'] , df['nota'] , 'o') # ou marker='o' , linestyle=None
# Eixos e marcadores (Plot da PREVISÃO)
ax.plot(X_deduplicado , y_previsto , marker=None) # porque agora queremos uma linha
# Linhas de grade
ax.grid(visible=True,linestyle='--')
# Configuração do eixo
titulo = "Relação Nota vs Cervejas - Reta ajustada"
titulo = titulo + '\n' + f"em y = {reg.intercept_:.2f} + ({reg.coef_[0]:.2f})x"
ax.set_title(titulo,loc='left')
ax.set_xlabel("Qtde de cervejas")
ax.set_ylabel("Nota final")
ax.set_xlim(left=0 , right=11)
ax.set_ylim(bottom=0 , top=11)

# Ajuster formato compacto
plt.tight_layout()
# Exibir o resultado
plt.show()


# %%

############################################
# BLOCO 3 - Implementando Arvore de Decisão
###########################################

from sklearn import tree

X_observado = df[['cerveja']]   # O argumento é uma MATRIZ/DATAFRAME
y_observado = df['nota']

# Definindo um hiperparametro de profundidade da arvore
arvore = tree.DecisionTreeRegressor(max_depth=2)
arvore.fit(X=X_observado , y=y_observado)

# Para prever os valores usamos o PREDICT, no nosso exercicio, com os valores
# da propria base de Qtde de Cervejas
X_deduplicado = X_observado.drop_duplicates()
y_previsto = arvore.predict(X=X_deduplicado)

# %%

# Agora vamos visualizar os resultados

fig , ax = plt.subplots()

## Agora vamos configurar o plot efetivamente

# Eixos e marcadores (Plot dos DADOS OBSERVADOS)
ax.plot(df['cerveja'] , df['nota'] , 'o') # ou marker='o' , linestyle=None
# Eixos e marcadores (Plot da PREVISÃO)
ax.plot(X_deduplicado , y_previsto , marker=None) # porque agora queremos uma linha
# Linhas de grade
ax.grid(visible=True,linestyle='--')
# Configuração do eixo
titulo = "Relação Nota vs Cervejas - Arvore de Decisao"
ax.set_title(titulo,loc='left')
ax.set_xlabel("Qtde de cervejas")
ax.set_ylabel("Nota final")
ax.set_xlim(left=0 , right=11)
ax.set_ylim(bottom=0 , top=11)

# Ajuster formato compacto
plt.tight_layout()
# Exibir o resultado
plt.show()

# %%
