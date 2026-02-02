'''
Aplicação dos algoritmo de regressão linear sobre a base que contém dados de vários alunos apontando
duas caracteristicas:

- A quantidade de cervejas ingeridas
- A nota final no exame, transformada em classes (aprovado/reprovado)

Queremos prever a PROBABILIDADE de ser APROVADO, então sabemos que

Target : Aprovado (se Nota >= 5)
Features : Qtde de cervejas

'''


# %%

import pandas as pd

df = pd.read_excel('../../data/dados_cerveja_nota.xlsx')
df
# %%

###################################
# BLOCO 1 - Transformando o Target
###################################

# Vamos criar uma coluna com a nossa regra de negócio para rotular o resultado

df["Aprovado"] = df['nota'] >= 5
df

# E vamos visualizar essa distribuição graficamente

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# Plotar a serie
ax.plot(df['cerveja'], df['Aprovado'] , 'o')

# Plotar a linha de decisão
ax.hlines(y=0.5 , xmin=0 , xmax=11 ,
          colors='black' , linestyle='--', linewidth=2)

# Configurar a serie
ax.grid(visible=True, linestyle="--")
ax.set_title("Relação Status Aprovação vs Cervejas")
ax.set_ylim(bottom=-0.1 , top=1.1)
ax.set_xlim(left=0 , right=11)
ax.set_xlabel("Qtde de cervejas")
ax.set_ylabel("Resultado Final")

# Retirar as grades superior e direita da area de plotagem
#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)

plt.show()
# %%

##############################################
# BLOCO 2 - Implementando Regressão Logistica
##############################################

'''
Diferente do modelo de regressão linear que trabalha com a minimização da função
de minimos quadradoo, a Regressão Logistica a função de erro é chamada de
LOG LOSS e o modelo tambem busca sua minimização
'''


# %%

# Agora vamos visualizar os resultados

fig, ax = plt.subplots()

# Plotar a serie
ax.plot(df['cerveja'], df['Aprovado'] , 'o')

# Plotar a linha de decisão
ax.hlines(y=0.5 , xmin=0 , xmax=11 ,
          colors='black' , linestyle='--', linewidth=2)

# Plotar a curva ajustada

# Configurar a serie
ax.grid(visible=True, linestyle="--")
ax.set_title("Relação Status Aprovação vs Cervejas")
ax.set_ylim(bottom=-0.1 , top=1.1)
ax.set_xlim(left=0 , right=11)
ax.set_xlabel("Qtde de cervejas")
ax.set_ylabel("Resultado Final")

# Ajuster formato compacto
plt.tight_layout()
# Exibir o resultado
plt.show()

# %%

############################################
# BLOCO 3 - Implementando Arvore de Decisão
###########################################

'''

Para medir a pureza podemos usar algumas métricas: 

1. Índice de Gini : calculado pela dispersão dos valores dentro de cada nó. Quanto mais 
proximo de 0, mais puro é o nó

2. Entropia : calculada pela proporção do evento dentro de cada nó, tambem tendendo a 0 quanto 
mais puro é o nó analisado

Idealmente testa-se ambas as métricas para identificar a eficácia do modelo através das 
combinações dos resultados

'''


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

#####################################
# BLOCO 4 - Implementando Naive Bayes
#####################################

'''
Naive Bayes é um modelo baseado em probabilidade que infere o futuro a partir de
valores previamente observados. É fundamentado no teorema de Bayes que usa a relação
de probabilidade condicional P(y|X) ou seja probabilidade de y dado um conjunto de X

P(y|X) = [ P(y) . P(X|y) ] / P(X)

Exemplo: 

Qual a probabilidade de uma pessoa ter diabetes dado 

    (a) Histórico Familias [Sim/Não na família]
    (b) Acima do peso [Sim/Não está em sobrepeso]
    (c) Atividade fisica [Sim/Não pratica]

Nesse modelo, podemos definir um cenário de teste:

    y =1    : tem diabetes
    x1 = 1  : tem historico familiar
    x2 = 0  : Não está acima do peso
    x3 = 0  : Não pratica atividade física

Então, se extrapolarmos o modelo para a população brasileira temos que:

    P(y|X)  : Probabilidade de ter diabetes dados os fatores analisados
    P(y)    : Probabilidade de ter diabetes AO ACASO (ex. % na população brasileira)
    P(X|y)  : Probabilidade de (x1=1,x2=0. x3=0) E TER TIDO DIABETES (no caso, testa-se 
            cada X dado y e multiplica-se os valores)
    P(X)    : Probabilidade de (x1=1,x2=0. x3=0), APENAS

A decisão de classificação de Naive Bayes gira em torno da probabilidade P(y=1|X) / P(y=0|X)
ser MAIOR ou IGUAL A 1. Em ambas as formas o denominador das razões é o mesmo em P(X), portanto
ao saber que

    P(y=1|X) / P(y=0|X) >= 1 , temos que o valor previsto é = 1

Em outras palavras, se a probabilidade de um evento ACONTECER, DADOS OS FATORES for PELO MENOS
igual à probabilidade desse evento NÃO ACONTECER, DADOS OS MESMOS FATORES, inferimos que o valor
final será a OCORRÊNCIA DO EVENTO.

No nosso caso, não temos a base da população então usaamos as % OBSERVADAS como uma forma de 
inferir e atualizar essa proporção (já que não sabemos os dados populacionais, o mais próximoq que
temos disso é a amostra. Ou seja, a amostra é O QUE SABEMOS).

A medida em que a amostra cresce os dados tendem a refletir o comportamento da população. 
A interpretação de todos os elementos se mantém, apenas focando nas quantidades OBSERVADAS 
em nosso ESPAÇO AMOSTRAL.

É um algoritmo que atualiza os valores de probabilidade a medida em que crescemos o volume de
informações que temos (adicionar novas variáveis ou observações). E dadas as premissas do modelo
já podemos estabelecer que combinações não observadas seriam encaradas como raras ou inexistentes

Extrapolando o conceito do modelo, Naive Bayes calcula a probabilidade de ocorrência do evento dadas
as DISTRIBUIÇÕES de cada variável aleatoria.

    > No nosso caso, assumimos todas as variaveis como BINÁRIAS, então a probabilidade de P(y=1|x1=0)
    segue uma BERNOULLI.

    > Para casos de uma variavel aleatoria continua assume-se distribuçição NORMAL, então teremos que
    para cada valor de y (0 ou 1) temos uma distribuição normal com MEDIA e DISPERSÃO especificas. Com
    isso, a cada novo dado adicionado verifica-se a PROBABILIDADE de aquele ponto estar na distribuição
    y=1 ou y=0

'''