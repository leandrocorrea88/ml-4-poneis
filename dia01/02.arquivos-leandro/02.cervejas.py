
# %%
import pandas as pd

df = pd.read_excel("../../data/dados_cerveja.xlsx")
df
# %%

# Ao abrir o arquivo vemos que 3 features estão em formato de texto
# (copo , espuma , cor) e o scikit precisa que sejam NUMEROS, então
# a primeira etapa é converter os valores para poder consumir no modelo

# Desconsideramos o Id porque ele não tem valor de preditor
features = ['temperatura' , 'copo' , 'espuma' , 'cor']
target = 'classe'

X = df[features]
y = df[target]

# Convertemos as features (formato não otimizado ainda) ...

X = X.replace({
    "mud" : 1 , "pint" : 0 ,
    "sim" : 1 , "não" : 0 ,
    "escura" : 1 , "clara" : 0
})

# %%

# ... e podemos treinar a árvore ...

from sklearn import tree

arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(X=X , y=y)

# %%

# ... e plotar os resultados

import matplotlib.pyplot as plt

plt.figure(dpi=1200)

plot = tree.plot_tree(decision_tree=arvore , 
                      class_names=arvore.classes_ ,
                      feature_names=features ,
                      filled=True)

plt.savefig('02a. Arvore Cerveja.png')

# %%

arvore.classes_

# Nessa arvore os nós vão ficando mais fortes a medida em que os nós vão ficando mais
# puros

# O primeiro nó vem classificado como PILSEN pelo fato de ser a CLASSE COM MAIOR QUANTIDADE
# de OBSERVAÇÕES. O resultado [3 , 5 , 4] é a lista de elementos de cada classe, de maneira
# respectiva com o resultado de arvore.classes_

# No segundo nivel, seguino pelo caminho do mud (falso para copo=0 [pint]) já temos um nó que
# não tem mais PALE-ALE ([0 , 5 , 2]) então o nó vai ficando mais puro

# %%

# Vamos testar as predições, lembrando da estrutura
#   features = ['temperatura' , 'copo' , 'espuma' , 'cor']
# e da conversão
#   X = X.replace({
#       "mud" : 1 , "pint" : 0 ,
#       "sim" : 1 , "não" : 0 ,
#       "escura" : 1 , "clara" : 0
#   })

proba = arvore.predict_proba([[-5 , 1 , 0 , 1]])
pd.Series(data=proba[0] , index=arvore.classes_)

# Caso usemos valores diferentes de temperatura que não sejam -1 ou -5 teremos uma
# probabilidade de previsão, porém a arvore definiu o ponto de corte em -3, então 
# esse será a linha de corte

# %%
