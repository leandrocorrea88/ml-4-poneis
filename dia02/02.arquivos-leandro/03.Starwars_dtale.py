
# %%

# Analisando oo Case de Star Wars usando D-TALE que é uma biblioteca para execução de Análise
# Exploratória Automatizada (EDA)

import pandas as pd

df = pd.read_parquet('../../data/dados_clones.parquet')
df

# %%

import dtale

# Primeiro carregamos o dataset para o objeto
s = dtale.show(df)

# Agora abrimos o browser para explorar
s.open_browser()
# %%

'''

Documentação D-Tale

Repositorio : https://github.com/man-group/dtale

Outros pacotes automatizados

https://www.nb-data.com/p/python-packages-for-automated-eda

'''
