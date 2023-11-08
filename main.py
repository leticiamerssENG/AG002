import pandas as pd

df = pd.read_csv('dados_tictac.csv', delimiter=',')


df['1'] = df['1'].replace({'o': -1, 'b': 0, 'x': 1})
df['2'] = df['2'].replace({'o': -1, 'b': 0, 'x': 1})
df['3'] = df['3'].replace({'o': -1, 'b': 0, 'x': 1})
df['4'] = df['4'].replace({'o': -1, 'b': 0, 'x': 1})
df['5'] = df['5'].replace({'o': -1, 'b': 0, 'x': 1})
df['6'] = df['6'].replace({'o': -1, 'b': 0, 'x': 1})
df['7'] = df['7'].replace({'o': -1, 'b': 0, 'x': 1})
df['8'] = df['8'].replace({'o': -1, 'b': 0, 'x': 1})
df['9'] = df['9'].replace({'o': -1, 'b': 0, 'x': 1})
df['resultado'] = df['resultado'].replace({'negativo': -1, 'positivo': 1})

