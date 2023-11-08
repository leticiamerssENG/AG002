import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# 2. Leitura dos dados utilizando a biblioteca Pandas
data = pd.read_csv("dados_tictac.csv")

# 3. Converter valores de acordo com o mapeamento
data = data.replace({'o': -1, 'b': 0, 'x': 1, 'negativo': -1, 'positivo': 1})

# 4. Escolher o modelo Perceptron
perceptron = Perceptron(random_state=42)

# 5. Separar o conjunto de dados em treinamento e teste
X = data.drop("resultado", axis=1)  # Substitua "target_column" pelo nome da coluna-alvo
y = data["resultado"]  # Substitua "target_column" pelo nome da coluna-alvo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo
perceptron.fit(X_train, y_train)

# 6. Avaliar o modelo
y_pred = perceptron.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy:.2f}')

# 7. Permitir ao usuário inserir dados arbitrários para classificação
while True:
    user_input = input("Insira dados (o, b, x, negativo, positivo) ou 'sair' para encerrar: ")
    if user_input == 'sair':
        break
    user_input = user_input.split()
    user_input = [x.replace('o', '-1').replace('b', '0').replace('x', '1').replace('negativo', '-1').replace('positivo', '1') for x in user_input]
    user_input = [int(x) for x in user_input]
    prediction = perceptron.predict([user_input])
    if prediction[0] == 1:
        print("Vitória de x: sim")
    else:
        print("Vitória de x: não")
