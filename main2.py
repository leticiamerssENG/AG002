import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Passo 1: Baixar o conjunto de dados em formato CSV (supondo que o arquivo seja 'dados.csv')

# Passo 2: Fazer a leitura dos dados utilizando a biblioteca Pandas
data = pd.read_csv('dados_tictac.csv')

# Passo 3: Converter os valores de acordo com o mapeamento
mapping = {'o': -1, 'b': 0, 'x': 1, 'negativo': -1, 'positivo': 1}
data.replace(mapping, inplace=True)

# Separar os recursos (features) e o alvo (target)
X = data.drop('resultado', axis=1)  # Supondo que 'alvo' seja a coluna que contém a variável alvo
y = data['resultado']

# Passo 5: Separar o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escolher um modelo de classificação, neste caso, KNN
model = KNeighborsClassifier(n_neighbors=5)  # Você pode ajustar o número de vizinhos conforme necessário

# Passo 5: Treinar o modelo escolhido usando 80% dos dados
model.fit(X_train, y_train)

# Passo 5: Avaliar o modelo usando os 20% restantes
y_pred = model.predict(X_test)

# Passo 6: Exibir métricas de avaliação
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy:.2f}')

# Passo 7: Permitir ao usuário inserir dados arbitrários para classificação
while True:
    user_input = input("Insira os dados para classificação (separados por vírgula): ")
    user_data = [int(item) if item.isdigit() else mapping.get(item, 0) for item in user_input.split(',')]

    prediction = model.predict([user_data])
    if prediction[0] == 1:
        print("Com base no modelo, os dados inseridos constituem vitória de x.")
        break;
    else:
        print("Com base no modelo, os dados inseridos não constituem vitória de x.")
        break;
