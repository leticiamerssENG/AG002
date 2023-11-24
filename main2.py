import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

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
print(f"Tamanho x de treino: {X_train.shape}")
print(f"Tamanho x de teste: {X_test.shape}")
print(f"Tamanho y de treino: {y_train.shape}")
print(f"Tamanho y de teste: {y_test.shape}")

# Escolher um modelo de classificação, neste caso, KNN
model = KNeighborsClassifier(n_neighbors=5)  # Você pode ajustar o número de vizinhos conforme necessário

# Passo 5: Treinar o modelo escolhido usando 80% dos dados
model.fit(X_train, y_train)

# Passo 5: Avaliar o modelo usando os 20% restantes
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)

print()
print(report)
# Passo 6: Exibir métricas de avaliação
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy:.2f}')

# Passo 7: Permitir ao usuário inserir dados arbitrários para classificação
# Permitir ao usuário inserir dados arbitrários para classificação

while True:
    user_input = input("Insira os dados para classificação (separados por vírgula): ")
    user_data = [int(item) if item.isdigit() else mapping.get(item, 0) for item in user_input.split(',')]

    # Criar um DataFrame temporário com nomes de colunas fictícios
    user_df = pd.DataFrame([user_data], columns=[f'feature_{i}' for i in range(1, 10)])

    # Garantir que as colunas do DataFrame do usuário correspondam às do conjunto de treinamento
    missing_columns = set(X_train.columns) - set(user_df.columns)
    for col in missing_columns:
        user_df[col] = 0

    # Reordenar as colunas para corresponder à ordem do conjunto de treinamento
    user_df = user_df[X_train.columns]

    # Fazer a previsão usando o modelo
    prediction = model.predict(user_df)

    print(f"Resultado teste do usuário: {prediction}")

    if prediction[0] == 1:
        print("Com base no modelo, os dados inseridos constituem vitória de x.")
    else:
        print("Com base no modelo, os dados inseridos não constituem vitória de x.")
    break




"""Precision (Precisão):

Para -1 (negative): 1.00 (ou 100%) - Isso significa que, de todas as instâncias que o modelo classificou como "negative", todas realmente eram "negative".
Para 1 (positive): 0.99 (ou 99%) - Isso significa que, de todas as instâncias que o modelo classificou como "positive", 99% eram realmente "positive".
Recall (Revocação):

Para -1 (negative): 0.99 (ou 99%) - Isso significa que o modelo identificou corretamente 99% de todas as instâncias "negative" no conjunto de teste.
Para 1 (positive): 1.00 (ou 100%) - Isso significa que o modelo identificou corretamente todas as instâncias "positive" no conjunto de teste.
F1-score (Média harmônica de precision e recall):

Para -1 (negative): 0.99 (ou 99%)
Para 1 (positive): 1.00 (ou 100%)
Acurácia:

A precisão global do modelo é de 99%, o que significa que ele classifica corretamente 99% de todas as instâncias no conjunto de teste."""