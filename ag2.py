import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv('tic-tac-toe.csv')

data.replace({'o': -1, 'b': 0, 'x': 1, 'negativo': -1, 'positivo': 1}, inplace=True)

X = data.drop(['resultado'], axis=1)
y = data['resultado']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=50)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
class_report = classification_report(y_test, predictions)
print(f'Model Performance:')
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')

user_input = input('\nEntre com os valores da tabela: ')
user_data = [int(x) for x in user_input.split(',')]

if len(user_data) != X.shape[1]:
    print(f"Erro: Entrada de precisa ter 9 valores")
else:
    user_data = [user_data]

    user_prediction = model.predict(user_data)[0]

    print("\nUser prediction:", user_prediction)

    if user_prediction == 1:
        print('Vitoria de x.')
    else:
        print('Derrota de x ou empate.')
