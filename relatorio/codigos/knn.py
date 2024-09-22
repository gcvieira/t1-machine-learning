import math
from collections import Counter

# Função para calcular a distância Euclidiana entre dois pontos
def euclidean_distance(point1, point2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

# Implementação do algoritmo KNN
class KNN:
    def __init__(self, k=3):
        self.k = k
        self.data = None
        self.labels = None

    # Função para treinar o modelo com os dados de treino
    def fit(self, X_train, y_train):
        self.data = X_train
        self.labels = y_train

    # Função para prever o rótulo (classificação)
    def predict(self, X_test):
        predictions = []
        for test_point in X_test:
            # Calcular a distância de todos os pontos de treino
            distances = [(euclidean_distance(test_point, train_point), label) 
                         for train_point, label in zip(self.data, self.labels)]
            # Ordenar as distâncias e pegar os k vizinhos mais próximos
            sorted_distances = sorted(distances)[:self.k]
            # Coletar os rótulos dos vizinhos mais próximos
            k_nearest_labels = [label for _, label in sorted_distances]
            # Votação majoritária
            most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common_label)
        return predictions

# Exemplo de uso
if __name__ == "__main__":
    # Dados de treino (4 amostras de 2 features)
    X_train = [[1, 2], [2, 3], [3, 4], [5, 6]]
    y_train = [0, 0, 1, 1]  # Rótulos correspondentes

    # Dados de teste (1 amostra de 2 features)
    X_test = [[1.5, 2.5]]

    # Instancia o modelo KNN
    knn = KNN(k=3)
    
    # Treina o modelo
    knn.fit(X_train, y_train)
    
    # Faz a previsão
    prediction = knn.predict(X_test)
    
    print("Predição:", prediction)