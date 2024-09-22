import numpy as np

class NaiveBayes:
    def fit(self, X_train, y_train):
        # Separar por classe
        self.classes = np.unique(y_train)
        self.mean = {}
        self.var = {}
        self.priors = {}
        
        # Para cada classe
        for c in self.classes:
            X_c = X_train[y_train == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / X_train.shape[0]

    # Função para calcular a densidade de probabilidade Gaussiana
    def gaussian_pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-(x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
    # Função para calcular a probabilidade posterior
    def predict(self, X_test):
        y_pred = [self._predict(x) for x in X_test]
        return np.array(y_pred)
    
    # Função interna que calcula a probabilidade posterior para um exemplo
    def _predict(self, x):
        posteriors = []
        
        for c in self.classes:
            prior = np.log(self.priors[c])  # Probabilidade a priori
            conditional = np.sum(np.log(self.gaussian_pdf(c, x)))  # Verossimilhança
            posterior = prior + conditional  # Posterior
            posteriors.append(posterior)
        
        return self.classes[np.argmax(posteriors)]  # Retorna a classe com maior probabilidade

# Exemplo de uso
if __name__ == "__main__":
    # Exemplo de dados: 2 features
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y_train = np.array([0, 0, 1, 1, 1])

    X_test = np.array([[1.5, 2.5], [3.5, 4.5]])

    # Instancia o classificador Naive Bayes
    nb = NaiveBayes()
    
    # Treina o modelo
    nb.fit(X_train, y_train)
    
    # Faz a predição
    predictions = nb.predict(X_test)
    
    print("Predições:", predictions)