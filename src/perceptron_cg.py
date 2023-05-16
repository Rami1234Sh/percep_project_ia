import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.umbral = np.random.random()
        self.W = np.random.rand(2,1)
        
    def fit(self, X):
        #D                      etiqueta del Diccionario
        # {llave 1: [5, 2, 1]}


        #n_samples, n_features = X.shape
        
        for _ in range(self.n_iterations):
            for i in range(len(X)):
                
                #x = np.insert(X[i], 0, 1) !!!!!!!
                sum = np.dot(X[i], self.weights)                   #Producto punto (o sumatoria entre nuestros pesos y datos entrada)
                funcEsc = sum - self.umbral
                if funcEsc >= 0:
                    y_pred = 1
                else:
                    y_pred = 0
                
                funcDelta = D - y_pred                      #Calculando el delta entre nuestro label y la prediccion de entrenamiento

                deltaI = self.learning_rate * funcDelta * X[i]       #Calculando el delta i que se sumara a nuestro vector de pesos

                for i in range(self.W):               #Ajustando los pesos
                    self.W[i] = self.W[i] + deltaI

                self.umbral = self.umbral - self.learning_rate * funcDelta      #Modificando el umbral o bias
                #if y[i] * y_pred <= 0:
                    #self.weights += self.learning_rate * y[i] * x
                    

                    
    def predict(self, X):
        #n_samples, n_features = X.shape
        predictions = []
        for i in range(len(X)):
            #x = np.insert(X[i], 0, 1)
            y_pred = np.dot(X[i], self.weights)
            predictions.append(np.sign(y_pred))
            
        return predictions
