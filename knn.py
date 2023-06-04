import math
import pandas as pd


class KNN:
    def __init__(self, k):
        self.k = k

    def clustering(self, lista, sample):
        distances = []
        for i in range(len(lista)):
            temp = 0
            for j in range(len(sample)):
                temp += pow((lista.iloc[i][j] - sample[j]), 2)
            distances.append(math.sqrt(temp))
        lista["distance"] = distances
        lista = lista.sort_values("distance")
        dictionary = {"blues": 0, "classical": 0, "country": 0, "disco": 0, "hiphop": 0, "jazz": 0, "metal": 0,
                      "pop": 0, "reggae": 0, "rock": 0}
        for i in range(self.k):
            dictionary[lista.iloc[i]["label"]] += 1
        return max(dictionary, key=dictionary.get)

    def knn_predict(self, lista, lista_t):
        pred = []
        for i in range(len(lista_t)):
            name = self.clustering(lista, pd.to_numeric(lista_t.iloc[i]))
            pred.append(name)
        return pred

    @staticmethod
    def accuracy(prediction, lista_v):
        counter = 0
        for i in range(len(lista_v)):
            if prediction[i] == lista_v.iloc[i]:
                counter += 1
        return (counter / len(lista_v)) * 100
