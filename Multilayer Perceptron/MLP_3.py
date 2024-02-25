import numpy as np
import pandas as pd

# Klasa reprezentująca wielowarstwowy perceptron
class WielowarstwowyPerceptron:
    def __init__(self, warstwy):
        self.warstwy = warstwy
        self.wagi = []
        self.biases = []
        for i in range(len(warstwy) - 1):
            self.wagi.append(np.random.randn(warstwy[i], warstwy[i + 1]))
            self.biases.append(np.random.randn(1, warstwy[i + 1]))

    def forward(self, X):
        aktywacja = X
        for w, b in zip(self.wagi, self.biases):
            z = np.dot(aktywacja, w) + b
            aktywacja = sigmoid(z)
        return aktywacja

    def ustawienie_parametrów(self, parametry):
        idx = 0
        self.wagi = []
        self.biases = []
        for i in range(len(self.warstwy) - 1):
            rozmiar_w = self.warstwy[i] * self.warstwy[i + 1]
            rozmiar_b = self.warstwy[i + 1]
            self.wagi.append(parametry[idx:idx + rozmiar_w].reshape(self.warstwy[i], self.warstwy[i + 1]))
            idx += rozmiar_w
            self.biases.append(parametry[idx:idx + rozmiar_b].reshape(1, self.warstwy[i + 1]))
            idx += rozmiar_b

    def pobieranie_parametrów(self):
        return np.concatenate([w.flatten() for w in self.wagi] + [b.flatten() for b in self.biases])

# Funkcje pomocnicze
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mse(prawdziwe_y, przewidywane_y):
    return np.mean(np.square(prawdziwe_y - przewidywane_y))

def dokladnosc(prawdziwe_y, przewidywane_y):
    return np.mean(np.argmax(prawdziwe_y, axis=1) == np.argmax(przewidywane_y, axis=1))

def funkcja_celu(parametry, model, X, Y):
    model.ustawienie_parametrów(parametry)
    prognozy = model.forward(X)
    return mse(Y, prognozy)

# Implementacja algorytmu ewolucji różnicowej
def ewolucja_roznicowa(funkcja, model, granice, X, Y, mut=0.8, krzyz=0.7, wielkosc_populacji=20, iteracje=1000):
    wymiary = len(model.pobieranie_parametrów())
    populacja = np.random.rand(wielkosc_populacji, wymiary)
    min_granica, max_granica = np.asarray(granice).T
    roznica = np.fabs(min_granica - max_granica)
    pop_denorm = min_granica + populacja * roznica
    dostosowanie = np.asarray([funkcja(ind, model, X, Y) for ind in pop_denorm])
    najlepszy_idx = np.argmin(dostosowanie)
    najlepszy = pop_denorm[najlepszy_idx]
    for i in range(iteracje):
        for j in range(wielkosc_populacji):
            idxs = [idx for idx in range(wielkosc_populacji) if idx != j]
            a, b, c = populacja[np.random.choice(idxs, 3, replace=False)]
            mutacja = np.clip(a + mut * (b - c), 0, 1)
            punkty_krzyzowania = np.random.rand(wymiary) < krzyz
            if not np.any(punkty_krzyzowania):
                punkty_krzyzowania[np.random.randint(0, wymiary)] = True
            proba = np.where(punkty_krzyzowania, mutacja, populacja[j])
            proba_denorm = min_granica + proba * roznica
            f = funkcja(proba_denorm, model, X, Y)
            if f < dostosowanie[j]:
                dostosowanie[j] = f
                populacja[j] = proba
                if f < dostosowanie[najlepszy_idx]:
                    najlepszy_idx = j
                    najlepszy = proba_denorm
        yield najlepszy, dostosowanie[najlepszy_idx]
        
# Wczytanie i przygotowanie danych
df = pd.read_csv('C:/Users/wojci/Downloads/Iris.csv')
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
y = df['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}).values

# Standaryzacja danych
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Konwersja y do postaci one-hot
Y = np.zeros((y.size, y.max() + 1))
Y[np.arange(y.size), y] = 1

# Podział na zbiór treningowy i testowy
indeksy = np.arange(X.shape[0])
np.random.shuffle(indeksy)
podzial = int(0.8 * len(indeksy))  # 80% danych jako zbiór treningowy
indeksy_treningowe, indeksy_testowe = indeksy[:podzial], indeksy[podzial:]
X_treningowe, X_testowe = X[indeksy_treningowe], X[indeksy_testowe]
Y_treningowe, Y_testowe = Y[indeksy_treningowe], Y[indeksy_testowe]

# Utworzenie modelu perceptronu
perceptron = WielowarstwowyPerceptron([4, 10, 3])  # Architektura sieci

# Uruchomienie algorytmu ewolucji różnicowej
granice = [(-1, 1)] * len(perceptron.pobieranie_parametrów())
generator_er = ewolucja_roznicowa(funkcja_celu, perceptron, granice, X_treningowe, Y_treningowe)
for i in range(1000):
    najlepszy, wynik = next(generator_er)
    print(f'Iteracja {i}, MSE: {wynik}')

# Ostateczna ocena modelu na zbiorze testowym
perceptron.ustawienie_parametrów(najlepszy)
prognozy = perceptron.forward(X_testowe)
print(f'Dokładność na zbiorze testowym: {dokladnosc(Y_testowe, prognozy)}')       
