import math
import random as rn
import warnings
import random
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing

pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')

general_path = 'dataset/archive/Data'
gen = pd.read_csv(f'{general_path}/features_30_sec.csv')


# Class for processing data in datasets
class DataProcessing:
    @staticmethod
    def shuffle(x):
        for i in range(len(x) - 1, 0, -1):
            j = rn.randint(0, i - 1)
            x.iloc[i], x.iloc[j] = x.iloc[j], x.iloc[i]

    @staticmethod
    def min_max_scaling(x):
        values = x.select_dtypes(exclude=["object", "string"])
        columnNames = values.columns.tolist()
        for column in columnNames:
            x[column] = pd.to_numeric(x[column])
            data = x.loc[:, column]
            min1 = min(data)
            max1 = max(data)
            for row in range(len(x)):
                xprim = (x.at[row, column] - min1) / (max1 - min1)
                x.at[row, column] = xprim

    @staticmethod
    def split(x, train_ratio, val_ratio):
        train_size = int(len(x) * train_ratio)
        val_size = int(len(x) * val_ratio)

        tr_set = x[:train_size]
        va_set = x[train_size:train_size + val_size]
        te_set = x[train_size + val_size:]

        return tr_set, va_set, te_set

    @staticmethod
    def find_similar_songs(name, label, number):
        data = pd.read_csv(f'{general_path}/features_30_sec.csv', index_col='filename')

        # Extract labels
        labels = data[['label']]

        # Drop labels from original dataframe
        data = data.drop(columns=['length', 'label'])

        # Scale the data
        data_scaled = preprocessing.scale(data)

        # Cosine similarity
        similarity = cosine_similarity(data_scaled)

        # Convert into a dataframe and then set the row index and column names as labels
        sim_df_labels = pd.DataFrame(similarity)
        sim_df_names = sim_df_labels.set_index(labels.index)
        sim_df_names.columns = labels.index

        # Find songs most similar to another song
        series = sim_df_names[name].sort_values(ascending=False)

        # Remove cosine similarity == 1
        series = series.drop(name)

        # Filter the series based on the label
        series = series[series.index.str.startswith(label)]

        # Display the "n" top matches
        print("\n*******\nSimilar songs to", name)
        print(series.head(number))


# Class for KNN
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
        dictionary = dict()
        dictionary1 = {"blues": 0, "classical": 0, "country": 0, "disco": 0, "hiphop": 0, "jazz": 0, "metal": 0,
                       "pop": 0, "reggae": 0, "rock": 0}
        for i in range(self.k):
            dictionary1[lista.iloc[i]["label"]] += 1
        return max(dictionary1, key=dictionary1.get)

    def knn_predict(self, lista, lista_t):
        pred = []
        for i in range(len(lista_t)):
            name = self.clustering(lista, pd.to_numeric(lista_t.iloc[i]))
            pred.append(name)
        return pred

    def accuracy(self, prediction, lista_v):
        counter = 0
        for i in range(len(lista_v)):
            if prediction[i] == lista_v.iloc[i]:
                counter += 1
        return (counter / len(lista_v)) * 100


# Shuffle the dataset
DataProcessing.shuffle(gen)
print("Shuffled")

# Split
train_set, test_set, _ = DataProcessing.split(gen, 0.8, 0.2)
print("Split")

# Normalize the sets
DataProcessing.min_max_scaling(train_set)
test_set = test_set.reset_index(drop=True)
DataProcessing.min_max_scaling(test_set)
print("Normalized")

# Drop features that negatively affect accuracy
features_list = [
    'chroma_stft_var',
    'rms_var',
    'spectral_centroid_var',
    'spectral_bandwidth_var',
    'rolloff_var',
    'zero_crossing_rate_var',
    'harmony_var',
    'perceptr_var',
    'perceptr_mean'
]
for feature in features_list:
    train_set = train_set.drop(feature, axis=1)
    test_set = test_set.drop(feature, axis=1)
print("Features dropped")

# Perform KNN algorithm and display the accuracy (can be performed for multiple k)
k = [5]
predicted = []
for i in k:
    knn = KNN(k=i)
    predicted = knn.knn_predict(train_set.drop(['filename'], axis=1), test_set.drop(['filename', 'label'], axis=1))
    accuracy = knn.accuracy(predicted, test_set['label'])
    print(f"Accuracy: {accuracy}, k={i}")

# Select a random entry with filename starting with "X"
rock_entries = test_set[test_set['filename'].str.startswith('rock')]
random_entry_index = random.choice(rock_entries.index)

# Get the filename and corresponding predicted label
filename = test_set.loc[random_entry_index, 'filename']
predicted_label = predicted[random_entry_index]
actual_label = test_set.loc[random_entry_index, 'label']

# Display the results
print("Random entry:")
print("Filename:", filename)
print("Predicted Label:", predicted_label)
print("Actual Label:", actual_label)

DataProcessing.find_similar_songs(filename, predicted_label, 3)
