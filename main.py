import random
import pandas as pd
from data_processing import DataProcessing
from knn import KNN
from music_analyser import MusicAnalyser

pd.options.mode.chained_assignment = None

general_path = 'dataset/archive/Data'
gen = pd.read_csv(f'{general_path}/features_30_sec.csv')

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
random_entries = test_set[test_set['filename'].str.startswith('rock')]
random_entry_index = random.choice(random_entries.index)

# Get the filename and corresponding predicted label
filename = test_set.loc[random_entry_index, 'filename']
predicted_label = predicted[random_entry_index]
actual_label = test_set.loc[random_entry_index, 'label']

# Display the results
print("Random entry:")
print("Filename:", filename)
print("Predicted Label:", predicted_label)
print("Actual Label:", actual_label)

# Create MusicAnalyzer instance
analyzer = MusicAnalyser(general_path)
series = analyzer.find_similar_songs(filename, predicted_label)

# Display the "n" top matches
n = 3
print("\n*******\nSimilar songs to", filename)
print(series.head(n))
