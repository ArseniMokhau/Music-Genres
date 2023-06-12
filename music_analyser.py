import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing


class MusicAnalyser:
    def __init__(self, general_path):
        self.general_path = general_path

    def find_similar_songs(self, name, label):
        data = pd.read_csv(f'{self.general_path}/features_30_sec.csv', index_col='filename')

        # Copy the entry with the specified name
        entry_with_name = data.loc[[name]]

        # Filter the data based on the label
        filtered_data = data[data['label'] == label]

        # Concatenate the entry with the filtered data
        filtered_data = pd.concat([filtered_data, entry_with_name])

        # Extract labels
        labels = filtered_data[['label']]

        # Drop unnecessary columns from the filtered dataframe
        filtered_data = filtered_data.drop(columns=['length', 'label'])

        # Scale the filtered data
        data_scaled = preprocessing.scale(filtered_data)

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

        return series
