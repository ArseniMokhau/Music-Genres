import random as rn
import pandas as pd


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
                prim = (x.at[row, column] - min1) / (max1 - min1)
                x.at[row, column] = prim

    @staticmethod
    def split(x, train_ratio, val_ratio):
        train_size = int(len(x) * train_ratio)
        val_size = int(len(x) * val_ratio)

        tr_set = x[:train_size]
        va_set = x[train_size:train_size + val_size]
        te_set = x[train_size + val_size:]

        return tr_set, va_set, te_set
