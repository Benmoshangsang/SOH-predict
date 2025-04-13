from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv, DataFrame
import numpy as np

def preprocess(dataset):
    scalers = MinMaxScaler(feature_range=(0, 1))
    scaled = scalers.fit_transform(dataset)
    return scaled, scalers

def extract_VIT_capacity(x_datasets, y_datasets, seq_len, hop, sample, v=False, II=False, t=False, c=False):
    x = []
    y = []
    scaler_C = None  # Initialize scaler for the capacity
    
    for x_data, y_data in zip(x_datasets, y_datasets):
        x_df = read_csv(x_data).dropna()
        x_df = x_df[['cycle', 'voltage_battery', 'current_battery', 'temp_battery']]
        x_df = x_df[x_df['cycle'] != 0]
        x_df = x_df.reset_index(drop=True)
        
        y_df = read_csv(y_data).dropna()
        y_df = y_df[['capacity']]
        y_df = y_df.values
        y_df = y_df.astype('float32')
        
        V = []
        I = []
        T = []
        C = []

        cycles = x_df['cycle'].unique()
        for cy in cycles:
            df = x_df[x_df['cycle'] == cy]
            cap = y_df[df.index[0], 0] if df.index[0] < len(y_df) else y_df[-1, 0]
            C.append([cap])

            if v:
                voltage = df['voltage_battery'].values
                if len(voltage) > sample:
                    V.extend(np.array_split(voltage, len(voltage) // sample))
                else:
                    V.append(voltage)  # Handle the case where data is less than the sample size

            elif II:
                current = df['current_battery'].values
                if len(current) > sample:
                    I.extend(np.array_split(current, len(current) // sample))
                else:
                    I.append(current)

            elif t:
                temperature = df['temp_battery'].values
                if len(temperature) > sample:
                    T.extend(np.array_split(temperature, len(temperature) // sample))
                else:
                    T.append(temperature)

        scaled_C, scaler_C = preprocess(np.array(C))
        
        if v:
            scaled_V, _ = preprocess(np.array(V))
        elif II:
            scaled_I, _ = preprocess(np.array(I))
        elif t:
            scaled_T, _ = preprocess(np.array(T))

        data_len = min(len(scaled_C), len(scaled_V) if v else len(scaled_I) if II else len(scaled_T) if t else 0)
        data_len = (data_len - seq_len) // hop

        for i in range(data_len):
            if v:
                x.append(scaled_V[(hop * i):(hop * i + seq_len)])
            elif II:
                x.append(scaled_I[(hop * i):(hop * i + seq_len)])
            elif t:
                x.append(scaled_T[(hop * i):(hop * i + seq_len)])
            elif c:
                x.append(scaled_C[(hop * i):(hop * i + seq_len)])

            y.append(scaled_C[hop * i + seq_len])

    return np.array(x), np.array(y), scaler_C