import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv, DataFrame

def preprocess(dataset):
    scalers = MinMaxScaler(feature_range=(0, 1))
    scaled = scalers.fit_transform(dataset)
    return scaled, scalers

def extract_VIT_capacity(x_datasets, y_datasets, seq_len, hop, sample, v=False, II=False, t=False, c=False, slicing=False, filp=False):
    V = []
    I = []
    T = []
    C = []

    x = []
    y = []

    for x_data, y_data in zip(x_datasets, y_datasets):
        x_df = read_csv(x_data).dropna()
        x_df = x_df[['cycle', 'voltage_battery', 'current_battery', 'temp_battery']]
        x_df = x_df[x_df['cycle'] != 0]  # cycle ke-0 tidak masuk
        x_df = x_df.reset_index().drop(columns="index")

        x_len = len(x_df.cycle.unique())  # - seq_len
        if slicing:
            x_slicing = int(x_len * 0.8)
            x_start = np.random.randint(1, x_len - x_slicing)
            slicing_x_list = x_df.cycle.unique()[x_start:x_start + x_slicing]
            bool_index = x_df['cycle'].isin(slicing_x_list)
            x_df = x_df[bool_index]
        x_df = x_df.reset_index().drop(columns="index")

        y_df = read_csv(y_data).dropna()
        y_df = y_df[['capacity']]
        y_df = y_df.values  # Convert pandas dataframe to numpy array
        y_df = y_df.astype('float32')  # Convert values to float

        if slicing:
            y_df = y_df[x_start:x_start + x_slicing]
        y_len = len(y_df)
        data_len = np.int32(np.floor((y_len - seq_len - 1) / hop)) + 1

        for i in range(y_len):
            if filp:
                cy = x_df.cycle.unique()[-i - 2]
                df = x_df.loc[x_df['cycle'] == cy]
                cap = np.array([y_df[-i - 1, 0]])
            else:
                cy = x_df.cycle.unique()[i]
                df = x_df.loc[x_df['cycle'] == cy]
                cap = np.array([y_df[i, 0]])
            C.append(cap)
            df_C = DataFrame(C).values
            scaled_C, scaler_C = preprocess(df_C)
            scaled_C = scaled_C.astype('float32')[:, :]

            le = len(df['voltage_battery']) % sample
            if v:
                vTemp = df['voltage_battery'].to_numpy()
                if le != 0:
                    vTemp = vTemp[0:-le]
                vTemp = np.reshape(vTemp, (len(vTemp) // sample, -1))
                vTemp = vTemp.mean(axis=0)
                V.append(vTemp)
                df_V = DataFrame(V).values
                scaled_V, scaler = preprocess(df_V)
                scaled_V = scaled_V.astype('float32')[:, :]

            elif II:
                iTemp = df['current_battery'].to_numpy()
                if le != 0:
                    iTemp = iTemp[0:-le]
                iTemp = np.reshape(iTemp, (len(iTemp) // sample, -1))
                iTemp = iTemp.mean(axis=0)
                I.append(iTemp)
                df_I = DataFrame(I).values
                scaled_I, scaler = preprocess(df_I)
                scaled_I = scaled_I.astype('float32')[:, :]

            elif t:
                tTemp = df['temp_battery'].to_numpy()
                if le != 0:
                    tTemp = tTemp[0:-le]
                tTemp = np.reshape(tTemp, (len(tTemp) // sample, -1))
                tTemp = tTemp.mean(axis=0)
                T.append(tTemp)
                df_T = DataFrame(T).values
                scaled_T, scaler = preprocess(df_T)
                scaled_T = scaled_T.astype('float32')[:, :]

        if v:
            for i in range(data_len):
                x.append(scaled_V[(hop * i):(hop * i + seq_len)])
        elif II:
            for i in range(data_len):
                x.append(scaled_I[(hop * i):(hop * i + seq_len)])
        elif t:
            for i in range(data_len):
                x.append(scaled_T[(hop * i):(hop * i + seq_len)])
        elif c:
            for i in range(data_len):
                x.append(scaled_C[(hop * i):(hop * i + seq_len)])

        for i in range(data_len):
            y.append(scaled_C[hop * i + seq_len])
    return np.array(x), np.array(y), scaler_C
