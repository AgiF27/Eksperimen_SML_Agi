import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data():
    train = pd.read_csv("../data-raw/churn-bigml-80.csv")
    test = pd.read_csv("../data-raw/churn-bigml-20.csv")
    return train, test

def add_features(df):
    df['Total minutes'] = (
        df['Total day minutes'] + df['Total eve minutes'] +
        df['Total night minutes'] + df['Total intl minutes']
    )
    df['Total calls'] = (
        df['Total day calls'] + df['Total eve calls'] +
        df['Total night calls'] + df['Total intl calls']
    )
    df['Total charge'] = (
        df['Total day charge'] + df['Total eve charge'] +
        df['Total night charge'] + df['Total intl charge']
    )
    df['High CSR calls'] = (df['Customer service calls'] > 3).astype(int)
    df['Call frequency'] = df['Total calls'] / df['Account length']
    return df

def main():
    # Load data
    train, test = load_data()
    
    # Konversi boolean
    for df in [train, test]:
        df['Churn'] = df['Churn'].astype(int)
        df['International plan'] = df['International plan'].map({'Yes': 1, 'No': 0})
        df['Voice mail plan'] = df['Voice mail plan'].map({'Yes': 1, 'No': 0})
    
    # Feature engineering
    train = add_features(train)
    test = add_features(test)
    
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    state_train = encoder.fit_transform(train[['State']])
    state_test = encoder.transform(test[['State']])
    state_cols = [f"State_{cat}" for cat in encoder.categories_[0][1:]]

    # Gabungkan ke dataframe
    train = pd.concat([train.reset_index(drop=True), pd.DataFrame(state_train, columns=state_cols)], axis=1)
    test = pd.concat([test.reset_index(drop=True), pd.DataFrame(state_test, columns=state_cols)], axis=1)

    train = train.drop(columns=['State'])
    test = test.drop(columns=['State'])
    
    # Scaling
    scaler = StandardScaler()
    numeric_cols = [
        'Account length', 'Number vmail messages', 'Total minutes', 'Total calls',
        'Total charge', 'Customer service calls', 'Call frequency', 'Area code'
    ]
    train[numeric_cols] = scaler.fit_transform(train[numeric_cols])
    test[numeric_cols] = scaler.transform(test[numeric_cols])
    
    # Fitur berbasis statistik TRAIN
    charge_75 = np.percentile(train['Total charge'], 75)
    day_minutes_75 = np.percentile(train['Total day minutes'], 75)
    
    train['Charge Group'] = (train['Total charge'] > charge_75).astype(int)
    test['Charge Group'] = (test['Total charge'] > charge_75).astype(int)
    train['Day usage high'] = (train['Total day minutes'] > day_minutes_75).astype(int)
    test['Day usage high'] = (test['Total day minutes'] > day_minutes_75).astype(int)
    
    # Hapus kolom rincian
    cols_to_drop = [
        'Total day minutes', 'Total eve minutes', 'Total night minutes', 'Total intl minutes',
        'Total day calls', 'Total eve calls', 'Total night calls', 'Total intl calls',
        'Total day charge', 'Total eve charge', 'Total night charge', 'Total intl charge',
        'Total charge'
    ]
    train = train.drop(columns=cols_to_drop)
    test = test.drop(columns=cols_to_drop)
    
    # Simpan
    train.to_csv("train_preprocessed.csv", index=False)
    test.to_csv("test_preprocessed.csv", index=False)
    print("Preprocessing selesai. File disimpan.")

if __name__ == "__main__":
    main()