import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import xgboost as xgb
import joblib

df = pd.read_csv('cars_dataset.csv')
df['model'] = df['model'].str.strip()


#filter
df = df[df['price'] > 100]
df = df[df['mileage'] < 200000]


#split
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=1/3, random_state=42)







#target encoding for `model`

def kfold_target_encode(train_df, col, target_col, n_splits=5, seed=42):
    train_df = train_df.copy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    global_mean = train_df[target_col].mean()
    encoded = pd.Series(index=train_df.index, dtype=float)
    for tr_idx, va_idx in kf.split(train_df):
        means = train_df.iloc[tr_idx].groupby(col)[target_col].mean()
        encoded.iloc[va_idx] = train_df.iloc[va_idx][col].map(means).fillna(global_mean).values
    full_mapping = train_df.groupby(col)[target_col].mean()
    return encoded, full_mapping, global_mean


train_df = train_df.copy()
train_df['model_encoded'], model_mapping, global_mean_price = kfold_target_encode(train_df, 'model', 'price')

val_df = val_df.copy()
val_df['model_encoded'] = val_df['model'].map(model_mapping).fillna(global_mean_price)




cat_cols = ['Make', 'transmission', 'fuelType']
train_oh = pd.get_dummies(train_df[cat_cols], drop_first=True)
val_oh = pd.get_dummies(val_df[cat_cols], drop_first=True).reindex(columns=train_oh.columns, fill_value=0)







num_cols = ['year', 'mileage', 'tax', 'mpg', 'engineSize', 'model_encoded']
X_train = pd.concat([train_df[num_cols].reset_index(drop=True), train_oh.reset_index(drop=True)], axis=1)
X_val = pd.concat([val_df[num_cols].reset_index(drop=True), val_oh.reset_index(drop=True)], axis=1)
y_train = train_df['price'].reset_index(drop=True)
y_val = val_df['price'].reset_index(drop=True)


model = xgb.XGBRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    early_stopping_rounds=30, random_state=42
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)








artifacts = {
    'model': model,
    'model_mapping': model_mapping.to_dict(),   
    'global_mean_price': global_mean_price,
    'feature_columns': list(X_train.columns),
    'onehot_columns': list(train_oh.columns),
    'num_cols': num_cols,
    'cat_cols': cat_cols,
    'valid_makes': sorted(df['Make'].unique().tolist()),
    'valid_transmissions': sorted(df['transmission'].unique().tolist()),
    'valid_fuel_types': sorted(df['fuelType'].unique().tolist()),
}
joblib.dump(artifacts, 'model_artifacts.joblib')
print("Saved model_artifacts.joblib")
print("Feature columns:", artifacts['feature_columns'])
print("Valid Makes:", artifacts['valid_makes'])
