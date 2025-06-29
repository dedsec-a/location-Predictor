# train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# -------------------------------
# 1️⃣ Delivery On-Time Classifier
# -------------------------------

# Load
df = pd.read_csv("Train.csv")

# Encode categorical columns
df_encoded = df.copy()
label_encoders = {}

for col in ['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

# Features & label
X = df_encoded.drop(['ID', 'Reached.on.Time_Y.N'], axis=1)
y = df_encoded['Reached.on.Time_Y.N']

# Split & train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ontime_model = RandomForestClassifier()
ontime_model.fit(X_train, y_train)

# Save model & encoders
with open("ontime_model.pkl", "wb") as f:
    pickle.dump({"model": ontime_model, "encoders": label_encoders}, f)

print("✅ Saved on-time delivery model as ontime_model.pkl")

# -------------------------------
# 2️⃣ Store Location Classifier
# -------------------------------

city_df = pd.read_csv("uscitypopdensity.csv")

# Create dummy target: high potential city if pop > 500,000 AND moderate density
city_df['HighPotential'] = ((city_df['2016 Population'] > 500000) & 
                            (city_df['Population Density (Persons/Square Mile)'] < 10000)).astype(int)

features = ['2016 Population', 'Population Density (Persons/Square Mile)', 'Land Area (Square Miles)']
X_city = city_df[features]
y_city = city_df['HighPotential']

Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_city, y_city, test_size=0.2, random_state=42)

store_model = RandomForestClassifier()
store_model.fit(Xc_train, yc_train)

# Save
with open("store_model.pkl", "wb") as f:
    pickle.dump(store_model, f)

print("✅ Saved store location model as store_model.pkl")
