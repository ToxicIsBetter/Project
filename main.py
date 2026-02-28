import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models

# === 1. LOAD & JOIN ===
df_price = pd.read_csv('btc_binance_daily.csv', parse_dates=['date'], index_col='date')
cm = pd.read_csv('btc.csv')
cm['date'] = pd.to_datetime(cm['time']).dt.date
cm.set_index('date', inplace=True)

# Top on-chain (file 4)[file:4]
onchain_cols = ['AdrActCnt', 'TxCnt', 'FeeTotNtv', 'HashRate', 'SplyCur', 'CapMVRVCur']
onchain_avail = [c for c in onchain_cols if c in cm.columns]
print(f"✅ Found on-chain: {onchain_avail}")

df = df_price.join(cm[onchain_avail], how='inner')
df.dropna(inplace=True)

print(f"\n✅ JOINED DATA")
print(f"Shape: {df.shape}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
print(df.tail(3))
df.to_csv('btc_full_onchain.csv')

# === 2. PREPARE FEATURES ===
features = ['price', 'volume', 'return', 'vol_7d'] + onchain_avail[:3]  # 7 features
print(f"\n📊 Features: {features}")

X = df[features].fillna(method='ffill').fillna(0).values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# === 3. CREATE SEQUENCES ===
def create_sequences(data, seq_len=60):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i+seq_len])
        ys.append(data[i+seq_len, 0])
    return np.array(xs), np.array(ys)

X_seq, y = create_sequences(X_scaled, 60)
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y[:split], y[split:]

print(f"\n📈 Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# === 4. TRANSFORMER MODEL (file 11/14 style)[file:11][file:14] ===
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.2):
    # Multi-head attention
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs
    
    # Feed-forward
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def build_transformer(seq_len, n_features, head_size=256, num_heads=4, ff_dim=128, num_blocks=2, dropout=0.2):
    inputs = layers.Input(shape=(seq_len, n_features))
    x = inputs
    
    # Stack transformer blocks
    for _ in range(num_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    
    # Global pooling + dense
    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1)(x)
    
    return models.Model(inputs, outputs)

model = build_transformer(
    seq_len=60, 
    n_features=len(features),
    head_size=256,
    num_heads=4,
    ff_dim=128,
    num_blocks=2,
    dropout=0.2
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="mse",
    metrics=["mae"]
)

print(f"\n🔥 TRANSFORMER MODEL")
model.summary()

# === 5. TRAIN ===
print("\n🔥 Training Transformer (50 epochs)...\n")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# === 6. EVALUATE ===
y_pred = model.predict(X_test)

y_test_full = np.hstack((y_test.reshape(-1,1), np.zeros((len(y_test), len(features)-1))))
y_pred_full = np.hstack((y_pred, np.zeros((len(y_pred), len(features)-1))))

y_test_inv = scaler.inverse_transform(y_test_full)[:, 0]
y_pred_inv = scaler.inverse_transform(y_pred_full)[:, 0]

rmse = np.sqrt(np.mean((y_test_inv - y_pred_inv)**2))
mae = np.mean(np.abs(y_test_inv - y_pred_inv))
mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100

print(f"\n✅ TRANSFORMER RESULTS:")
print(f"   RMSE: ${rmse:,.2f}")
print(f"   MAE:  ${mae:,.2f}")
print(f"   MAPE: {mape:.2f}%")
print(f"\n📊 File 11 Transformer: Enhanced prediction vs LSTM")

# Save
model.save('btc_transformer_model.h5')
print("\n✅ Model saved: btc_transformer_model.h5")
print("✅ Data saved: btc_full_onchain.csv")
