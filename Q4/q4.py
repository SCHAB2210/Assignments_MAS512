import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

data = pd.read_csv('medical_costs.csv')

data = pd.get_dummies(data, columns=['sex', 'smoker', 'region'], drop_first=True)
X = data.drop('charges', axis=1).values
y = data['charges'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def create_model(optimizer, loss):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(X_train.shape[1],)))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=optimizer, loss=loss)
    return model

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

models = [
    {'name': 'Model 1', 'optimizer': tf.keras.optimizers.SGD(learning_rate=0.001), 'loss': 'mse'},
    {'name': 'Model 2', 'optimizer': tf.keras.optimizers.Adam(learning_rate=0.001), 'loss': 'msle'},
    {'name': 'Model 3', 'optimizer': tf.keras.optimizers.Adagrad(learning_rate=0.001), 'loss': 'mae'},
    {'name': 'Model 4', 'optimizer': tf.keras.optimizers.SGD(learning_rate=0.001), 'loss': 'mse'},
    {'name': 'Model 5', 'optimizer': tf.keras.optimizers.SGD(learning_rate=0.001), 'loss': 'mae'},
]

results = []

for model_config in models:
    model_name = model_config['name']
    optimizer = model_config['optimizer']
    loss = model_config['loss']
    
    model = create_model(optimizer=optimizer, loss=loss)
    history = model.fit(X_train, y_train, epochs=500, validation_split=0.2, callbacks=[early_stopping], verbose=0)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

    total_params = model.count_params()

    results.append({
        'model': model_name,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
        'total_params': total_params,
        'history': history,
        'train_errors': y_train - y_train_pred,
        'test_errors': y_test - y_test_pred,
    })

plt.figure(figsize=(16, 12))
for i, result in enumerate(results, start=1):
    plt.subplot(3, 2, i)
    plt.plot(result['history'].history['loss'], label='Training Loss', color='blue')
    plt.plot(result['history'].history['val_loss'], label='Validation Loss', color='orange')
    plt.title(f'{result["model"]} - Training and Validation Loss', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid()

plt.tight_layout()
plt.show()

model_names = [result['model'] for result in results]
r2_scores = [result['r2_test'] for result in results]
rmse_scores = [result['rmse_test'] for result in results]

plt.figure(figsize=(14, 7))
x = np.arange(len(model_names))

plt.bar(x - 0.2, r2_scores, 0.4, label='R² Score', color='lightblue')
plt.bar(x + 0.2, rmse_scores, 0.4, label='RMSE', color='salmon')

plt.title('Model Performance Comparison', fontsize=16)
plt.xlabel('Models', fontsize=14)
plt.ylabel('Scores', fontsize=14)
plt.xticks(x, model_names)
plt.legend(fontsize=10)
plt.grid()

plt.tight_layout()
plt.show()

summary_data = {
    'Model': model_names,
    'R2 Score': r2_scores,
    'RMSE': rmse_scores,
    'Total Parameters': [result['total_params'] for result in results],
}
summary_df = pd.DataFrame(summary_data)

print("Summary of Model Performance:")
print(summary_df)

colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']

for i, result in enumerate(results):
    model_name = result['model']
    train_errors = result['train_errors']
    test_errors = result['test_errors']

    train_errors = np.array(train_errors).flatten()
    test_errors = np.array(test_errors).flatten()

    plt.figure(figsize=(12, 6))
    plt.hist(train_errors, bins=30, alpha=0.5, label='Train Error', color=colors[i % len(colors)])
    plt.hist(test_errors, bins=30, alpha=0.5, label='Test Error', color=colors[(i + 1) % len(colors)])
    plt.title(f'Prediction Error Histogram for {model_name}', fontsize=14)
    plt.xlabel('Error', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid()
    plt.show()

    mean_train = np.mean(train_errors)
    std_train = np.std(train_errors)
    mean_test = np.mean(test_errors)
    std_test = np.std(test_errors)

    print(f'{model_name} Train Error: Mean = {mean_train:.2f}, Std = {std_train:.2f}')
    print(f'{model_name} Test Error: Mean = {mean_test:.2f}, Std = {std_test:.2f}')
