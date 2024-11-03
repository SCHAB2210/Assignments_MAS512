import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reduce training set size for efficiency
x_train, y_train = x_train[:500].astype('float32') / 255.0, y_train[:500]
x_test = x_test.astype('float32') / 255.0

# Reshape for Conv2D layer and split validation set
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
x_val, y_val = x_train[-100:], y_train[-100:]
x_train, y_train = x_train[:-100], y_train[:-100]

# Define and compile the model
model = tf.keras.models.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
history = model.fit(x_train, y_train, epochs=100, batch_size=64, 
                    validation_data=(x_val, y_val), 
                    callbacks=[early_stopping])

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Display confusion matrix for the first 500 test samples
y_pred = np.argmax(model.predict(x_test[:500]), axis=1)
conf_matrix = confusion_matrix(y_test[:500], y_pred)
ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.arange(10)).plot(cmap=plt.cm.Blues)
plt.show()

# Print model accuracy
accuracy = np.mean(y_pred == y_test[:500])
print(f'Accuracy: {accuracy * 100:.2f}%')
