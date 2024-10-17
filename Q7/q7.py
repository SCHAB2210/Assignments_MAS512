import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape data to include channel dimension
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Split the training data into training and validation sets
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Set up early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

# Train the model
history = model.fit(x_train, y_train, epochs=100, batch_size=64, 
                    validation_data=(x_val, y_val), 
                    callbacks=[early_stopping])

# Function to plot model predictions versus true labels
def plot_predictions(x, y_true, y_pred, title):
    plt.figure(figsize=(12, 4))
    # Randomly select 6 indices to plot
    print(len(x))

    indices = np.random.choice(len(x), size=6, replace=False)
    print(f'Selected indices for plotting: {indices}')  # Debugging line to check selected indices
    for i, idx in enumerate(indices):
        plt.subplot(2, 6, i + 1)
        plt.imshow(x[idx].reshape(28, 28), cmap='gray')
        plt.title(f'True: {y_true[idx]}\nPred: {y_pred[idx]}')
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

# Make predictions on the test set
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print(len(x_test))

# Plot predictions on the test set
plot_predictions(x_test, y_test, y_pred_classes, 'Model Predictions vs True Labels')
