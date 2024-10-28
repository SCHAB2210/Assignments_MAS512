# This repo includes both Assignment1 and 2

During the work, all questions were solved in each separate folders (Q1, Q2, Q3 etc..)
The final assembly is Assignment1_gr2.ipynb and Assignment2_gr2.ipynb

Its important to download dependencies from requirements.txt in order to compile code.


# Signal Analysis and Fault Classification Project

This project includes three main tasks involving signal analysis and fault classification. The tasks are organized as follows:

1. **Fourier Transformation of Signals**
2. **Pulsed Waveform Analysis with Duty Cycle Variation**
3. **Fault Classification with Convolutional Neural Network (CNN)**

## 1. Fourier Transformation of Signals

### Objective
The objective of this task is to generate and analyze different types of waveforms using Fourier transformations. The following waveforms are considered:
   - **Sinusoidal Signal**: Duration of 0.5 seconds with Amplitude = 5V, Frequency = 10 Hz.
   - **Triangular Waveform**: Duration of 0.5 seconds with Amplitude = 5V, Frequency = 10 Hz.
   - **Single Pulse Waveform**: Duty cycle of 0.2, Amplitude = 5V, time period = 0.1 seconds.
   - **Pulse Train**: Duration of 0.5 seconds with duty cycle 0.2, Amplitude = 5V, time period = 0.1 seconds.
   - **Chirp Signal**: Frequency increases from 10 Hz to 20 Hz over the duration from 0 to 1 second.

### Steps
1. Generate each signal and plot it in the time domain.
2. Apply Fourier Transformation to each signal to obtain its frequency spectrum.
3. Plot the Fourier Transform for each signal to observe its frequency components.

### Key Observations
Each waveform has unique frequency characteristics, which can be visualized in the Fourier domain. The Fourier Transform helps reveal the primary frequencies and harmonics for each type of waveform.

---

## 2. Pulsed Waveform Analysis with Duty Cycle Variation

### Objective
To understand how the **duty cycle** of a pulsed waveform affects its frequency spectrum. Specifically, we analyze the following cases:
   - A pulsed waveform of duration 0.5 seconds with Amplitude = 5V, Frequency = 10 Hz, and number of pulses = 4, varying the duty cycle from 0.1 to 0.5.

### Steps
1. Generate the pulsed waveform for each specified duty cycle (0.1, 0.2, 0.3, 0.4, 0.5).
2. Apply Fourier Transformation to each waveform to analyze the effect of duty cycle variation on the frequency spectrum.
3. Plot each waveform in the time domain and its corresponding frequency spectrum.

### Key Observations
The duty cycle affects the frequency spectrum by changing the energy distribution across frequencies. A higher duty cycle broadens the spectral density, resulting in more harmonics.

---

## 3. Fault Classification with CNN

### Objective
To classify motor shaft health as **healthy** or **faulty** using a Convolutional Neural Network (CNN) based on vibration data.

### Data Structure
- `Data_fault_classification/`
  - `train/`: Training data with two subfolders:
    - `00_15`: Healthy motor shaft data.
    - `10_15`: Faulty motor shaft data.
  - `val/`: Validation data with the same folder structure as training.
  - `test/`: Test data with the same folder structure as training.

### Model Architecture
A CNN is designed to classify the vibration data. The model structure includes:
   - Convolutional and MaxPooling layers for feature extraction.
   - Dense layers with Dropout and L2 regularization to prevent overfitting.
   - Sigmoid activation in the final layer for binary classification.

### Steps
1. Load and preprocess the data using `ImageDataGenerator`, with data augmentation applied to the training set.
2. Define and compile the CNN model with metrics for accuracy, precision, recall, and AUC.
3. Train the model with callbacks for Early Stopping and Learning Rate Scheduling to optimize training.
4. Evaluate the model on the test set and generate a confusion matrix and classification report for performance assessment.

### Key Observations
- Data augmentation helps improve model generalization.
- The model's accuracy is verified using the confusion matrix and classification report, providing insight into precision, recall, and AUC.

### Running the Project

1. **Install Requirements**:
   - Ensure Python and libraries are installed:
     ```bash
     pip install numpy pandas tensorflow matplotlib seaborn
     ```

2. **Run Fourier Transformation and Pulsed Waveform Analysis**:
   - Use the provided script to run the Fourier analysis for different waveforms.

3. **Run the CNN Model**:
   - Ensure the data is in the correct structure (`train/`, `val/`, `test/` folders).
   - Run the CNN training and evaluation script to train the model and evaluate its performance.

---

