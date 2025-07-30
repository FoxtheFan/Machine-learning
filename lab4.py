import os
import zipfile
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt

def A2():
    # Set seed for reproducibility
    np.random.seed(42)

    # Generate 20 data points with 2 features between 1 and 10
    X = np.random.uniform(1, 10, size=(20, 2))

    # Randomly assign class labels: 0 (Blue), 1 (Red)
    labels = np.random.randint(0, 2, size=20)

    # Create DataFrame for convenience
    df = pd.DataFrame(X, columns=['X', 'Y'])
    df['Class'] = labels

    # Plot
    colors = ['blue' if label == 0 else 'red' for label in df['Class']]
    plt.figure(figsize=(8, 6))
    plt.scatter(df['X'], df['Y'], c=colors, s=100, edgecolor='black')

    # Annotate the plot
    plt.title("Scatter Plot of 20 Data Points by Class")
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.grid(True)
    plt.show()


# Function to extract MFCC features from audio files
def extract_features_from_audio(data_dir, n_mfcc=13):
    features = []
    labels = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)

                # Extract label from filename (e.g., '101_s1.wav' → '101')
                label = file.split('_')[0]

                y, sr = librosa.load(file_path, sr=None)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
                mfcc_mean = np.mean(mfcc.T, axis=0)  # Average over time
                features.append(mfcc_mean)
                labels.append(label)
    return np.array(features), np.array(labels)

# Function to train and return KNN model
def train_knn_classifier(X_train, y_train, n_neighbors=3):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

# Function to evaluate model and return metrics
def evaluate_model(model, X, y_true):
    y_pred = model.predict(X)
    all_labels = np.unique(np.concatenate([y_true, y_pred]))  # ensure full label set
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return cm, precision, recall, f1


def A4():
    np.random.seed(42)
    X_train = np.random.uniform(1, 10, size=(20, 2))  # 20 points with 2 features
    y_train = np.random.randint(0, 2, size=20)        # Class labels: 0 or 1

    # Step 1: Train kNN classifier on training data
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # Step 2: Create a mesh grid of 2D points across the range [0, 11] for both X and Y
    x_min, x_max = 0, 11
    y_min, y_max = 0, 11
    h = 0.1  # Step size for grid

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    # Step 3: Flatten the grid to pass to the model and predict the class for each point
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = knn.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # Step 4: Plot the decision boundaries
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.3)  # Red-Blue colormap for regions

    # Step 5: Overlay the training data points
    for i in range(len(X_train)):
        color = 'blue' if y_train[i] == 0 else 'red'
        plt.scatter(X_train[i][0], X_train[i][1], c=color, edgecolors='black', s=80,
                    label=f"Class {y_train[i]}" if f"Class {y_train[i]}" not in plt.gca().get_legend_handles_labels()[1] else "")

    # Step 6: Add labels and legend
    plt.title("A4: Decision Boundary using kNN (k=3)")
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main program
if __name__ == "__main__":
    A2()
    A4()

    # Step 2: Feature Extraction
    """ X, y = extract_features_from_audio("Con_wav")

    # Step 3: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Train Classifier
    knn_model = train_knn_classifier(X_train, y_train)

    # Step 5: Evaluation on Training Set
    cm_train, precision_train, recall_train, f1_train = evaluate_model(knn_model, X_train, y_train)

    # Step 6: Evaluation on Test Set
    cm_test, precision_test, recall_test, f1_test = evaluate_model(knn_model, X_test, y_test)

    # Print results
    print("Training Set Evaluation:")
    print("Confusion Matrix:\n", cm_train)
    print("Precision:", precision_train)
    print("Recall:", recall_train)
    print("F1-Score:", f1_train)

    print("\nTest Set Evaluation:")
    print("Confusion Matrix:\n", cm_test)
    print("Precision:", precision_test)
    print("Recall:", recall_test)
    print("F1-Score:", f1_test)

    # Learning outcome observation
    if f1_train > 0.95 and f1_test < 0.75:
        print("\nModel is Overfitting.")
    elif abs(f1_train - f1_test) < 0.1 and f1_test > 0.8:
        print("\nModel is Regularfitting (Good Fit).")
    else:
        print("\nModel is Underfitting.")"""
        
