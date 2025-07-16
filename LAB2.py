import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


def A1(df_clean):
    # Matrix A: Quantities purchased of each product
    A = df_clean[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].to_numpy()

    # Matrix C: Total payment made by each customer
    C = df_clean[['Payment (Rs)']].to_numpy()

    # Dimensionality (number of features/products)
    dimensionality = A.shape[1]
    print("Dimensionality of vector space:", dimensionality)

    # Number of vectors (number of customers)
    num_vectors = A.shape[0]
    print("Number of vectors in the space:", num_vectors)

    # Rank of matrix A
    rank_A = np.linalg.matrix_rank(A)
    print("Rank of matrix A:", rank_A)

    # Use pseudo-inverse to estimate cost per product (solving AX = C)
    A_pinv = np.linalg.pinv(A)
    X = A_pinv @ C  # X contains the estimated cost per unit of each product

    # Print product cost estimates
    product_names = ['Candy', 'Mango (Kg)', 'Milk Packet']
    print("\nEstimated product costs (Rs/unit):")
    for name, cost in zip(product_names, X.flatten()):
        print(f"{name}: ₹{round(cost, 2)}")


def A2(df_clean):
    df_clean['Class'] = df_clean['Payment (Rs)'].apply(lambda x: 'RICH' if x > 200 else 'POOR')

    # Features and target
    X = df_clean[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']]
    y = df_clean['Class']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    # Create and train Decision Tree model
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

file_path = "labdata.xlsx" 
df = pd.read_excel(file_path, sheet_name="Purchase data")

def A3():
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Plot Chg% vs Day of the Week
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=irctc_df, x='Day', y='Chg%', hue='Day', palette='Set2', s=100)

    plt.title("Change % (Chg%) vs Day of the Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Change % (Chg%)")
    plt.grid(True)
    plt.legend(title="Day", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Clean the data: select only relevant columns and drop NaNs
df_clean = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)', 'Payment (Rs)']].dropna()
A1(df_clean)
A2(df_clean)
A3()