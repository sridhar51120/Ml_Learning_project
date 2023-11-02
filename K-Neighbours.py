import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('/content/data.csv')

# Convert categorical column 'Gender' using one-hot encoding
data = pd.get_dummies(data, columns=['Gender'])

# Handling missing values (NaN) by imputing with the mean
data = data.fillna(data.mean())  # This fills NaN values with the mean of each column

# Define features (X) and the target variable (y)
X = data.drop('Age', axis=1)  # Features (excluding the target variable)
y = data['Age']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the KNN model
k = 5  # You can choose an appropriate value for 'k'
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Make predictions
predictions = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
