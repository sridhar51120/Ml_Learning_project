import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('/content/data.csv')

# Handle categorical variables
data['Gender'] = LabelEncoder().fit_transform(data['Gender'])  # Encode 'Gender' to numeric values

# Handling missing values (NaN) by imputing with the mean
data = data.fillna(data.mean())  # This fills NaN values with the mean of each column

# Define features (X) and the target variable (y)
X = data.drop('Age', axis=1)  # Features (excluding the target variable)
y = data['Age']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree model
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# Make predictions
predictions = decision_tree.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
