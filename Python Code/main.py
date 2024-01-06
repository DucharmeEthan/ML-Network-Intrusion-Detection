# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix

model = ""
default = "SNN"
LEARNING_RATE = 0.001


def main():

    csv_file_path = "../BaseData/kddcup99_converted.csv"
    data = pd.read_csv(csv_file_path)

    data_x = data.drop(data.columns[-1], axis=1)
    data_y = data['Labels ']

    label_encoder = LabelEncoder()
    data_y = label_encoder.fit_transform(data_y)

    categorical_cols = ['protocol_type', 'service', 'flag']
    data_x = pd.get_dummies(data_x, columns=categorical_cols)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data_x, data_y, test_size=0.2)

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create the neural network model
    model = MLPClassifier(hidden_layer_sizes=(
        100, 100), max_iter=50, verbose=2)

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    score = model.score(X_test, y_test)
    print('Accuracy:', score)

    # Make predictions
    y_pred = model.predict(X_test)

    # Plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    plt.show()


if __name__ == '__main__':
    main()
