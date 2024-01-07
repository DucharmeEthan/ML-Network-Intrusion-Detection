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
    print(data.shape)
    print(data.head)
    print(data.describe())

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
        10, 10), max_iter=10, verbose=True)

    # Train the model
    model.fit(X_train, y_train)
    # Evaluate the model
    score = model.score(X_test, y_test)
    print('Accuracy:', score)

    # Make predictions
    y_pred = model.predict(X_test)

   # print(model.n_features_in_)
   # print(model.t_)
   # print(model.n_iter_)
   # print(model.n_layers_)
   # print(model.n_outputs_)
   # print(model.out_activation_)

    # Plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.text(0, 0, cm[0, 0], horizontalalignment="center")
    plt.text(0, 1, cm[0, 1], horizontalalignment="center")
    plt.text(1, 0, cm[1, 0], horizontalalignment="center")
    plt.text(1, 1, cm[1, 1], horizontalalignment="center")
    plt.colorbar()
    plt.xticks([0, 1], ["normal packet", "malicious packet"])
    plt.yticks([0, 1], ["True", "False"])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()


if __name__ == '__main__':
    main()
