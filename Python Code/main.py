# Imports
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import os
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


names = [
    "Nearest Neighbors",
    "Linear SVM",
    # "RBF SVM",
    # "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
]

toTrain = [

]

classifiers = [
    KNeighborsClassifier(),
    SVC(kernel="linear", C=0.025, random_state=42, verbose=True),
    # SVC(C=1, random_state=42, verbose=True),
    # GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42), I do not have enough ram for this
    DecisionTreeClassifier(max_depth=10, random_state=42),
    RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1, random_state=42, verbose=1
    ),
    MLPClassifier(hidden_layer_sizes=(60, 60), max_iter=50,
                  verbose=True, random_state=42),
    AdaBoostClassifier(random_state=42),
    GaussianNB(),
]

training_needed = True


def loadModels():
    for name, model in zip(names, classifiers):
        path = "ML_models/" + name + '.txt'
        print("Trying to load" + name)
        if (os.path.exists(path)):
            f = open(path, "rb")
            model = pickle.load(f)
            f.close()
            toTrain.append(False)
            print("Loaded " + name)

        else:
            toTrain.append(True)
            print(name + " was not loaded")


def saveModel(model, name):
    f = open("ML_models/" + name + ".txt", "wb")
    m = pickle.dumps(model)
    f.write(m)
    f.close()
    print("Saved model " + name)


def printAttrs(model):
    print(model.n_features_in_)
    print(model.t_)
    print(model.n_iter_)
    print(model.n_layers_)
    print(model.n_outputs_)
    print(model.out_activation_)


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

    # load all the models also determines which ones need to train if files are missing
    loadModels()

    # Choose the model
    for name, model, trainNeeded in zip(names, classifiers, toTrain):
        # Train the model
        if (trainNeeded):
            print("training " + name)
            model.fit(X_train, y_train)
            print("training over")
            saveModel(model, name)

            # Evaluate the model
            score = model.score(X_test, y_test)
            print('Accuracy:', score)

        # Make predictions
        y_pred = model.predict(X_test)

        # Plot the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.title(name + " confusion matrix")
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.text(0, 0, cm[0, 0], horizontalalignment="center")
        plt.text(0, 1, cm[0, 1], horizontalalignment="center")
        plt.text(1, 0, cm[1, 0], horizontalalignment="center")
        plt.text(1, 1, cm[1, 1], horizontalalignment="center")
        plt.colorbar()
        plt.xticks([0, 1], ["normal packet", "malicious packet"])
        plt.yticks([0, 1], ["normal", "malicious"])
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.show()


if __name__ == '__main__':
    main()
