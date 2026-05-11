import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

def train_model(file_path):
    data = pd.read_csv(file_path)

    X = data.drop("prognosis", axis=1)
    y = data["prognosis"]

    model = RandomForestClassifier()
    model.fit(X, y)

    preds = model.predict(X)
    acc = accuracy_score(y, preds)

    # Confusion matrix
    cm = confusion_matrix(y, preds)
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.savefig("static/confusion_matrix.png")
    plt.close()

    with open("model.pkl","wb") as f:
        pickle.dump(model,f)

    return acc, list(X.columns)