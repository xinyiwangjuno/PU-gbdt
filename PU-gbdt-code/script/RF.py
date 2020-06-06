from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
def main():

    print ("-- Gradient Boosting Classification --")

    path = '/Users/junowang/Desktop/PU-gbdt-code/data/test_new.csv'
    data = pd.read_csv(path, header=0, encoding = "utf-8")
    print(data.info())
    y = data['label']
    data = data.drop('label',axis = 1)
    X = data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test,y_pred)

    print ("Accuracy:", accuracy)
    print('AUC', auc)

    # Plot().plot_in_2d(X_test, y_pred,
        #title="Gradient Boosting",
        #accuracy=accuracy,
        #legend_labels=data.target_names)



if __name__ == "__main__":
    main()