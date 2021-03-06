from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
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
    gbdt = GradientBoostingClassifier(
        loss='deviance'
        , learning_rate=0.1
        , n_estimators=100
        , subsample=1
        , min_samples_split=2
        , min_samples_leaf=1
        , max_depth=3
        , init=None
        , random_state=None
        , max_features=None
        , verbose=0
        , max_leaf_nodes=None
        , warm_start=False
    )
    gbdt.fit(X_train, y_train)
    y_pred = gbdt.predict(X_test)
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