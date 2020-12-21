from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier


# @TODO: Maybe extend the sklearn classifier
class Classifier:
    def __init__(self, data: list, column_order: list):
        self.data = data
        self.column_order = column_order
        self.encoder = None
        self.classifier = None

    @staticmethod
    def _split_variables(ds):
        X = ds.iloc[:, :-1].values
        y = ds.iloc[:, -1].values
        return X, y

    @staticmethod
    def _encode():
        pass

    # @TODO: Improve column list for encoding
    def _encode_and_save_model(self, X, columns: list = [0, 1]):
        ct = ColumnTransformer(
            transformers=[('encoder', OneHotEncoder(), columns)],
            remainder='passthrough')
        self.encoder = ct.fit(X)
        return self.encoder.transform(X).toarray()

    def get_classifier_model(self):
        return self.classifier

    def get_encoder_model(self):
        return self.encoder

    def get_accuracy(self, X_test, y_test):
        accuracies = cross_val_score(
            estimator=self.classifier, X=X_test, y=y_test, cv=5)
        return round(accuracies.mean().round(3) * 100, 2)

    def train(self):
        ds = self.data[self.column_order]
        X, y = self._split_variables(ds)
        X = self._encode_and_save_model(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=0)

        self.classifier = DecisionTreeClassifier()
        self.classifier.fit(X_train, y_train)

        accuracy = self.get_accuracy(X_test, y_test)
        print("Accuracy is \x1b[1;31m{}%\x1b[0m".format(accuracy))

    def predict(self):
        pass
