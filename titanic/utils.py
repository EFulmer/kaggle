import pandas._typing
import sklearn.base
import sklearn.model_selection


def train_test_model(
        X: pandas._typing.ArrayLike,
        y: pandas._typing.ArrayLike,  # TODO score column as a param and pass full DF?
        model: sklearn.base.BaseEstimator,
        # TODO scorer as a param?
    ) -> (sklearn.base.BaseEstimator, float):
    """Train model on the data provided, splitting it into training and
    CV sets.

    Args:
        X: Full data set to be split
        y: Target variable
        model: Model to fit.

    Returns:
        Tuple of the trained estimator and the accuracy.
    """
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=0)
    model.fit(X_train, y_train)
    accuracy = sklearn.metrics.accuracy_score(model.predict(X_test), y_test)
    return model, accuracy
