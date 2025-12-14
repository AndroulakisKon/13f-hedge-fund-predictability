from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor


def train_ols(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_ridge(X_train, y_train):
    # IMPORTANT: default alpha=1.0
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return model


def train_lasso(X_train, y_train):
    # IMPORTANT: small alpha, otherwise it collapses
    model = Lasso(alpha=0.001, max_iter=10_000)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    # IMPORTANT: this is where your results changed
    model = RandomForestRegressor(
        n_estimators=100,      # NOT 500
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model

