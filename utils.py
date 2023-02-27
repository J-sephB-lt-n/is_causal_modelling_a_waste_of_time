import catboost


class true_system_bayes_network:
    def __init__(
        self,
        system_name,
        system_source,
        n_vars_in_system,
        can_control_varnames,
        outcome_varnames,
        observed_only_varnames,
        daft_model,
        pgmpy_bayes_network_model,
        model_train_data,
        model_test_data,
    ):
        self.system_name = system_name
        self.system_source = system_source
        self.n_vars_in_system = n_vars_in_system
        self.can_control_varnames = can_control_varnames
        self.outcome_varnames = outcome_varnames
        self.observed_only_varnames = observed_only_varnames
        self.daft_model = daft_model
        self.pgmpy_bayes_network_model = pgmpy_bayes_network_model
        self.model_train_data = model_train_data
        self.model_test_data = model_test_data


def fit_catboost_binary_classifier(X_df, y_vec, verbose):
    train_df = catboost.Pool(
        X_df,
        y_vec,
        feature_names=list(X_df.columns),
        cat_features=list(X_df.columns),
    )

    model_params = {
        # "iterations": 500,
        "loss_function": "Logloss",
        "train_dir": "catboost_training",  # if allow_writing_files=True then training will create this folder and write training logs to it
        "allow_writing_files": False,  # don't allow training to write any files
        # "random_seed": 69,
    }

    catboost_model = catboost.CatBoostClassifier(**model_params)
    catboost_model.fit(train_df, verbose=verbose, plot=False)

    return catboost_model
