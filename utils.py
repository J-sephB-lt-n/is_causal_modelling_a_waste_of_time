class true_system_bayes_network:
    def __init__(
        self,
        system_name,
        system_source,
        can_control_varnames,
        daft_model,
        pgmpy_bayes_network_model,
        model_train_data,
        model_test_data,
    ):
        self.system_name = system_name
        self.system_source = system_source
        self.variable_roles = can_control_varnames
        self.daft_model = daft_model
        self.pgmpy_bayes_network_model = pgmpy_bayes_network_model
        self.model_train_data = model_train_data
        self.model_test_data = model_test_data
