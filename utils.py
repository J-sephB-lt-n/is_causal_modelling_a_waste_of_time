class true_system_bayes_network:
    def __init__(self, system_name, system_source, can_control_varnames, daft_model):
        self.system_name = system_name
        self.system_source = system_source
        self.variable_roles = can_control_varnames
        self.daft_model = daft_model

    def visualize_system(self):
        self.daft_model.render()
