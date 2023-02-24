"""
Causal Structure from:
    "Construction of a Bayesian Network for Mammographic Diagnosis of Breast Cancer"
    Kahn Jr et al. (1996)
"""

import daft
from pgmpy.models.BayesianModel import BayesianNetwork
from pgmpy.factors.discrete import (
    TabularCPD,
)  # discrete Conditional Probability Distribution

if __name__ == "__main__":
    # allow visibility of parent directory
    import sys

    sys.path.append("..")

import utils

daft_net = daft.PGM(shape=[8, 10], grid_unit=3.0, node_unit=2.0)
daft_net.add_node(daft.Node("previous_biopsy", r"previous_biopsy", 1, 3))
daft_net.add_node(daft.Node("age_at_menarche", r"age_at_menarche", 1, 4))
daft_net.add_node(daft.Node("age_at_1st_live_birth", r"age_at_1st_live_birth", 1, 5))
daft_net.add_node(daft.Node("num_relatives", r"num_relatives", 1, 6))
daft_net.add_node(daft.Node("age", r"age", 1, 7))
daft_net.add_node(
    daft.Node("architectural_distortion", r"architectural_distortion", 1, 1)
)
daft_net.add_node(daft.Node("asymmetry", r"asymmetry", 3, 1))
daft_net.add_node(daft.Node("developing_density", r"developing_density", 5, 1))
daft_net.add_node(daft.Node("breast_cancer", r"breast_cancer", 3, 4))
daft_net.add_node(daft.Node("mass", r"mass", 5, 6))
daft_net.add_node(daft.Node("mass_present", r"mass_present", 5, 5))
daft_net.add_node(daft.Node("calcification_present", r"calcification_present", 5, 4))
daft_net.add_node(daft.Node("calcification", r"calcification", 5, 3))

daft_net.add_node(daft.Node("pain", r"pain", 3, 8))
daft_net.add_node(daft.Node("nipple_discharge", r"nipple_discharge", 4, 8))

daft_net.add_node(daft.Node("calcification_size", r"calcification_size", 8, 1))
daft_net.add_node(
    daft.Node("calcification_arrangement", r"calcification_arrangement", 8, 2)
)
daft_net.add_node(daft.Node("calcification_density", r"calcification_density", 8, 3))
daft_net.add_node(daft.Node("calcification_shape", r"calcification_shape", 8, 4))
daft_net.add_node(daft.Node("num_in_cluster", r"num_in_cluster", 8, 5))
daft_net.add_node(
    daft.Node("calcification_cluster_shape", r"calcification_cluster_shape", 8, 6)
)
daft_net.add_node(daft.Node("mass_location", r"mass_location", 8, 7))
daft_net.add_node(daft.Node("halo_sign", r"halo_sign", 8, 8))
daft_net.add_node(daft.Node("mass_density", r"mass_density", 8, 9))
daft_net.add_node(daft.Node("mass_margin", r"mass_margin", 8, 10))
daft_net.add_edge("age", "breast_cancer")
daft_net.add_edge("num_relatives", "breast_cancer")
daft_net.add_edge("age_at_1st_live_birth", "breast_cancer")
daft_net.add_edge("age_at_menarche", "breast_cancer")
daft_net.add_edge("previous_biopsy", "breast_cancer")
daft_net.add_edge("breast_cancer", "pain")
daft_net.add_edge("breast_cancer", "nipple_discharge")
daft_net.add_edge("breast_cancer", "architectural_distortion")
daft_net.add_edge("breast_cancer", "asymmetry")
daft_net.add_edge("breast_cancer", "developing_density")
daft_net.add_edge("breast_cancer", "mass")
daft_net.add_edge("breast_cancer", "calcification")
daft_net.add_edge("previous_biopsy", "architectural_distortion")
daft_net.add_edge("mass", "mass_present")
daft_net.add_edge("calcification", "calcification_present")
daft_net.add_edge("mass", "mass_margin")
daft_net.add_edge("mass", "mass_density")
daft_net.add_edge("mass", "halo_sign")
daft_net.add_edge("mass", "mass_location")
daft_net.add_edge("calcification", "calcification_cluster_shape")
daft_net.add_edge("calcification", "num_in_cluster")
daft_net.add_edge("calcification", "calcification_shape")
daft_net.add_edge("calcification", "calcification_density")
daft_net.add_edge("calcification", "calcification_arrangement")
daft_net.add_edge("calcification", "calcification_size")

mammonet_system = utils.true_system_bayes_network(
    system_name="MammoNet",
    system_source="[paper] Construction of a Bayesian Network for Mammographic Diagnosis of Breast Cancer Kahn Jr et al. (1996)",
    can_control_varnames=[],
    daft_model=daft_net,
)

if __name__ == "__main__":
    daft_net.render()
