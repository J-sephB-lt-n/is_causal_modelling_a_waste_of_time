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
import numpy as np

if __name__ == "__main__":
    # allow visibility of parent directory
    import sys

    sys.path.append("..")

import utils

daft_net = daft.PGM(shape=[8, 10], grid_unit=3.0, node_unit=2.0)
daft_net.add_node(daft.Node("previous_biopsy", r"previous_biopsy", 1, 3))
daft_net.add_node(daft.Node("age_at_menarche", r"age_at_menarche", 1, 4))
daft_net.add_node(
    daft.Node(
        "age_at_1st_live_birth",
        r"age_at_1st_live_birth",
        1,
        5,
        plot_params={"facecolor": "orange"},
    )
)
daft_net.add_node(daft.Node("num_relatives", r"num_relatives", 1, 6))
daft_net.add_node(daft.Node("age", r"age", 1, 7))
daft_net.add_node(
    daft.Node(
        "architectural_distortion",
        r"architectural_distortion",
        1,
        1,
        plot_params={"facecolor": "orange"},
    )
)
daft_net.add_node(daft.Node("asymmetry", r"asymmetry", 3, 1))
daft_net.add_node(daft.Node("developing_density", r"developing_density", 5, 1))
daft_net.add_node(
    daft.Node(
        "breast_cancer", r"breast_cancer", 3, 4, plot_params={"facecolor": "green"}
    )
)
daft_net.add_node(daft.Node("mass", r"mass", 5, 6))
daft_net.add_node(daft.Node("mass_present", r"mass_present", 5, 5))
daft_net.add_node(daft.Node("calcification_present", r"calcification_present", 5, 4))
daft_net.add_node(
    daft.Node(
        "calcification", r"calcification", 5, 3, plot_params={"facecolor": "orange"}
    )
)

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
daft_net.add_node(
    daft.Node(
        "mass_location", r"mass_location", 8, 7, plot_params={"facecolor": "orange"}
    )
)
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

pgmpy_net = BayesianNetwork(
    [(edge.node1.name, edge.node2.name) for edge in daft_net._edges]
)
cpd_dict = {
    "breast_cancer": TabularCPD(
        variable="breast_cancer",
        variable_card=2,
        evidence=[
            "age",
            "num_relatives",
            "age_at_1st_live_birth",
            "age_at_menarche",
            "previous_biopsy",
        ],
        evidence_card=[3, 3, 3, 3, 2],
        values=np.random.dirichlet(alpha=np.ones(2) * 2, size=162).transpose().tolist(),
        state_names={
            "breast_cancer": ["no", "yes"],
            "age": ["young", "middle", "old"],
            "num_relatives": ["none", "some", "many"],
            "age_at_1st_live_birth": ["young", "middle", "old"],
            "age_at_menarche": ["young", "middle", "old"],
            "previous_biopsy": ["no", "yes"],
        },
    ),
    "age": TabularCPD(
        variable="age",
        variable_card=3,
        values=[[i] for i in np.random.dirichlet(alpha=np.ones(3) * 2, size=1)[0]],
        state_names={"age": ["young", "middle", "old"]},
    ),
    "num_relatives": TabularCPD(
        variable="num_relatives",
        variable_card=3,
        values=[[i] for i in np.random.dirichlet(alpha=np.ones(3) * 2, size=1)[0]],
        state_names={"num_relatives": ["none", "some", "many"]},
    ),
    "age_at_1st_live_birth": TabularCPD(
        variable="age_at_1st_live_birth",
        variable_card=3,
        values=[[i] for i in np.random.dirichlet(alpha=np.ones(3) * 2, size=1)[0]],
        state_names={"age_at_1st_live_birth": ["young", "middle", "old"]},
    ),
    "age_at_menarche": TabularCPD(
        variable="age_at_menarche",
        variable_card=3,
        values=[[i] for i in np.random.dirichlet(alpha=np.ones(3) * 2, size=1)[0]],
        state_names={"age_at_menarche": ["young", "middle", "old"]},
    ),
    "previous_biopsy": TabularCPD(
        variable="previous_biopsy",
        variable_card=2,
        values=[[i] for i in np.random.dirichlet(alpha=np.ones(2) * 2, size=1)[0]],
        state_names={"previous_biopsy": ["no", "yes"]},
    ),
    "pain": TabularCPD(
        #  breast_cancer | no | yes |
        #  --------------|----|-----|
        #  pain=no       | x  |  x  |
        #  pain=yes      | x  |  x  |
        # (columns sum to 1.0)
        variable="pain",
        variable_card=2,
        evidence=["breast_cancer"],
        evidence_card=[2],
        values=np.random.dirichlet(alpha=np.ones(2) * 2, size=2).transpose().tolist(),
        state_names={
            "breast_cancer": ["no", "yes"],
            "pain": ["no", "yes"],
        },
    ),
    "nipple_discharge": TabularCPD(
        #  breast_cancer        | no | yes |
        #  ---------------------|---|---|
        #  nipple_discharge=no  | x | x |
        #  nipple_discharge=yes | x | x |
        # (columns sum to 1.0)
        variable="nipple_discharge",
        variable_card=2,
        evidence=["breast_cancer"],
        evidence_card=[2],
        values=np.random.dirichlet(alpha=np.ones(2) * 2, size=2).transpose().tolist(),
        state_names={
            "nipple_discharge": ["no", "yes"],
            "breast_cancer": ["no", "yes"],
        },
    ),
    "asymmetry": TabularCPD(
        #  breast_cancer  | no | yes |
        #  ---------------|----|-----|
        #  asymmetry=no   | x  |  x  |
        #  asymmetry=yes  | x  |  x  |
        # (columns sum to 1.0)
        variable="asymmetry",
        variable_card=2,
        evidence=["breast_cancer"],
        evidence_card=[2],
        values=np.random.dirichlet(alpha=np.ones(2) * 2, size=2).transpose().tolist(),
        state_names={
            "asymmetry": ["no", "yes"],
            "breast_cancer": ["no", "yes"],
        },
    ),
    "developing_density": TabularCPD(
        #  breast_cancer           | no | yes |
        #  ------------------------|----|-----|
        #  developing_density=low  | x  |  x  |
        #  developing_density=med  | x  |  x  |
        #  developing_density=high | x  |  x  |
        # (columns sum to 1.0)
        variable="developing_density",
        variable_card=3,
        evidence=["breast_cancer"],
        evidence_card=[2],
        values=np.random.dirichlet(alpha=np.ones(3) * 2, size=2).transpose().tolist(),
        state_names={
            "developing_density": ["low", "med", "high"],
            "breast_cancer": ["no", "yes"],
        },
    ),
    "architectural_distortion": TabularCPD(
        # previous_biopsy               | no       | yes      |
        # breast_cancer                 | no | yes | no | yes |
        # ------------------------------|----|-----|----|-----|
        # architectural_distortion=low  | x  |  x  | x  |  x  |
        # architectural_distortion=med  | x  |  x  | x  |  x  |
        # architectural_distortion=high | x  |  x  | x  |  x  |
        # (columns sum to 1.0)
        variable="architectural_distortion",
        variable_card=3,
        evidence=["previous_biopsy", "breast_cancer"],
        evidence_card=[2, 2],
        values=np.random.dirichlet(alpha=np.ones(3) * 2, size=4).transpose().tolist(),
        state_names={
            "architectural_distortion": ["low", "med", "high"],
            "previous_biopsy": ["no", "yes"],
            "breast_cancer": ["no", "yes"],
        },
    ),
    "mass": TabularCPD(
        #  breast_cancer | no | yes |
        #  --------------|----|-----|
        #  mass=low      | x  |  x  |
        #  mass=med      | x  |  x  |
        #  mass=high     | x  |  x  |
        # (columns sum to 1.0)
        variable="mass",
        variable_card=3,
        evidence=["breast_cancer"],
        evidence_card=[2],
        values=np.random.dirichlet(alpha=np.ones(3) * 2, size=2).transpose().tolist(),
        state_names={
            "mass": ["low", "med", "high"],
            "breast_cancer": ["no", "yes"],
        },
    ),
    "mass_present": TabularCPD(
        #  mass             | low | med | high |
        #  -----------------|-----|-----|------|
        #  mass_present=no  | x   |  x  |  x   |
        #  mass_present=yes | x   |  x  |  x   |
        # (columns sum to 1.0)
        variable="mass_present",
        variable_card=2,
        evidence=["mass"],
        evidence_card=[3],
        values=np.random.dirichlet(alpha=np.ones(2) * 2, size=3).transpose().tolist(),
        state_names={"mass": ["low", "med", "high"], "mass_present": ["no", "yes"]},
    ),
    "calcification": TabularCPD(
        #  breast_cancer     | no | yes |
        #  ------------------|----|-----|
        #  calcification=no  | x  |  x  |
        #  calcification=yes | x  |  x  |
        # (columns sum to 1.0)
        variable="calcification",
        variable_card=2,
        evidence=["breast_cancer"],
        evidence_card=[2],
        values=np.random.dirichlet(alpha=np.ones(2) * 2, size=2).transpose().tolist(),
        state_names={
            "calcification": ["no", "yes"],
            "breast_cancer": ["no", "yes"],
        },
    ),
    "calcification_present": TabularCPD(
        # calcification             | no | yes |
        # --------------------------|----|-----|
        # calcification_present=no  | x  |  x  |
        # calcification_present=yes | x  |  x  |
        # (columns sum to 1.0)
        variable="calcification_present",
        variable_card=2,
        evidence=["calcification"],
        evidence_card=[2],
        values=np.random.dirichlet(alpha=np.ones(2) * 2, size=2).transpose().tolist(),
        state_names={
            "calcification": ["no", "yes"],
            "calcification_present": ["no", "yes"],
        },
    ),
    "mass_margin": TabularCPD(
        # mass                      | low | med | high |
        # --------------------------|-----|-----|------|
        # mass_margin=low           |  x  |  x  |   x  |
        # mass_margin=high          |  x  |  x  |   x  |
        # (columns sum to 1.0)
        variable="mass_margin",
        variable_card=2,
        evidence=["mass"],
        evidence_card=[3],
        values=np.random.dirichlet(alpha=np.ones(2) * 2, size=3).transpose().tolist(),
        state_names={
            "mass": ["low", "med", "high"],
            "mass_margin": ["low", "high"],
        },
    ),
    "mass_density": TabularCPD(
        # mass                      | low | med | high |
        # --------------------------|-----|-----|------|
        # mass_density=low          |  x  |  x  |   x  |
        # mass_density=high         |  x  |  x  |   x  |
        # (columns sum to 1.0)
        variable="mass_density",
        variable_card=2,
        evidence=["mass"],
        evidence_card=[3],
        values=np.random.dirichlet(alpha=np.ones(2) * 2, size=3).transpose().tolist(),
        state_names={
            "mass": ["low", "med", "high"],
            "mass_density": ["low", "high"],
        },
    ),
    "halo_sign": TabularCPD(
        # mass                   | low | med | high |
        # -----------------------|-----|-----|------|
        # halo_sign=low          |  x  |  x  |   x  |
        # halo_sign=high         |  x  |  x  |   x  |
        # (columns sum to 1.0)
        variable="halo_sign",
        variable_card=2,
        evidence=["mass"],
        evidence_card=[3],
        values=np.random.dirichlet(alpha=np.ones(2) * 2, size=3).transpose().tolist(),
        state_names={
            "mass": ["low", "med", "high"],
            "halo_sign": ["low", "high"],
        },
    ),
    "mass_location": TabularCPD(
        # mass                   | low | med | high |
        # -----------------------|-----|-----|------|
        # mass_location=bottom   |  x  |  x  |   x  |
        # mass_location=centre   |  x  |  x  |   x  |
        # mass_location=side     |  x  |  x  |   x  |
        # (columns sum to 1.0)
        variable="mass_location",
        variable_card=3,
        evidence=["mass"],
        evidence_card=[3],
        values=np.random.dirichlet(alpha=np.ones(3) * 2, size=3).transpose().tolist(),
        state_names={
            "mass": ["low", "med", "high"],
            "mass_location": ["bottom", "centre", "side"],
        },
    ),
    "calcification_cluster_shape": TabularCPD(
        # calcification                   | no | yes |
        # --------------------------------|----|-----|
        # calcification_cluster_shape=A   |  x |  x  |
        # calcification_cluster_shape=B   |  x |  x  |
        # calcification_cluster_shape=C   |  x |  x  |
        # (columns sum to 1.0)
        variable="calcification_cluster_shape",
        variable_card=3,
        evidence=["calcification"],
        evidence_card=[2],
        values=np.random.dirichlet(alpha=np.ones(3) * 2, size=2).transpose().tolist(),
        state_names={
            "calcification_cluster_shape": ["A", "B", "C"],
            "calcification": ["no", "yes"],
        },
    ),
    "num_in_cluster": TabularCPD(
        # calcification         | no | yes |
        # ----------------------|----|-----|
        # num_in_cluster=none   |  x |  x  |
        # num_in_cluster=some   |  x |  x  |
        # num_in_cluster=many   |  x |  x  |
        # (columns sum to 1.0)
        variable="num_in_cluster",
        variable_card=3,
        evidence=["calcification"],
        evidence_card=[2],
        values=np.random.dirichlet(alpha=np.ones(3) * 2, size=2).transpose().tolist(),
        state_names={
            "num_in_cluster": ["none", "some", "many"],
            "calcification": ["no", "yes"],
        },
    ),
    "calcification_shape": TabularCPD(
        # calcification         | no | yes |
        # ----------------------|----|-----|
        # calcification_shape=A |  x |  x  |
        # calcification_shape=B |  x |  x  |
        # (columns sum to 1.0)
        variable="calcification_shape",
        variable_card=2,
        evidence=["calcification"],
        evidence_card=[2],
        values=np.random.dirichlet(alpha=np.ones(2) * 2, size=2).transpose().tolist(),
        state_names={
            "calcification_shape": ["A", "B"],
            "calcification": ["no", "yes"],
        },
    ),
    "calcification_density": TabularCPD(
        # calcification               | no | yes |
        # ----------------------------|----|-----|
        # calcification_density=low   |  x |  x  |
        # calcification_density=med   |  x |  x  |
        # calcification_density=high  |  x |  x  |
        # (columns sum to 1.0)
        variable="calcification_density",
        variable_card=3,
        evidence=["calcification"],
        evidence_card=[2],
        values=np.random.dirichlet(alpha=np.ones(3) * 2, size=2).transpose().tolist(),
        state_names={
            "calcification_density": ["low", "med", "high"],
            "calcification": ["no", "yes"],
        },
    ),
    "calcification_arrangement": TabularCPD(
        # calcification               | no | yes |
        # ----------------------------|----|-----|
        # calcification_arrangement=X |  x |  x  |
        # calcification_arrangement=Y |  x |  x  |
        # (columns sum to 1.0)
        variable="calcification_arrangement",
        variable_card=2,
        evidence=["calcification"],
        evidence_card=[2],
        values=np.random.dirichlet(alpha=np.ones(2) * 2, size=2).transpose().tolist(),
        state_names={
            "calcification_arrangement": ["X", "Y"],
            "calcification": ["no", "yes"],
        },
    ),
    "calcification_size": TabularCPD(
        # calcification              | no | yes |
        # ---------------------------|----|-----|
        # calcification_size=small   |  x |  x  |
        # calcification_size=med     |  x |  x  |
        # calcification_size=large   |  x |  x  |
        # (columns sum to 1.0)
        variable="calcification_size",
        variable_card=3,
        evidence=["calcification"],
        evidence_card=[2],
        values=np.random.dirichlet(alpha=np.ones(3) * 2, size=2).transpose().tolist(),
        state_names={
            "calcification_size": ["small", "med", "large"],
            "calcification": ["no", "yes"],
        },
    ),
}

for cpd in cpd_dict.values():
    pgmpy_net.add_cpds(cpd)

assert pgmpy_net.check_model(), "pgmpy model incorrectly specified"

can_control_varnames = set(
    [
        "age_at_1st_live_birth",
        "architectural_distortion",
        "calcification",
        "mass_location",
    ]
)
outcome_varnames = set(["breast_cancer"])
observed_only_varnames = (
    set(pgmpy_net._node.keys()) - can_control_varnames - outcome_varnames
)

mammonet_system = utils.true_system_bayes_network(
    system_name="MammoNet",
    system_source="[paper] Construction of a Bayesian Network for Mammographic Diagnosis of Breast Cancer Kahn Jr et al. (1996)",
    n_vars_in_system=pgmpy_net.size(),
    can_control_varnames=list(can_control_varnames),
    outcome_varnames=list(outcome_varnames),
    observed_only_varnames=list(observed_only_varnames),
    daft_model=daft_net,
    pgmpy_bayes_network_model=pgmpy_net,
    model_train_data=pgmpy_net.simulate(1_000),
    model_test_data=pgmpy_net.simulate(500),
)

if __name__ == "__main__":
    daft_net.render(dpi=80)

    # overwrite the string truncation method in TabularCPD class to print full CPDs:
    TabularCPD._truncate_strtable = lambda self, x: x
    for x_name in cpd_dict:
        print(cpd_dict[x_name])
