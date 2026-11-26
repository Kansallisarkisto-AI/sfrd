from ortools.sat.python import cp_model


def fit_model(unary_potentials, pairwise_potentials, dropped_penalty=0.5, unary_weight=1.0, pairwise_weight=1.0):
    model = cp_model.CpModel()

    U, labels = unary_potentials

    # labels for every box in unary potentials
    box_labels = ...

    accumulated_unary_potential = sum(...)

    accumulated_pairwise_potential = sum(...)

    # dropped item penalty relative to closest matching line itemset in annotations
    ...
    dropped_penalty = ...

    model.maximize(unary_weight*accumulated_unary_potential + pairwise_weight*accumulated_pairwise_potential - dropped_penalty)

    solver = cp_model.CpSolver()
    solver.solve(model)

    print("Optimal labels:", ...)
    print("Total value:", solver.objective_value)