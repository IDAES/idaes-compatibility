#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES), and is copyright (c) 2018-2021
# by the software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia University
# Research Corporation, et al.  All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and
# license information.
#################################################################################
import json
from math import isclose


def compare_results_to_baseline(
    current_results, baseline_results, rel_tol=0.1, abs_tol=1
):
    """Return sample indices whose convergence behavior regressed."""
    current = {str(k): v for k, v in current_results.items()}
    baseline = {str(k): v for k, v in baseline_results.items()}
    regressions = {
        "sample_set": [],
        "success": [],
        "iters": [],
        "iters_in_restoration": [],
        "iters_w_regularization": [],
        "numerical_issues": [],
    }

    if current.keys() != baseline.keys():
        regressions["sample_set"] = sorted(current.keys() ^ baseline.keys())
        return regressions

    metrics = (
        "iters",
        "iters_in_restoration",
        "iters_w_regularization",
    )
    for sample, expected in baseline.items():
        observed = current[sample]
        if expected["success"] and not observed["success"]:
            regressions["success"].append(sample)
            continue
        if not expected["success"] and observed["success"]:
            continue

        expected_stats = expected["results"]
        observed_stats = observed["results"]
        for metric in metrics:
            expected_value = expected_stats[metric]
            observed_value = observed_stats[metric]
            if observed_value > expected_value and not isclose(
                observed_value,
                expected_value,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            ):
                regressions[metric].append(sample)

        if not expected_stats["numerical_issues"] and observed_stats[
            "numerical_issues"
        ]:
            regressions["numerical_issues"].append(sample)

    return regressions


def assert_baseline_not_regressed(
    convergence_analysis, filename, rel_tol=0.1, abs_tol=1
):
    """Run a saved convergence sweep and reject only degraded behavior."""
    with open(filename, "r") as baseline_file:
        baseline = json.load(baseline_file)

    convergence_analysis.run_convergence_analysis_from_dict(baseline)
    regressions = compare_results_to_baseline(
        convergence_analysis.results,
        baseline["results"],
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )
    regressions = {k: v for k, v in regressions.items() if v}
    if regressions:
        raise AssertionError(f"Convergence regressions detected: {regressions}")
