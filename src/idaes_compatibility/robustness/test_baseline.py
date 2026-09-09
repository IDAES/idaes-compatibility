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
from copy import deepcopy

from idaes_compatibility.robustness.baseline import compare_results_to_baseline


def _result(
    success=True, iters=10, restoration=2, regularization=1, warnings=False
):
    return {
        "success": success,
        "results": {
            "iters": iters,
            "iters_in_restoration": restoration,
            "iters_w_regularization": regularization,
            "numerical_issues": warnings,
        },
    }


def test_improvements_are_not_regressions():
    baseline = {"0": _result(), "1": _result(success=False)}
    current = {0: _result(iters=5), 1: _result(success=True)}

    regressions = compare_results_to_baseline(current, baseline)

    assert all(not samples for samples in regressions.values())


def test_degraded_results_are_reported():
    baseline = {str(i): _result() for i in range(5)}
    current = deepcopy(baseline)
    current["0"]["success"] = False
    current["1"]["results"]["iters"] = 13
    current["2"]["results"]["iters_in_restoration"] = 5
    current["3"]["results"]["iters_w_regularization"] = 4
    current["4"]["results"]["numerical_issues"] = True

    regressions = compare_results_to_baseline(current, baseline)

    assert regressions == {
        "sample_set": [],
        "success": ["0"],
        "iters": ["1"],
        "iters_in_restoration": ["2"],
        "iters_w_regularization": ["3"],
        "numerical_issues": ["4"],
    }


def test_sample_set_changes_are_reported():
    regressions = compare_results_to_baseline({0: _result()}, {"1": _result()})

    assert regressions["sample_set"] == ["0", "1"]
