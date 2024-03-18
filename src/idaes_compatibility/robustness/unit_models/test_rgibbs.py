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
"""
This module contains the code for convergence testing of the
Gibbs reactor model
"""
import pytest
import os
import json

from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Suffix,
    TransformationFactory,
)
from pyomo.common.fileutils import this_file_dir

from idaes.core import FlowsheetBlock
from idaes.models.unit_models.gibbs_reactor import GibbsReactor
from idaes.models.properties.activity_coeff_models.methane_combustion_ideal import (
    MethaneParameterBlock as MethaneCombustionParameterBlock,
)
from idaes.core.initialization import (
    BlockTriangularizationInitializer,
    InitializationStatus,
)

from idaes.core.surrogate.pysmo.sampling import LatinHypercubeSampling
from idaes.core.util.parameter_sweep import ParameterSweepSpecification
from idaes.core.util.model_diagnostics import IpoptConvergenceAnalysis


currdir = this_file_dir()
fname = os.path.join(currdir, "rgibbs.json")


def build_model():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.properties = MethaneCombustionParameterBlock()

    m.fs.unit = GibbsReactor(
        property_package=m.fs.properties,
        has_heat_transfer=True,
        has_pressure_change=True,
    )

    m.fs.unit.inlet.flow_mol[0].fix(230.0)
    m.fs.unit.inlet.mole_frac_comp[0, "H2"].fix(0.0435)
    m.fs.unit.inlet.mole_frac_comp[0, "N2"].fix(0.6522)
    m.fs.unit.inlet.mole_frac_comp[0, "O2"].fix(0.1739)
    m.fs.unit.inlet.mole_frac_comp[0, "CO2"].fix(1e-5)
    m.fs.unit.inlet.mole_frac_comp[0, "CH4"].fix(0.1304)
    m.fs.unit.inlet.mole_frac_comp[0, "CO"].fix(1e-5)
    m.fs.unit.inlet.mole_frac_comp[0, "H2O"].fix(1e-5)
    m.fs.unit.inlet.mole_frac_comp[0, "NH3"].fix(1e-5)
    m.fs.unit.inlet.temperature[0].fix(1500.0)
    m.fs.unit.inlet.pressure[0].fix(101325.0)

    m.fs.unit.outlet.temperature[0].fix(2844.38)
    m.fs.unit.deltaP.fix(0)

    # Fix some bounds to avoid potential log(0)
    # TODO: This really should be fixed in the property package, but breaks other tests
    m.fs.unit.control_volume.properties_out[0].pressure.setlb(1000)
    m.fs.unit.control_volume.properties_out[0].mole_frac_phase_comp.setlb(1e-12)

    initializer = BlockTriangularizationInitializer(constraint_tolerance=2e-5)
    initializer.initialize(
        m.fs.unit,
        initial_guesses={
            "control_volume.properties_out[0].pressure": 101325.0,
            "control_volume.properties_out[0].flow_mol": 251.05,
            "control_volume.properties_out[0].mole_frac_comp[CH4]": 1e-5,
            "control_volume.properties_out[0].mole_frac_comp[CO]": 0.0916,
            "control_volume.properties_out[0].mole_frac_comp[CO2]": 0.0281,
            "control_volume.properties_out[0].mole_frac_comp[H2]": 0.1155,
            "control_volume.properties_out[0].mole_frac_comp[H2O]": 0.1633,
            "control_volume.properties_out[0].mole_frac_comp[N2]": 0.59478,
            "control_volume.properties_out[0].mole_frac_comp[NH3]": 1e-5,
            "control_volume.properties_out[0].mole_frac_comp[O2]": 0.0067,
        },
    )

    assert initializer.summary[m.fs.unit]["status"] == InitializationStatus.Ok

    m.fs.sum_mol_frac = Constraint(
        expr=1==sum(m.fs.unit.inlet.mole_frac_comp[0, j] for j in m.fs.properties.component_list)
    )
    m.fs.unit.inlet.mole_frac_comp[0, "N2"].unfix()

    m.scaling_factor = Suffix(direction=Suffix.EXPORT)

    m.scaling_factor[
        m.fs.unit.control_volume.element_balances[0.0, "C"]
    ] = 0.0038968315684515787
    m.scaling_factor[
        m.fs.unit.control_volume.element_balances[0.0, "H"]
    ] = 0.0009690314543471861
    m.scaling_factor[
        m.fs.unit.control_volume.element_balances[0.0, "N"]
    ] = 0.0016665906198716563
    m.scaling_factor[
        m.fs.unit.control_volume.element_balances[0.0, "O"]
    ] = 0.0067608566657646
    m.scaling_factor[
        m.fs.unit.control_volume.enthalpy_balances[0.0]
    ] = 6.343688225967796e-08
    m.scaling_factor[
        m.fs.unit.control_volume.heat[0.0]
    ] = 1.3415588575040103e-07
    m.scaling_factor[
        m.fs.unit.control_volume.pressure_balance[0.0]
    ] = 9.869232667160129e-06
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].enth_mol_phase["Vap"]
    ] = 0.00010271414106049353
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].enth_mol_phase_comp[
            "Vap", "CH4"
        ]
    ] = 1.3404825737265415e-05
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].enth_mol_phase_comp[
            "Vap", "CO2"
        ]
    ] = 2.5411669038422445e-06
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].enth_mol_phase_comp[
            "Vap", "CO"
        ]
    ] = 9.047317470370035e-06
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].enth_mol_phase_comp[
            "Vap", "H2O"
        ]
    ] = 4.135136252739528e-06
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].enth_mol_phase_comp[
            "Vap", "H2"
        ]
    ] = 1
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].enth_mol_phase_comp[
            "Vap", "N2"
        ]
    ] = 1
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].enth_mol_phase_comp[
            "Vap", "NH3"
        ]
    ] = 2.1786492374727668e-05
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].enth_mol_phase_comp[
            "Vap", "O2"
        ]
    ] = 1
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].eq_comp["CH4"]
    ] = 0.03334222459322486
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].eq_comp["CO2"]
    ] = 434.782608695652
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].eq_comp["CO"]
    ] = 434.782608695652
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].eq_comp["H2O"]
    ] = 434.782608695652
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].eq_comp["H2"]
    ] = 0.09995002498750626
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].eq_comp["N2"]
    ] = 0.00666640001066624
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].eq_comp["NH3"]
    ] = 434.782608695652
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].eq_comp["O2"]
    ] = 0.025001875140635548
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].eq_enth_mol_phase["Vap"]
    ] = 5.9334197643529735e-05
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].eq_enth_mol_phase_comp[
            "Vap", "CH4"
        ]
    ] = 1.3404825737265415e-05
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].eq_enth_mol_phase_comp[
            "Vap", "CO2"
        ]
    ] = 2.5411669038422445e-06
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].eq_enth_mol_phase_comp[
            "Vap", "CO"
        ]
    ] = 9.047317470370035e-06
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].eq_enth_mol_phase_comp[
            "Vap", "H2O"
        ]
    ] = 4.135136252739528e-06
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].eq_enth_mol_phase_comp[
            "Vap", "H2"
        ]
    ] = 1.0
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].eq_enth_mol_phase_comp[
            "Vap", "N2"
        ]
    ] = 1.0
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].eq_enth_mol_phase_comp[
            "Vap", "NH3"
        ]
    ] = 2.1786492374727668e-05
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].eq_enth_mol_phase_comp[
            "Vap", "O2"
        ]
    ] = 1.0
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].eq_total
    ] = 0.004347826086956522
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].flow_mol_phase["Vap"]
    ] = 0.004347826086956522
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].mole_frac_phase_comp[
            "Vap", "CH4"
        ]
    ] = 7.668711656441719
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].mole_frac_phase_comp[
            "Vap", "CO2"
        ]
    ] = 99999.99999999997
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].mole_frac_phase_comp[
            "Vap", "CO"
        ]
    ] = 99999.99999999997
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].mole_frac_phase_comp[
            "Vap", "H2O"
        ]
    ] = 99999.99999999997
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].mole_frac_phase_comp[
            "Vap", "H2"
        ]
    ] = 22.98850574712644
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].mole_frac_phase_comp[
            "Vap", "N2"
        ]
    ] = 1.5332720024532351
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].mole_frac_phase_comp[
            "Vap", "NH3"
        ]
    ] = 99999.99999999997
    m.scaling_factor[
        m.fs.unit.control_volume.properties_in[0.0].mole_frac_phase_comp[
            "Vap", "O2"
        ]
    ] = 5.750431282346176
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].enth_mol_phase["Vap"]
    ] = 2.5797225634987406e-05
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].enth_mol_phase_comp[
            "Vap", "CH4"
        ]
    ] = 9.131356578608373e-05
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].enth_mol_phase_comp[
            "Vap", "CO2"
        ]
    ] = 3.8432408761344524e-06
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].enth_mol_phase_comp[
            "Vap", "CO"
        ]
    ] = 8.617482264231038e-06
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].enth_mol_phase_comp[
            "Vap", "H2O"
        ]
    ] = 5.614692038136825e-06
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].enth_mol_phase_comp[
            "Vap", "H2"
        ]
    ] = 1.1286872760356127e-05
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].enth_mol_phase_comp[
            "Vap", "N2"
        ]
    ] = 0.0007529656318755848
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].enth_mol_phase_comp[
            "Vap", "NH3"
        ]
    ] = 0.000147611121632822
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].enth_mol_phase_comp[
            "Vap", "O2"
        ]
    ] = 0.00011456899776931075
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].entr_mol_phase_comp[
            "Vap", "CH4"
        ]
    ] = 0.002273113949091129
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].entr_mol_phase_comp[
            "Vap", "CO2"
        ]
    ] = 0.0032746835561418483
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].entr_mol_phase_comp[
            "Vap", "CO"
        ]
    ] = 0.004593745058364088
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].entr_mol_phase_comp[
            "Vap", "H2O"
        ]
    ] = 0.004278478461249582
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].entr_mol_phase_comp[
            "Vap", "H2"
        ]
    ] = 0.005287458822573278
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].entr_mol_phase_comp[
            "Vap", "N2"
        ]
    ] = 0.005016168207707185
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].entr_mol_phase_comp[
            "Vap", "NH3"
        ]
    ] = 0.0029937456289480776
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].entr_mol_phase_comp[
            "Vap", "O2"
        ]
    ] = 0.003654125560818891
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].eq_comp["CH4"]
    ] = 0.002827763345315764
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].eq_comp["CO2"]
    ] = 0.08851222463945824
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].eq_comp["CO"]
    ] = 0.020535852025143013
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].eq_comp["H2O"]
    ] = 0.011302753862264373
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].eq_comp["H2"]
    ] = 0.019411801141636542
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].eq_comp["N2"]
    ] = 0.0033331760499149495
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].eq_comp["NH3"]
    ] = 3476.4619148053275
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].eq_comp["O2"]
    ] = 8.288029770353534
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].eq_enth_mol_phase["Vap"]
    ] = 1.5616764299049252e-05
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].eq_enth_mol_phase_comp[
            "Vap", "CH4"
        ]
    ] = 9.131356578608373e-05
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].eq_enth_mol_phase_comp[
            "Vap", "CO2"
        ]
    ] = 3.8432408761344524e-06
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].eq_enth_mol_phase_comp[
            "Vap", "CO"
        ]
    ] = 8.617482264231038e-06
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].eq_enth_mol_phase_comp[
            "Vap", "H2O"
        ]
    ] = 5.614692038136825e-06
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].eq_enth_mol_phase_comp[
            "Vap", "H2"
        ]
    ] = 1.1286872760356127e-05
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].eq_enth_mol_phase_comp[
            "Vap", "N2"
        ]
    ] = 0.0007529656318755848
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].eq_enth_mol_phase_comp[
            "Vap", "NH3"
        ]
    ] = 0.000147611121632822
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].eq_enth_mol_phase_comp[
            "Vap", "O2"
        ]
    ] = 0.00011456899776931075
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].eq_entr_mol_phase_comp[
            "Vap", "CH4"
        ]
    ] = 1.2281784483868407e-12
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].eq_entr_mol_phase_comp[
            "Vap", "CO2"
        ]
    ] = 0.0032722586564032413
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].eq_entr_mol_phase_comp[
            "Vap", "CO"
        ]
    ] = 0.004587058233184441
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].eq_entr_mol_phase_comp[
            "Vap", "H2O"
        ]
    ] = 0.004273074490321421
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].eq_entr_mol_phase_comp[
            "Vap", "H2"
        ]
    ] = 0.005277269330062221
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].eq_entr_mol_phase_comp[
            "Vap", "N2"
        ]
    ] = 0.005007465522090417
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].eq_entr_mol_phase_comp[
            "Vap", "NH3"
        ]
    ] = 0.0029918924828130663
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].eq_entr_mol_phase_comp[
            "Vap", "O2"
        ]
    ] = 0.003650757205096906
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].eq_gibbs_mol_phase_comp[
            "Vap", "CH4"
        ]
    ] = 5.675584743853875e-07
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].eq_gibbs_mol_phase_comp[
            "Vap", "CO2"
        ]
    ] = 6.906678667100022e-07
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].eq_gibbs_mol_phase_comp[
            "Vap", "CO"
        ]
    ] = 1.0328408888379269e-06
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].eq_gibbs_mol_phase_comp[
            "Vap", "H2O"
        ]
    ] = 9.189402632308884e-07
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].eq_gibbs_mol_phase_comp[
            "Vap", "H2"
        ]
    ] = 1.415412212742754e-06
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].eq_gibbs_mol_phase_comp[
            "Vap", "N2"
        ]
    ] = 1.248468142622135e-06
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].eq_gibbs_mol_phase_comp[
            "Vap", "NH3"
        ]
    ] = 7.41580819400193e-07
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].eq_gibbs_mol_phase_comp[
            "Vap", "O2"
        ]
    ] = 9.033010042075156e-07
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].eq_mol_frac_out
    ] = 0.8416262137210224
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].eq_total
    ] = 0.002827763345315764
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].flow_mol
    ] = 0.003999061274127067
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].flow_mol_phase["Vap"]
    ] = 0.003999061274127067
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].gibbs_mol_phase_comp[
            "Vap", "CH4"
        ]
    ] = 8.062155808993221e-07
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].gibbs_mol_phase_comp[
            "Vap", "CO2"
        ]
    ] = 8.859012593383004e-07
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].gibbs_mol_phase_comp[
            "Vap", "CO"
        ]
    ] = 1.3601211872489997e-06
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].gibbs_mol_phase_comp[
            "Vap", "H2O"
        ]
    ] = 1.1863588686277076e-06
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].gibbs_mol_phase_comp[
            "Vap", "H2"
        ]
    ] = 2.225437251662786e-06
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].gibbs_mol_phase_comp[
            "Vap", "N2"
        ]
    ] = 1.767676703310131e-06
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].gibbs_mol_phase_comp[
            "Vap", "NH3"
        ]
    ] = 1.0450609388337302e-06
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].gibbs_mol_phase_comp[
            "Vap", "O2"
        ]
    ] = 1.2704369868483197e-06
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].mole_frac_comp["CH4"]
    ] = 1
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].mole_frac_comp["CO2"]
    ] = 44.26650084714147
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].mole_frac_comp["CO"]
    ] = 10.270336270166638
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].mole_frac_comp["H2O"]
    ] = 5.6527035158927825
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].mole_frac_comp["H2"]
    ] = 9.70817890049651
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].mole_frac_comp["N2"]
    ] = 1.6669792340916454
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].mole_frac_comp["NH3"]
    ] = 1738638.9837520984
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].mole_frac_comp["O2"]
    ] = 4144.987636961217
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].mole_frac_phase_comp[
            "Vap", "CH4"
        ]
    ] = 1
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].mole_frac_phase_comp[
            "Vap", "CO2"
        ]
    ] = 44.26650084714147
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].mole_frac_phase_comp[
            "Vap", "CO"
        ]
    ] = 10.270336270166638
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].mole_frac_phase_comp[
            "Vap", "H2O"
        ]
    ] = 5.6527035158927825
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].mole_frac_phase_comp[
            "Vap", "H2"
        ]
    ] = 9.70817890049651
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].mole_frac_phase_comp[
            "Vap", "N2"
        ]
    ] = 1.6669792340916454
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].mole_frac_phase_comp[
            "Vap", "NH3"
        ]
    ] = 1738638.9837520984
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].mole_frac_phase_comp[
            "Vap", "O2"
        ]
    ] = 4144.987636961217
    m.scaling_factor[
        m.fs.unit.control_volume.properties_out[0.0].pressure
    ] = 9.869232667160129e-06
    m.scaling_factor[
        m.fs.unit.gibbs_minimization[0.0, "Vap", "CH4"]
    ] = 6.37201815837067e-07
    m.scaling_factor[
        m.fs.unit.gibbs_minimization[0.0, "Vap", "CO2"]
    ] = 7.052606407804767e-07
    m.scaling_factor[
        m.fs.unit.gibbs_minimization[0.0, "Vap", "CO"]
    ] = 1.1096130740800528e-06
    m.scaling_factor[
        m.fs.unit.gibbs_minimization[0.0, "Vap", "H2O"]
    ] = 9.679516364316312e-07
    m.scaling_factor[
        m.fs.unit.gibbs_minimization[0.0, "Vap", "H2"]
    ] = 1.5736217717559093e-06
    m.scaling_factor[
        m.fs.unit.gibbs_minimization[0.0, "Vap", "N2"]
    ] = 1.2499361838560746e-06
    m.scaling_factor[
        m.fs.unit.gibbs_minimization[0.0, "Vap", "NH3"]
    ] = 8.304717457164266e-07
    m.scaling_factor[
        m.fs.unit.gibbs_minimization[0.0, "Vap", "O2"]
    ] = 8.983346084706515e-07
    m.scaling_factor[
        m.fs.unit.lagrange_mult[0.0, "C"]
    ] = 2.926858666284934e-06
    m.scaling_factor[
        m.fs.unit.lagrange_mult[0.0, "H"]
    ] = 4.450874503325572e-06
    m.scaling_factor[
        m.fs.unit.lagrange_mult[0.0, "N"]
    ] = 3.535353406620262e-06
    m.scaling_factor[
        m.fs.unit.lagrange_mult[0.0, "O"]
    ] = 2.5408739736966393e-06

    scaling = TransformationFactory("core.scale_model")
    sm = scaling.create_using(m, rename=False)

    return sm


def generate_baseline():
    model = build_model()

    spec = ParameterSweepSpecification()
    spec.add_sampled_input("fs.unit.inlet.flow_mol[0]", lower=10, upper=1000)
    spec.add_sampled_input("fs.unit.inlet.mole_frac_comp[0,H2]", lower=1e-5, upper=0.1)
    spec.add_sampled_input("fs.unit.inlet.mole_frac_comp[0,CH4]", lower=1e-5, upper=0.2)
    spec.add_sampled_input("fs.unit.inlet.mole_frac_comp[0,H2]", lower=1e-5, upper=0.2)
    spec.add_sampled_input("fs.unit.inlet.mole_frac_comp[0,H2O]", lower=1e-5, upper=0.1)
    spec.add_sampled_input("fs.unit.inlet.mole_frac_comp[0,O2]", lower=1e-5, upper=0.2)
    spec.add_sampled_input("fs.unit.inlet.pressure[0]", lower=5e4, upper=4e5)
    spec.add_sampled_input("fs.unit.inlet.temperature[0]", lower=1000, upper=2000)
    spec.add_sampled_input("fs.unit.outlet.temperature[0]", lower=2000, upper=3000)
    spec.add_sampled_input("fs.unit.deltaP[0]", lower=0, upper=1e5)
    spec.set_sampling_method(LatinHypercubeSampling)
    spec.set_sample_size(200)

    spec.generate_samples()

    ca = IpoptConvergenceAnalysis(
        model,
        input_specification=spec,
    )

    ca.run_convergence_analysis()

    ca.to_json_file(fname)

    ca.report_convergence_summary()


def test_gibbs_reactor_robustness():
    model = build_model()
    ca = IpoptConvergenceAnalysis(model)

    ca.assert_baseline_comparison(fname)


if __name__ == "__main__":
    generate_baseline()
