##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2019, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# Please see the files COPYRIGHT.txt and LICENSE.txt for full copyright and
# license information, respectively. Both files are also available online
# at the URL "https://github.com/IDAES/idaes-pse".
##############################################################################
import pytest
import pandas
import os

from pyomo.environ import ConcreteModel, value, units

from idaes.core import FlowsheetBlock
import idaes.models.properties.swco2 as swco2

path = os.path.dirname(os.path.abspath(__file__))

H_OFFSET = 506.778
S_OFFSET = 2.738255753


def test_swco2_liquid_phase():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.swco2 = swco2.SWCO2ParameterBlock(amount_basis=swco2.AmountBasis.MASS)
    m.fs.state = m.fs.swco2.build_state_block(m.fs.time, defined_state=True)

    # Load data
    data_file = os.path.join(path, "co2_liquid_data.csv")
    df = pandas.read_csv(data_file)

    # Flow rate does not affect any other property, so keep constant at 1
    m.fs.state[0].flow_mass.fix(1)

    for i, r in df.iterrows():
        P = r["Pressure (MPa)"]
        h = r["Enthalpy (kJ/kg)"] - H_OFFSET
        T = r["Temperature (K)"]

        print(P, T, h)

        m.fs.state[0].pressure.fix(P * units.MPa)
        m.fs.state[0].enth_mass.fix(h * units.kJ / units.kg)

        assert value(m.fs.state[0].phase_frac["Liq"]) == pytest.approx(1, rel=1e-4)

        assert value(m.fs.state[0].temperature) == pytest.approx(T, rel=1e-4)
        assert value(m.fs.state[0].dens_mass_phase["Liq"]) == pytest.approx(
            r["Density (kg/m3)"], rel=1e-3
        )
        assert value(m.fs.state[0].energy_internal_mass_phase["Liq"]) == pytest.approx(
            1e3 * (r["Internal Energy (kJ/kg)"] - H_OFFSET), rel=1e-3
        )
        assert value(m.fs.state[0].entr_mass_phase["Liq"]) == pytest.approx(
            1e3 * (r["Entropy (J/g*K)"] - S_OFFSET), rel=1e-2
        )
        assert value(m.fs.state[0].cv_mass_phase["Liq"]) == pytest.approx(
            1e3 * r["Cv (J/g*K)"], rel=2e-2
        )
        assert value(m.fs.state[0].cp_mass_phase["Liq"]) == pytest.approx(
            1e3 * r["Cp (J/g*K)"], rel=2e-2
        )
        assert value(m.fs.state[0].speed_sound_phase["Liq"]) == pytest.approx(
            r["Sound Spd. (m/s)"], rel=5e-3
        )
        # Viscosity and thermal conductivity does not match to less than 5% error.


def test_swco2_vapor_phase():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.swco2 = swco2.SWCO2ParameterBlock(amount_basis=swco2.AmountBasis.MASS)
    m.fs.state = m.fs.swco2.build_state_block(m.fs.time, defined_state=True)

    # Load data
    data_file = os.path.join(path, "co2_vapor_data.csv")
    df = pandas.read_csv(data_file)

    # Flow rate does not affect any other property, so keep constant at 1
    m.fs.state[0].flow_mass.fix(1)

    for i, r in df.iterrows():
        P = r["Pressure (MPa)"]
        h = r["Enthalpy (kJ/kg)"] - H_OFFSET
        T = r["Temperature (K)"]

        m.fs.state[0].pressure.fix(P * units.MPa)
        m.fs.state[0].enth_mass.fix(h * units.kJ / units.kg)

        assert value(m.fs.state[0].phase_frac["Vap"]) == pytest.approx(1, rel=1e-4)

        assert value(m.fs.state[0].temperature) == pytest.approx(T, rel=1e-4)
        assert value(m.fs.state[0].dens_mass_phase["Vap"]) == pytest.approx(
            r["Density (kg/m3)"], rel=5e-3
        )
        assert value(m.fs.state[0].energy_internal_mass_phase["Vap"]) == pytest.approx(
            1e3 * (r["Internal Energy (kJ/kg)"] - H_OFFSET), rel=2e-3
        )

        if not (
            (P == 1 and T == 476.59)
            or (P == 7.3589 and T == 684.13)
            or (P == 7.3589 and T == 704.13)
        ):  # Something off with these data points
            assert value(m.fs.state[0].entr_mass_phase["Vap"]) == pytest.approx(
                1e3 * (r["Entropy (J/g*K)"] - S_OFFSET), rel=2e-2
            )
        if not (
            P == 7.3589 and T == 304.13
        ):  # Something off for this point for the following properties
            assert value(m.fs.state[0].cv_mass_phase["Vap"]) == pytest.approx(
                1e3 * r["Cv (J/g*K)"], rel=2e-2
            )
            assert value(m.fs.state[0].cp_mass_phase["Vap"]) == pytest.approx(
                1e3 * r["Cp (J/g*K)"], rel=2e-2
            )
            assert value(m.fs.state[0].speed_sound_phase["Vap"]) == pytest.approx(
                r["Sound Spd. (m/s)"], rel=5e-3
            )
        # Viscosity and thermal conductivity does not match to less than 5% error.


def test_swco2_supercritical():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.swco2 = swco2.SWCO2ParameterBlock(amount_basis=swco2.AmountBasis.MASS)
    m.fs.state = m.fs.swco2.build_state_block(m.fs.time, defined_state=True)

    # Load data
    data_file = os.path.join(path, "co2_supercritical_data.csv")
    df = pandas.read_csv(data_file)

    # Flow rate does not affect any other property, so keep constant at 1
    m.fs.state[0].flow_mass.fix(1)

    for i, r in df.iterrows():
        P = r["Pressure (MPa)"]
        h = r["Enthalpy (kJ/kg)"] - H_OFFSET
        T = r["Temperature (K)"]

        print(P, T, h)

        m.fs.state[0].pressure.fix(P * units.MPa)
        m.fs.state[0].enth_mass.fix(h * units.kJ / units.kg)

        assert value(m.fs.state[0].temperature) == pytest.approx(T, rel=1e-4)
        assert value(m.fs.state[0].dens_mass) == pytest.approx(
            r["Density (kg/m3)"], rel=5e-3
        )
        assert value(m.fs.state[0].energy_internal_mass_phase["Liq"]) == pytest.approx(
            1e3 * (r["Internal Energy (kJ/kg)"] - H_OFFSET), rel=2e-3
        )
        if not ((P == 7.3773 and T == 684.13) or (P == 7.3773 and T == 704.13)):
            assert value(m.fs.state[0].entr_mass_phase["Liq"]) == pytest.approx(
                1e3 * (r["Entropy (J/g*K)"] - S_OFFSET), rel=2e-2
            )
        if not r["Cv (J/g*K)"] == "ND":
            assert value(m.fs.state[0].cv_mass) == pytest.approx(
                1e3 * float(r["Cv (J/g*K)"]), rel=2e-2
            )
            assert value(m.fs.state[0].cp_mass) == pytest.approx(
                1e3 * float(r["Cp (J/g*K)"]), rel=5e-2
            )
            assert value(m.fs.state[0].speed_sound_phase["Vap"]) == pytest.approx(
                float(r["Sound Spd. (m/s)"]), rel=1e-2
            )
        # Viscosity and thermal conductivity does not match to less than 5% error.


def test_swco2_saturated():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.swco2 = swco2.SWCO2ParameterBlock(amount_basis=swco2.AmountBasis.MASS)
    m.fs.state = m.fs.swco2.build_state_block(m.fs.time, defined_state=True)

    # Load data
    data_file = os.path.join(path, "co2_saturated_data.csv")
    df = pandas.read_csv(data_file)

    # Flow rate does not affect any other property, so keep constant at 1
    m.fs.state[0].flow_mass.fix(1)

    for i, r in df.iterrows():
        P = r["Pressure (MPa)"]
        h = r["Enthalpy (kJ/kg)"] - H_OFFSET
        T = r["Temperature (K)"]

        if r["Phase"] == "liquid":
            phase = "Liq"
        else:
            phase = "Vap"

        m.fs.state[0].pressure.fix(P * units.MPa)
        m.fs.state[0].enth_mass.fix(h * units.kJ / units.kg)

        assert value(m.fs.state[0].phase_frac[phase]) == pytest.approx(1, rel=1e-4)

        assert value(m.fs.state[0].temperature) == pytest.approx(T, rel=1e-4)
        assert value(m.fs.state[0].dens_mass_phase[phase]) == pytest.approx(
            r["Density (kg/m3)"], rel=5e-3
        )
        assert value(m.fs.state[0].energy_internal_mass_phase[phase]) == pytest.approx(
            1e3 * (r["Internal Energy (kJ/kg)"] - H_OFFSET), rel=1e-3
        )
        assert value(m.fs.state[0].entr_mass_phase[phase]) == pytest.approx(
            1e3 * (r["Entropy (J/g*K)"] - S_OFFSET), rel=1e-2
        )
        assert value(m.fs.state[0].cv_mass_phase[phase]) == pytest.approx(
            1e3 * r["Cv (J/g*K)"], rel=2e-2
        )
        assert value(m.fs.state[0].cp_mass_phase[phase]) == pytest.approx(
            1e3 * r["Cp (J/g*K)"], rel=2e-2
        )
        assert value(m.fs.state[0].speed_sound_phase[phase]) == pytest.approx(
            r["Sound Spd. (m/s)"], rel=5e-3
        )
        # Viscosity and thermal conductivity does not match to less than 5% error.
