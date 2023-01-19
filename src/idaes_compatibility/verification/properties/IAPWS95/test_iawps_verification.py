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
import idaes.models.properties.iapws95 as iapws95

path = os.path.dirname(os.path.abspath(__file__))


def test_iapws95_liquid_phase():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.iapws = iapws95.Iapws95ParameterBlock(amount_basis=iapws95.AmountBasis.MASS)
    m.fs.state = m.fs.iapws.build_state_block(m.fs.time, defined_state=True)

    # Load data
    data_file = os.path.join(path, "h2o_liquid_data.csv")
    df = pandas.read_csv(data_file)

    # Flow rate does not affect any other property, so keep constant at 1
    m.fs.state[0].flow_mass.fix(1)

    for i, r in df.iterrows():
        P = r["Pressure (MPa)"]
        h = r["Enthalpy (kJ/kg)"]
        T = r["Temperature (K)"]

        if not (P == 22 and T > 646.85):  # Skip near the critical point
            m.fs.state[0].pressure.fix(P * units.MPa)
            m.fs.state[0].enth_mass.fix(h * units.kJ / units.kg)

            assert value(m.fs.state[0].phase_frac["Liq"]) == pytest.approx(1, rel=5e-4)

            assert value(m.fs.state[0].temperature) == pytest.approx(T, rel=1e-4)
            assert value(m.fs.state[0].dens_mass_phase["Liq"]) == pytest.approx(
                r["Density (kg/m3)"], rel=1e-3
            )
            assert value(
                m.fs.state[0].energy_internal_mass_phase["Liq"]
            ) == pytest.approx(1e3 * r["Internal Energy (kJ/kg)"], rel=1e-3)
            assert value(m.fs.state[0].entr_mass_phase["Liq"]) == pytest.approx(
                1e3 * r["Entropy (J/g*K)"], rel=1e-2
            )
            assert value(m.fs.state[0].cv_mass_phase["Liq"]) == pytest.approx(
                1e3 * r["Cv (J/g*K)"], rel=2e-2
            )
            assert value(m.fs.state[0].cp_mass_phase["Liq"]) == pytest.approx(
                1e3 * r["Cp (J/g*K)"], rel=2e-2
            )
            assert value(m.fs.state[0].speed_sound_phase["Liq"]) == pytest.approx(
                r["Sound Spd. (m/s)"], rel=1e-3
            )
            assert value(m.fs.state[0].visc_d_phase["Liq"]) == pytest.approx(
                r["Viscosity (Pa*s)"], rel=2e-2
            )
            # Thermal conductivity does not match to less than 5% error.


def test_iapws95_vapor_phase():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.iapws = iapws95.Iapws95ParameterBlock(amount_basis=iapws95.AmountBasis.MASS)
    m.fs.state = m.fs.iapws.build_state_block(m.fs.time, defined_state=True)

    # Load data
    data_file = os.path.join(path, "h2o_vapor_data.csv")
    df = pandas.read_csv(data_file)

    # Flow rate does not affect any other property, so keep constant at 1
    m.fs.state[0].flow_mass.fix(1)

    for i, r in df.iterrows():
        P = r["Pressure (MPa)"]
        h = r["Enthalpy (kJ/kg)"]
        T = r["Temperature (K)"]

        if not (P == 22 and T < 647):  # Skip near the critical point
            m.fs.state[0].pressure.fix(P * units.MPa)
            m.fs.state[0].enth_mass.fix(h * units.kJ / units.kg)

            assert value(m.fs.state[0].phase_frac["Vap"]) == pytest.approx(1, rel=5e-4)

            assert value(m.fs.state[0].temperature) == pytest.approx(T, rel=1e-4)
            assert value(m.fs.state[0].dens_mass_phase["Vap"]) == pytest.approx(
                r["Density (kg/m3)"], rel=5e-3
            )
            assert value(
                m.fs.state[0].energy_internal_mass_phase["Vap"]
            ) == pytest.approx(1e3 * r["Internal Energy (kJ/kg)"], rel=1e-3)
            assert value(m.fs.state[0].entr_mass_phase["Vap"]) == pytest.approx(
                1e3 * r["Entropy (J/g*K)"], rel=1e-2
            )
            assert value(m.fs.state[0].cv_mass_phase["Vap"]) == pytest.approx(
                1e3 * r["Cv (J/g*K)"], rel=2e-2
            )
            assert value(m.fs.state[0].cp_mass_phase["Vap"]) == pytest.approx(
                1e3 * r["Cp (J/g*K)"], rel=2e-2
            )
            assert value(m.fs.state[0].speed_sound_phase["Vap"]) == pytest.approx(
                r["Sound Spd. (m/s)"], rel=1e-3
            )
            assert value(m.fs.state[0].visc_d_phase["Vap"]) == pytest.approx(
                r["Viscosity (Pa*s)"], rel=3e-2
            )
            # Thermal conductivity does not match to less than 5% error.


def test_iapws95_supercritical():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.iapws = iapws95.Iapws95ParameterBlock(amount_basis=iapws95.AmountBasis.MASS)
    m.fs.state = m.fs.iapws.build_state_block(m.fs.time, defined_state=True)

    # Load data
    data_file = os.path.join(path, "h2o_supercritical_data.csv")
    df = pandas.read_csv(data_file)

    # Flow rate does not affect any other property, so keep constant at 1
    m.fs.state[0].flow_mass.fix(1)

    for i, r in df.iterrows():
        P = r["Pressure (MPa)"]
        h = r["Enthalpy (kJ/kg)"]
        T = r["Temperature (K)"]

        m.fs.state[0].pressure.fix(P * units.MPa)
        m.fs.state[0].enth_mass.fix(h * units.kJ / units.kg)

        assert value(m.fs.state[0].temperature) == pytest.approx(T, rel=1e-4)
        assert value(m.fs.state[0].dens_mass) == pytest.approx(
            r["Density (kg/m3)"], rel=5e-3
        )
        assert value(m.fs.state[0].energy_internal_mass) == pytest.approx(
            1e3 * r["Internal Energy (kJ/kg)"], rel=1e-3
        )
        assert value(m.fs.state[0].entr_mass) == pytest.approx(
            1e3 * r["Entropy (J/g*K)"], rel=1e-2
        )
        assert value(m.fs.state[0].cv_mass) == pytest.approx(
            1e3 * r["Cv (J/g*K)"], rel=2e-2
        )
        assert value(m.fs.state[0].cp_mass) == pytest.approx(
            1e3 * r["Cp (J/g*K)"], rel=5e-2
        )
        assert value(m.fs.state[0].speed_sound_phase["Vap"]) == pytest.approx(
            r["Sound Spd. (m/s)"], rel=1e-2
        )
        assert value(m.fs.state[0].visc_d_phase["Vap"]) == pytest.approx(
            r["Viscosity (Pa*s)"], rel=5e-2
        )
        # Thermal conductivity does not match to less than 5% error.


def test_iapws95_saturated():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.iapws = iapws95.Iapws95ParameterBlock(amount_basis=iapws95.AmountBasis.MASS)
    m.fs.state = m.fs.iapws.build_state_block(m.fs.time, defined_state=True)

    # Load data
    data_file = os.path.join(path, "h2o_saturated_data.csv")
    df = pandas.read_csv(data_file)

    # Flow rate does not affect any other property, so keep constant at 1
    m.fs.state[0].flow_mass.fix(1)

    for i, r in df.iterrows():
        P = r["Pressure (MPa)"]
        h = r["Enthalpy (kJ/kg)"]
        T = r["Temperature (K)"]

        if r["Phase"] == "liquid":
            phase = "Liq"
        else:
            phase = "Vap"

        if not (
            i == 0 or (P == 21.899 and T == 646.47)
        ):  # Skip first point & near critical point
            m.fs.state[0].pressure.fix(P * units.MPa)
            m.fs.state[0].enth_mass.fix(h * units.kJ / units.kg)

            assert value(m.fs.state[0].phase_frac[phase]) == pytest.approx(1, rel=5e-4)

            assert value(m.fs.state[0].temperature) == pytest.approx(T, rel=1e-4)
            assert value(m.fs.state[0].dens_mass_phase[phase]) == pytest.approx(
                r["Density (kg/m3)"], rel=5e-3
            )
            assert value(
                m.fs.state[0].energy_internal_mass_phase[phase]
            ) == pytest.approx(1e3 * r["Internal Energy (kJ/kg)"], rel=1e-3)
            assert value(m.fs.state[0].entr_mass_phase[phase]) == pytest.approx(
                1e3 * r["Entropy (J/g*K)"], rel=1e-2
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
            assert value(m.fs.state[0].visc_d_phase[phase]) == pytest.approx(
                r["Viscosity (Pa*s)"], rel=3e-2
            )
            # Thermal conductivity does not match to less than 5% error.
