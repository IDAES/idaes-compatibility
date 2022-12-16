# IDAES Backward Compatibility Repository

This repository contains backward compatibility tests for IDAES tools. Tests are organized into three categories:

1. API Tests
2. Verification Tests
3. Robustness Tests

## API Tests

API tests are intended to ensure that the core IDAES APIs and workflows retain backward compatibility, and are based on the IDAES Tutorials and Examples. Tests are organized by the IDAES release version in which they were introduced, and as examples evolve multiple versions of the same example may be added as tests to ensure backward compatibility is maintained across different versions. Tests will only be removed from here when a major feature is fully deprecated and a given example no longer works, which should only correspond to sufficient warning period and a major (or significant minor) release.

## Verification Tests

Verification tests are intended to ensure that models within the IDAES toolset provide the same, correct result across multiple versions. All core models should be tested against literature data across the widest range of data possible. As these tests are focused on ensuring the correct solution, to keep run times to a minimum these tests should avoid calling a solver where possible and should instead load data from a file and confirm that all constraints are satisfied.

## Robustness Tests

Robustness tests are intended to ensure that models within the IDAES toolset can reliably converge across a wide range of conditions. These are in many ways counterparts to the verification tests; verification tests are concerned about the final answer, whereas these tests focus on being able to find a solution. As these tests inherently require calling a solver, it is not feasible to test all possible combinations of models and conditions, so test cases should be chosen judiciously from a subset of the solutions available from data used in the verification tests. When writing robustness tests for unit models, developers should try to ensure that we use a wide range of property packages so that all property packages get tested in unit model applications.
