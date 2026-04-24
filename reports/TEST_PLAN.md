# Test Plan

| Test | Purpose | Command | Expected Result | Status |
|------|---------|---------|----------------|--------|
| DSL executor basics | Verify Program execution | `pytest tests/test_templatebank.py::TestExecutor -v` | 5/5 pass | PASS |
| Composition executor | Old implicit binding works for compatible cases | `pytest tests/test_templatebank.py::TestCompositionExecutor -v` | 4/4 pass | PASS |
| Library save/load | Serialization round-trip | `pytest tests/test_templatebank.py::TestLibrarySaveLoad -v` | 3/3 pass | PASS |
| MCD split | Split covers all, no overlap | `pytest tests/test_templatebank.py::TestMCDSplit -v` | 3/3 pass | PASS |
| Inline program | Plan-to-flat conversion | `pytest tests/test_templatebank.py::TestInlineProgram -v` | 5/5 pass | PASS |
| Multi-call faithfulness | Chained calls execute correctly | `pytest tests/test_templatebank.py::TestMultiCallPlanFaithfulness -v` | 3/3 pass | PASS |
| MCD compounds | Rich compound types | `pytest tests/test_templatebank.py::TestMCDRichCompounds -v` | 3/3 pass | PASS |
| **GIFT import** | New modules importable | `pytest tests/test_dataflow_plan.py::TestDataflowPlanImport -v` | 4/4 pass | PASS |
| **GIFT explicit dataflow** | ADD→MUL with call_output binding | `pytest tests/test_dataflow_plan.py::TestExplicitDataflow -v` | 4/4 pass | PASS |
| **GIFT rejects empty** | Empty/missing bindings rejected | (included above) | pass | PASS |
| **GIFT serialization** | JSON round-trip | `pytest tests/test_dataflow_plan.py::TestDataflowSerialization -v` | 1/1 pass | PASS |
| **GIFT active binding** | Perturbation changes output | `pytest tests/test_dataflow_plan.py::TestActiveBidingPerturbation -v` | 1/1 pass | PASS |
| **All tests** | Full suite | `pytest tests/ -v` | 41/41 pass | PASS |
