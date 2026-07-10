from __future__ import annotations

import pytest

from evoquant.features.proposals import (
    IndicatorProposal,
    ProposalValidationError,
)


def _valid_doc() -> dict:
    return {
        "name": "volatility_adjusted_efficiency",
        "hypothesis": "Persistent moves with low path noise are more tradable "
        "after normalizing by current volatility.",
        "inputs": ["close", "atr"],
        "output_type": "OSCILLATOR_0_1",
        "lookback_parameters": ["trend_period", "vol_period"],
        "causal": True,
        "incremental_formula": "er(close, trend_period) / (1 + atr(vol_period))",
        "expected_regimes": ["trend_low_vol", "trend_high_vol"],
        "failure_modes": ["gap-dominated markets", "very sparse trading"],
        "complexity_cost": 4,
    }


def test_valid_proposal_passes():
    p = IndicatorProposal.from_dict(_valid_doc())
    assert p.name == "volatility_adjusted_efficiency"
    assert p.to_dict()["complexity_cost"] == 4


def test_missing_fields_rejected():
    doc = _valid_doc()
    del doc["failure_modes"]
    with pytest.raises(ProposalValidationError) as exc:
        IndicatorProposal.from_dict(doc)
    assert "failure_modes" in str(exc.value.details["missing_fields"])


def test_non_causal_rejected_outright():
    doc = _valid_doc()
    doc["causal"] = False
    with pytest.raises(ProposalValidationError):
        IndicatorProposal.from_dict(doc)


def test_unknown_inputs_rejected():
    doc = _valid_doc()
    doc["inputs"] = ["close", "future_close"]
    with pytest.raises(ProposalValidationError):
        IndicatorProposal.from_dict(doc)


@pytest.mark.parametrize("output_type", ["BOOL", "UNSCALED", "MAGIC"])
def test_bad_output_types_rejected(output_type):
    doc = _valid_doc()
    doc["output_type"] = output_type
    with pytest.raises(ProposalValidationError):
        IndicatorProposal.from_dict(doc)


@pytest.mark.parametrize(
    "formula",
    [
        "__import__('os').system('rm -rf /')",
        "eval(x)",
        "exec(code)",
        "import numpy",
        "lambda w: w[-1]",
        "a; b",
    ],
)
def test_forbidden_formula_constructs_rejected(formula):
    doc = _valid_doc()
    doc["incremental_formula"] = formula
    with pytest.raises(ProposalValidationError) as exc:
        IndicatorProposal.from_dict(doc)
    assert exc.value.code == "PROPOSAL_INVALID"


def test_bad_name_rejected():
    doc = _valid_doc()
    doc["name"] = "Bad Name!"
    with pytest.raises(ProposalValidationError):
        IndicatorProposal.from_dict(doc)
