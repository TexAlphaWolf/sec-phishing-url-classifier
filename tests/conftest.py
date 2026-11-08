from __future__ import annotations

import pytest

from urlclf import data, synthetic
from urlclf.model import Model


@pytest.fixture(scope="session")
def samples():
    return synthetic.generate(3000, seed=7)


@pytest.fixture(scope="session")
def split(samples):
    return data.split(samples, seed=3)


@pytest.fixture(scope="session")
def trained(split):
    from urlclf.train import train

    model, report = train(split, seed=3, compare=False)
    return model, report


@pytest.fixture(scope="session")
def model(trained) -> Model:
    return trained[0]
