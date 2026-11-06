import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="also run slow checks")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="needs --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
