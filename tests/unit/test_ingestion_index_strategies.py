"""Tests for index strategies."""

import numpy as np
from mdio.ingestion.index_strategies import IndexStrategyRegistry
from mdio.segy.geometry import GridOverrides


def test_index_strategy_registry_default():
    registry = IndexStrategyRegistry()
    strategy = registry.create_strategy(None)
    assert strategy.name == "RegularGridStrategy"


def test_index_strategy_registry_non_binned():
    registry = IndexStrategyRegistry()
    overrides = GridOverrides(non_binned=True, chunksize=64)
    strategy = registry.create_strategy(overrides)
    assert strategy.name == "NonBinnedStrategy"


def test_index_strategy_registry_composite():
    registry = IndexStrategyRegistry()
    overrides = GridOverrides(auto_channel_wrap=True, non_binned=True, chunksize=64)
    strategy = registry.create_strategy(overrides)
    assert strategy.name == "CompositeStrategy"
    assert len(strategy.strategies) == 2
    assert strategy.strategies[0].name == "ChannelWrappingStrategy"
    assert strategy.strategies[1].name == "NonBinnedStrategy"
