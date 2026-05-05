import pytest
from pydantic import ValidationError
from mdio.segy.geometry import GridOverrides

def test_grid_overrides_defaults():
    overrides = GridOverrides()
    assert not overrides.auto_channel_wrap
    assert not overrides.auto_shot_wrap
    assert not overrides.non_binned
    assert not overrides.has_duplicates
    assert overrides.chunksize is None
    assert overrides.replace_dims is None
    assert overrides.extra_params == {}
    assert not bool(overrides)

def test_grid_overrides_aliases():
    overrides = GridOverrides(AutoChannelWrap=True, chunksize=64)
    assert overrides.auto_channel_wrap is True
    assert overrides.chunksize == 64
    
    # Test bool
    assert bool(overrides) is True

def test_grid_overrides_validation():
    with pytest.raises(ValidationError):
        GridOverrides(chunksize=0)
    
    with pytest.raises(ValidationError):
        GridOverrides(chunksize=-1)

def test_grid_overrides_serialization():
    overrides = GridOverrides(AutoChannelWrap=True, chunksize=64)
    dumped = overrides.model_dump(by_alias=True, exclude_defaults=True)
    assert dumped == {"AutoChannelWrap": True, "chunksize": 64}
    
    dumped_modern = overrides.model_dump(exclude_defaults=True)
    assert dumped_modern == {"auto_channel_wrap": True, "chunksize": 64}

def test_grid_overrides_extra_params():
    overrides = GridOverrides(extra_params={"custom_key": "custom_value"})
    assert overrides.extra_params == {"custom_key": "custom_value"}
