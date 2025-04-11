"""Dataset template factory for MDIO v1.

This module provides a factory for creating MDIO dataset templates, both canonical
and custom. It includes a builder pattern for flexible dataset creation.
"""

from __future__ import annotations

from datetime import datetime
from datetime import timezone
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

from mdio.schema.base import BaseDataset
from mdio.schema.compressors import Blosc
from mdio.schema.compressors import ZFP
from mdio.schema.v1.dataset import Dataset
from mdio.schema.v1.dataset import DatasetMetadata
from mdio.schema.chunk_grid import RegularChunkGrid
from mdio.schema.chunk_grid import RectilinearChunkGrid


class DatasetTemplateFactory:
    """Factory for creating MDIO dataset templates.
    
    This factory provides methods to create both canonical dataset templates
    and allows for custom dataset creation with user-defined parameters.
    """
    
    def __init__(self):
        self._templates = {}
        self._register_canonical_templates()
    
    def _register_canonical_templates(self):
        """Register built-in canonical templates."""
        # Register common seismic data templates
        self._templates["seismic_poststack"] = self._create_poststack_template
        self._templates["seismic_prestack"] = self._create_prestack_template
        # Add more canonical templates as needed
    
    def create_template(self, template_name: str, **kwargs) -> Dataset:
        """Create a dataset template by name with optional customization.
        
        Args:
            template_name: Name of the template to create
            **kwargs: Customization parameters for the template
            
        Returns:
            A configured Dataset instance
            
        Raises:
            ValueError: If template_name is not registered
        """
        if template_name not in self._templates:
            raise ValueError(f"Unknown template: {template_name}")
            
        return self._templates[template_name](**kwargs)
    
    def register_template(self, name: str, template_func: Callable):
        """Register a custom template function.
        
        Args:
            name: Name of the template
            template_func: Function that returns a configured Dataset
        """
        self._templates[name] = template_func
    
    def _create_poststack_template(self, **kwargs) -> Dataset:
        """Create a post-stack seismic dataset template."""
        # Default configuration for post-stack data
        default_config = {
            "variables": [
                {
                    "name": "data",
                    "data_type": "float32",
                    "dimensions": ["inline", "crossline", "sample"],
                    "compressor": Blosc(),
                    "chunk_grid": RegularChunkGrid(chunk_shape=[64, 64, 64])
                }
            ],
            "metadata": {
                "name": kwargs.get("name", "poststack_seismic"),
                "api_version": "1.0",
                "created_on": datetime.now(timezone.utc)
            }
        }
        
        # Merge with any custom configuration
        config = {**default_config, **kwargs}
        return Dataset(**config)
    
    def _create_prestack_template(self, **kwargs) -> Dataset:
        """Create a pre-stack seismic dataset template."""
        # Default configuration for pre-stack data
        default_config = {
            "variables": [
                {
                    "name": "data",
                    "data_type": "float32",
                    "dimensions": ["inline", "crossline", "offset", "sample"],
                    "compressor": Blosc(),
                    "chunk_grid": RegularChunkGrid(chunk_shape=[32, 32, 32, 64])
                }
            ],
            "metadata": {
                "name": kwargs.get("name", "prestack_seismic"),
                "api_version": "1.0",
                "created_on": datetime.now(timezone.utc)
            }
        }
        
        # Merge with any custom configuration
        config = {**default_config, **kwargs}
        return Dataset(**config)


class DatasetBuilder:
    """Builder for creating custom MDIO datasets."""
    
    def __init__(self):
        self._variables = []
        self._metadata = {}
        
    def add_variable(self, name: str, data_type: str, dimensions: List[str],
                    compressor: Optional[Union[Blosc, ZFP]] = None,
                    chunk_grid: Optional[Union[RegularChunkGrid, RectilinearChunkGrid]] = None) -> "DatasetBuilder":
        """Add a variable to the dataset.
        
        Args:
            name: Variable name
            data_type: Data type (from ScalarType)
            dimensions: List of dimension names
            compressor: Optional compressor configuration
            chunk_grid: Optional chunk grid configuration
            
        Returns:
            self for method chaining
        """
        variable = {
            "name": name,
            "data_type": data_type,
            "dimensions": dimensions
        }
        
        if compressor:
            variable["compressor"] = compressor
        if chunk_grid:
            variable["chunk_grid"] = chunk_grid
            
        self._variables.append(variable)
        return self
        
    def set_metadata(self, **kwargs) -> "DatasetBuilder":
        """Set dataset metadata.
        
        Args:
            **kwargs: Metadata key-value pairs
            
        Returns:
            self for method chaining
        """
        self._metadata.update(kwargs)
        return self
        
    def build(self) -> Dataset:
        """Build the dataset with configured variables and metadata.
        
        Returns:
            A configured Dataset instance
        """
        return Dataset(
            variables=self._variables,
            metadata=DatasetMetadata(
                **self._metadata,
                api_version="1.0",
                created_on=datetime.now(timezone.utc)
            )
        )


def create_dataset(template_name: Optional[str] = None, **kwargs) -> Dataset:
    """Create a new MDIO dataset.
    
    This is the main entry point for creating MDIO datasets. It can either:
    1. Create a dataset from a canonical template
    2. Create a custom dataset using the builder pattern
    
    Args:
        template_name: Optional name of a canonical template to use
        **kwargs: Additional configuration parameters
        
    Returns:
        A configured Dataset instance
    """
    factory = DatasetTemplateFactory()
    
    if template_name:
        return factory.create_template(template_name, **kwargs)
    else:
        builder = DatasetBuilder()
        return builder.build()


if __name__ == "__main__":
    # Example 1: Create a post-stack dataset using the canonical template
    poststack = create_dataset(
        template_name="seismic_poststack",
        name="my_survey",
        description="A post-stack seismic dataset"
    )
    print("Post-stack dataset created:")
    print(f"Name: {poststack.metadata.name}")
    print(f"Variables: {[var.name for var in poststack.variables]}")
    print(f"Dimensions: {poststack.variables[0].dimensions}")
    print()

    # Example 2: Create a pre-stack dataset using the canonical template
    prestack = create_dataset(
        template_name="seismic_prestack",
        name="my_prestack_survey",
        description="A pre-stack seismic dataset"
    )
    print("Pre-stack dataset created:")
    print(f"Name: {prestack.metadata.name}")
    print(f"Variables: {[var.name for var in prestack.variables]}")
    print(f"Dimensions: {prestack.variables[0].dimensions}")
    print()

    # Example 3: Create a custom dataset using the builder pattern
    custom = (
        DatasetBuilder()
        .add_variable(
            name="data",
            data_type="float32",
            dimensions=["x", "y", "z"],
            compressor=Blosc(),
            chunk_grid=RegularChunkGrid(chunk_shape=[32, 32, 32])
        )
        .add_variable(
            name="quality",
            data_type="uint8",
            dimensions=["x", "y", "z"],
            compressor=ZFP()
        )
        .set_metadata(
            name="custom_survey",
            description="A custom seismic dataset with quality control",
            author="John Doe",
            date_acquired="2024-01-01"
        )
        .build()
    )
    print("Custom dataset created:")
    print(f"Name: {custom.metadata.name}")
    print(f"Variables: {[var.name for var in custom.variables]}")
    print(f"Description: {custom.metadata.description}") 