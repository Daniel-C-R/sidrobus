"""Bus models for the simulation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from sidrobus.bus import Bus
from sidrobus.bus.emissions_standard import EmissionsStandard
from sidrobus.bus.emissions_standard.euro_standards import EURO_VI
from sidrobus.bus.engine import AbstractEngine, ElectricEngine, FuelEngine
from sidrobus.unit_conversions import kwh_to_joules


@dataclass
class CapacityOption:
    """Data class representing a bus engine capacity option."""

    capacity_kwh: float


@dataclass
class AbstractEngineConfig(ABC):
    """Abstract base class for engine configuration.

    All engine configurations should inherit from this class.
    """

    efficiency: float
    capacity_options: dict[str, CapacityOption]

    @property
    @abstractmethod
    def engine_type(self) -> str:
        """Return the engine type."""
        pass

    @abstractmethod
    def create_engine(
        self, capacity_option_id: str, energy: float | None = None
    ) -> AbstractEngine:
        """Create an engine instance based on the configuration."""
        pass


@dataclass
class ElectricEngineConfig(AbstractEngineConfig):
    """Configuration for electric bus engines."""

    regenerative_braking_efficiency: float

    @property
    def engine_type(self) -> str:
        """Return the engine type."""
        return "Electric"

    def create_engine(
        self, capacity_option_id: str, energy: float | None = None
    ) -> AbstractEngine:
        """Create an electric engine instance based on the configuration."""
        if capacity_option_id not in self.capacity_options:
            error_message = (
                f"Capacity option '{capacity_option_id}' not found in "
                f"ElectricEngineConfig."
            )
            raise KeyError(error_message)
        capacity = self.capacity_options[capacity_option_id].capacity_kwh
        return ElectricEngine(
            efficiency=self.efficiency,
            capacity=capacity,
            regenerative_braking_efficiency=self.regenerative_braking_efficiency,
            mass=0.0,  # Engine mass is still not implemented
            energy=energy,
        )


@dataclass
class FuelEngineConfig(AbstractEngineConfig):
    """Configuration for fuel bus engines."""

    emissions_standard: EmissionsStandard

    @property
    def engine_type(self) -> str:
        """Return the engine type."""
        return "Fuel"

    def create_engine(
        self, capacity_option_id: str, energy: float | None = None
    ) -> AbstractEngine:
        """Create a fuel engine instance based on the configuration."""
        if capacity_option_id not in self.capacity_options:
            error_message = (
                f"Capacity option '{capacity_option_id}' not found in FuelEngineConfig."
            )
            raise KeyError(error_message)
        capacity = self.capacity_options[capacity_option_id].capacity_kwh
        return FuelEngine(
            efficiency=self.efficiency,
            capacity=capacity,
            emissions_standard=self.emissions_standard,
            mass=0.0,  # Engine mass is still not implemented
            energy=energy,
        )


@dataclass
class BusModel:
    """Data class representing a bus model."""

    name: str
    manufacturer: str
    engine_options: AbstractEngineConfig
    mass: float
    frontal_area: float
    aerodynamic_drag_coef: float
    rolling_resistance_coef: float

    @property
    def engine_type(self) -> str:
        """Return the type of engine (Electric or Fuel)."""
        return self.engine_options.engine_type

    @property
    def capacity_options(self) -> dict[str, CapacityOption]:
        """Return available capacity options for the engine."""
        return self.engine_options.capacity_options


class BusFactory:
    """Factory class for creating Bus instances from predefined models."""

    _models: ClassVar[dict[str, BusModel]] = {}

    @classmethod
    def register_model(cls, model_id: str, model: BusModel) -> None:
        """Register a new bus model.

        Args:
            model_id: Unique identifier for the model
            model: BusModel instance with all configuration
        """
        cls._models[model_id] = model

    @classmethod
    def get_available_models(cls) -> dict[str, BusModel]:
        """Return all registered bus models."""
        return cls._models

    @classmethod
    def get_model(cls, model_id: str) -> BusModel:
        """Get a specific bus model by its ID.

        Args:
            model_id: Unique identifier for the model

        Returns:
            BusModel instance if found, otherwise raises KeyError
        """
        if model_id not in cls._models:
            error_message = f"Model '{model_id}' not found."
            raise KeyError(error_message)
        return cls._models[model_id]

    @classmethod
    def create_bus(
        cls,
        model_id: str,
        capacity_option_id: str,
        energy: float | None = None,
        mass: float | None = None,
    ) -> Bus:
        """Create a Bus instance from a predefined model.

        Args:
            model_id: Unique identifier for the model
            capacity_option_id: Identifier for the capacity option to use
            energy: Optional initial energy level. If None, uses default capacity
            mass: Optional custom mass in kg. If None, uses model's default mass

        Returns:
            Bus instance configured according to the model

        Raises:
            KeyError: If model_id is not found
        """
        model = cls.get_model(model_id)
        selected_capacity = model.capacity_options.get(capacity_option_id)
        if not selected_capacity:
            error_message = (
                f"Capacity option '{capacity_option_id}' not found for model "
                f"'{model_id}'."
            )
            raise KeyError(error_message)

        engine = model.engine_options.create_engine(capacity_option_id, energy=energy)

        # Use custom mass if provided, otherwise use model's default mass
        bus_mass = mass if mass is not None else model.mass

        return Bus(
            engine,
            model.frontal_area,
            bus_mass,
            model.aerodynamic_drag_coef,
            model.rolling_resistance_coef,
            model.name,
            model.manufacturer,
        )

    @classmethod
    def get_available_capacity_options_for_model(
        cls, model_id: str
    ) -> dict[str, CapacityOption]:
        """Get available capacity options for a specific bus model.

        Args:
            model_id: Unique identifier for the model

        Returns:
            Dictionary of capacity options available for the model

        Raises:
            KeyError: If model_id is not found
        """
        model = cls.get_model(model_id)
        return model.capacity_options


# Electric models

ECITARO = BusModel(
    name="eCitaro",
    manufacturer="Mercedes-Benz",
    mass=20_000,
    frontal_area=8.67,
    aerodynamic_drag_coef=0.6,
    rolling_resistance_coef=0.01,
    engine_options=ElectricEngineConfig(
        efficiency=0.9,
        capacity_options={
            "3mod": CapacityOption(kwh_to_joules(294)),
            "4mod": CapacityOption(kwh_to_joules(392)),
            "5mod": CapacityOption(kwh_to_joules(490)),
            "6mod": CapacityOption(kwh_to_joules(588)),
        },
        regenerative_braking_efficiency=0.5,
    ),
)

ECITARO_G = BusModel(
    name="eCitaro G",
    manufacturer="Mercedes-Benz",
    mass=30_000,
    frontal_area=8.67,
    aerodynamic_drag_coef=0.6,
    rolling_resistance_coef=0.01,
    engine_options=ElectricEngineConfig(
        efficiency=0.9,
        capacity_options={
            "4mod": CapacityOption(kwh_to_joules(392)),
            "5mod": CapacityOption(kwh_to_joules(490)),
            "6mod": CapacityOption(kwh_to_joules(588)),
            "7mod": CapacityOption(kwh_to_joules(686)),
        },
        regenerative_braking_efficiency=0.5,
    ),
)

URBINO_12_ELECTRIC = BusModel(
    name="Urbino 12 Electric",
    manufacturer="Solaris",
    mass=18_745,
    frontal_area=8.415,
    aerodynamic_drag_coef=0.6,
    rolling_resistance_coef=0.01,
    engine_options=ElectricEngineConfig(
        efficiency=0.9,
        capacity_options={
            "528kwh": CapacityOption(kwh_to_joules(528)),
        },
        regenerative_braking_efficiency=0.5,
    ),
)

BusFactory.register_model("ecitaro", ECITARO)
BusFactory.register_model("ecitaro_g", ECITARO_G)
BusFactory.register_model("urbino_12_electric", URBINO_12_ELECTRIC)

# Fuel models

CITARO = BusModel(
    name="Citaro",
    manufacturer="Mercedes-Benz",
    mass=19_500,
    frontal_area=8.67,
    aerodynamic_drag_coef=0.6,
    rolling_resistance_coef=0.01,
    engine_options=FuelEngineConfig(
        efficiency=0.4,
        capacity_options={
            "diesel": CapacityOption(kwh_to_joules(300)),
            "cng": CapacityOption(kwh_to_joules(250)),
        },
        emissions_standard=EURO_VI,
    ),
)

CITARO_G = BusModel(
    name="Citaro G",
    manufacturer="Mercedes-Benz",
    mass=28_000,
    frontal_area=8.67,
    aerodynamic_drag_coef=0.6,
    rolling_resistance_coef=0.01,
    engine_options=FuelEngineConfig(
        efficiency=0.4,
        capacity_options={
            "diesel": CapacityOption(kwh_to_joules(400)),
            "cng": CapacityOption(kwh_to_joules(350)),
        },
        emissions_standard=EURO_VI,
    ),
)

BusFactory.register_model("citaro", CITARO)
BusFactory.register_model("citaro_g", CITARO_G)
