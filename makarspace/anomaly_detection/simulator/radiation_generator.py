import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

class RadiationAnomalyGenerator:
    """
    Generator for synthetic spacecraft telemetry data with radiation anomalies.
    
    This class simulates telemetry data including temperature, radiation levels,
    voltage, and other sensor readings typically found in spacecraft. It can
    inject realistic anomalies based on physics models of radiation effects.
    """
    
    def __init__(
        self,
        base_temperature: float = 20.0,  # °C
        base_radiation: float = 10.0,    # rads
        base_voltage: float = 28.0,      # V
        base_current: float = 5.0,       # A
        base_pressure: float = 1.0,      # kPa
        radiation_spike_threshold: float = 500.0,  # rads
        random_seed: Optional[int] = None
    ):
        """
        Initialize the radiation anomaly generator.
        
        Args:
            base_temperature: Baseline temperature in °C
            base_radiation: Baseline radiation level in rads
            base_voltage: Baseline voltage in V
            base_current: Baseline current in A
            base_pressure: Baseline pressure in kPa
            radiation_spike_threshold: Threshold for radiation spike anomalies in rads
            random_seed: Seed for random number generation
        """
        self.base_temperature = base_temperature
        self.base_radiation = base_radiation
        self.base_voltage = base_voltage
        self.base_current = base_current
        self.base_pressure = base_pressure
        self.radiation_spike_threshold = radiation_spike_threshold
        
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # Define mission phases and their characteristics
        self.mission_phases = {
            "launch": {
                "temp_range": (30, 50),
                "rad_range": (5, 50),
                "duration": timedelta(minutes=10)
            },
            "orbit_insertion": {
                "temp_range": (20, 40),
                "rad_range": (10, 100),
                "duration": timedelta(hours=2)
            },
            "nominal_orbit": {
                "temp_range": (15, 25),
                "rad_range": (5, 20),
                "duration": timedelta(days=365)
            },
            "solar_flare": {
                "temp_range": (25, 60),
                "rad_range": (100, 800),
                "duration": timedelta(hours=6)
            }
        }
    
    def _parse_duration(self, duration: Union[str, int, timedelta]) -> timedelta:
        """Parse a duration string into a timedelta object."""
        if isinstance(duration, timedelta):
            return duration
        
        if isinstance(duration, int):
            # Assume days if just a number
            return timedelta(days=duration)
        
        # Parse string format like "30d", "12h", "45m", "30s"
        unit = duration[-1].lower()
        value = int(duration[:-1])
        
        if unit == 'd':
            return timedelta(days=value)
        elif unit == 'h':
            return timedelta(hours=value)
        elif unit == 'm':
            return timedelta(minutes=value)
        elif unit == 's':
            return timedelta(seconds=value)
        else:
            raise ValueError(f"Unknown duration unit: {unit}. Use 'd', 'h', 'm', or 's'.")
    
    def _generate_sine_wave(
        self, 
        start_val: float, 
        amplitude: float, 
        frequency: float, 
        timestamps: List[datetime],
        noise_level: float = 0.1
    ) -> np.ndarray:
        """Generate a sine wave with additive noise."""
        n_samples = len(timestamps)
        t = np.linspace(0, 2 * np.pi * frequency * n_samples / 86400, n_samples)
        wave = start_val + amplitude * np.sin(t)
        noise = np.random.normal(0, noise_level * amplitude, n_samples)
        return wave + noise
    
    def _inject_anomalies(
        self, 
        data: pd.DataFrame, 
        anomaly_rate: float = 0.05
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Inject anomalies into the telemetry data.
        
        Args:
            data: DataFrame with telemetry data
            anomaly_rate: Percentage of data points to mark as anomalies
            
        Returns:
            Tuple of (data with anomalies, anomaly labels)
        """
        n_samples = len(data)
        anomaly_indices = np.random.choice(
            n_samples,
            size=int(n_samples * anomaly_rate),
            replace=False
        )
        
        # Create anomaly labels DataFrame
        anomalies = pd.DataFrame(
            False,
            index=data.index,
            columns=["is_anomaly", "radiation_spike", "temperature_surge", "voltage_drop"]
        )
        
        # Radiation spikes (most common anomaly)
        rad_spike_indices = anomaly_indices[:int(len(anomaly_indices) * 0.5)]
        for idx in rad_spike_indices:
            spike_magnitude = np.random.uniform(1.5, 4.0)
            data.loc[data.index[idx], "radiation"] *= spike_magnitude
            
            # If radiation exceeds threshold, mark as anomaly
            if data.loc[data.index[idx], "radiation"] > self.radiation_spike_threshold:
                anomalies.loc[data.index[idx], ["is_anomaly", "radiation_spike"]] = True
                
                # Radiation affects temperature with some delay
                if idx + 3 < n_samples:
                    data.loc[data.index[idx+3], "temperature"] += np.random.uniform(5, 15)
                    anomalies.loc[data.index[idx+3], ["is_anomaly", "temperature_surge"]] = True
        
        # Voltage drops (less common)
        voltage_drop_indices = anomaly_indices[int(len(anomaly_indices) * 0.5):]
        for idx in voltage_drop_indices:
            drop_magnitude = np.random.uniform(0.3, 0.8)
            data.loc[data.index[idx], "voltage"] *= drop_magnitude
            anomalies.loc[data.index[idx], ["is_anomaly", "voltage_drop"]] = True
        
        return data, anomalies
    
    def generate(
        self, 
        duration: Union[str, int, timedelta] = "30d",
        sampling_interval: Union[str, int, timedelta] = "5m",
        include_anomalies: bool = True,
        anomaly_rate: float = 0.05,
        include_mission_phases: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate synthetic telemetry data.
        
        Args:
            duration: Total duration to generate data for (e.g., "30d", "12h")
            sampling_interval: Interval between data points (e.g., "5m", "1h")
            include_anomalies: Whether to inject anomalies
            anomaly_rate: Percentage of data points to mark as anomalies
            include_mission_phases: Whether to include different mission phases
            
        Returns:
            Dictionary with telemetry data and anomaly labels
        """
        # Parse duration and sampling interval
        duration_td = self._parse_duration(duration)
        sampling_interval_td = self._parse_duration(sampling_interval)
        
        # Generate timestamps
        start_time = datetime.now() - duration_td
        end_time = datetime.now()
        timestamps = []
        
        current_time = start_time
        while current_time <= end_time:
            timestamps.append(current_time)
            current_time += sampling_interval_td
        
        # Create DataFrame with timestamps
        data = pd.DataFrame(index=timestamps)
        data.index.name = "timestamp"
        
        # Generate baseline data
        n_samples = len(timestamps)
        
        # Mission phases if enabled
        if include_mission_phases:
            # Start with launch
            current_phase = "launch"
            phase_end = start_time + self.mission_phases[current_phase]["duration"]
            
            # Generate data for each phase
            phase_data = []
            phase_labels = []
            
            for i, ts in enumerate(timestamps):
                if ts > phase_end:
                    # Transition to next phase
                    if current_phase == "launch":
                        current_phase = "orbit_insertion"
                    elif current_phase == "orbit_insertion":
                        current_phase = "nominal_orbit"
                    elif current_phase == "nominal_orbit" and np.random.random() < 0.02:
                        # Random chance to encounter solar flare
                        current_phase = "solar_flare"
                    elif current_phase == "solar_flare":
                        current_phase = "nominal_orbit"
                        
                    phase_end = ts + self.mission_phases[current_phase]["duration"]
                
                # Get phase characteristics
                temp_range = self.mission_phases[current_phase]["temp_range"]
                rad_range = self.mission_phases[current_phase]["rad_range"]
                
                # Generate data for this timestamp
                phase_data.append({
                    "temperature": np.random.uniform(*temp_range),
                    "radiation": np.random.uniform(*rad_range),
                    "voltage": self.base_voltage + np.random.normal(0, 0.5),
                    "current": self.base_current + np.random.normal(0, 0.2),
                    "pressure": self.base_pressure + np.random.normal(0, 0.1)
                })
                
                phase_labels.append(current_phase)
            
            # Create DataFrame
            data = pd.DataFrame(phase_data, index=timestamps)
            data["mission_phase"] = phase_labels
            
        else:
            # Generate simple sinusoidal patterns with noise
            data["temperature"] = self._generate_sine_wave(
                self.base_temperature, 5.0, 1.0, timestamps, 0.1
            )
            data["radiation"] = self._generate_sine_wave(
                self.base_radiation, 2.0, 2.0, timestamps, 0.3
            )
            data["voltage"] = self._generate_sine_wave(
                self.base_voltage, 1.0, 0.5, timestamps, 0.05
            )
            data["current"] = self._generate_sine_wave(
                self.base_current, 0.5, 0.7, timestamps, 0.1
            )
            data["pressure"] = self._generate_sine_wave(
                self.base_pressure, 0.2, 0.3, timestamps, 0.05
            )
        
        # Add some correlations (temperature affects pressure, etc.)
        data["pressure"] += 0.05 * (data["temperature"] - self.base_temperature)
        data["voltage"] -= 0.02 * (data["temperature"] - self.base_temperature)
        
        # Inject anomalies if requested
        if include_anomalies:
            data, anomalies = self._inject_anomalies(data, anomaly_rate)
        else:
            # Create empty anomaly labels
            anomalies = pd.DataFrame(
                False,
                index=data.index,
                columns=["is_anomaly", "radiation_spike", "temperature_surge", "voltage_drop"]
            )
        
        return {
            "telemetry": data,
            "anomalies": anomalies
        }
