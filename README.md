# MakarSpace: AI-Driven Anomaly Detection for Spacecraft

## Overview
MakarSpace is an open-source AI-driven anomaly detection system for spacecraft, enabling predictive maintenance, explainable diagnostics, and real-time monitoring of critical systems.

## Key Features
- **Real-Time Anomaly Detection**: Sensor fusion, dynamic thresholds, and edge-optimized inference
- **Predictive Maintenance**: Component failure prediction and remaining useful life estimation
- **Explainability Engine**: Transparent AI reasoning with SHAP/LIME visualizations
- **Modular Architecture**: Plugin system for custom sensors and ROS 2 compatibility
- **Simulation Tools**: Synthetic data generation and digital twin integration

## Getting Started

### Installation
```bash
pip install -e .
```

### Quick Start
```python
from makarspace.simulator import RadiationAnomalyGenerator
from makarspace.anomaly_detection import AnomalyDetector

# Generate synthetic data
generator = RadiationAnomalyGenerator(
    base_temperature=20,  # °C
    radiation_spike_threshold=500  # rads
)
data = generator.generate(duration="30d")

# Train anomaly detector
detector = AnomalyDetector()
detector.train(data)

# Run predictions
anomalies = detector.detect(new_data)
```

### Dashboard
To launch the interactive dashboard:
```bash
streamlit run makarspace/dashboard/app.py
```

## Project Structure
```
/makarspace
  ├── /anomaly_detection  # Core MVP  
  │   ├── /models         # Pre-trained LSTM/Transformer  
  │   ├── /simulator      # Synthetic data tools  
  │   ├── /visualization  # Explainability visualization
  │   └── /api            # REST API
  ├── /dashboard          # Streamlit mission control dashboard
  ├── /tests              # Unit and integration tests
  ├── LICENSE             # Apache 2.0  
  └── CONTRIBUTING.md     # Guidelines for PRs  
```

## Roadmap
- Phase 1: Research & Data Strategy
- Phase 2: MVP Development
- Phase 3: Open-Source Launch

## Contributing
We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
