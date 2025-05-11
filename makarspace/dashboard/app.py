import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import os
import sys
import uuid
import time

# Add parent directory to path to make imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from makarspace.anomaly_detection.detector import AnomalyDetector
from makarspace.anomaly_detection.simulator.radiation_generator import RadiationAnomalyGenerator

# Set page configuration
st.set_page_config(
    page_title="MakarSpace Anomaly Detection",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper functions
def load_or_simulate_data():
    """Load existing data or simulate new data."""
    if 'telemetry_data' not in st.session_state or st.session_state.regenerate_data:
        with st.spinner("Generating synthetic telemetry data..."):
            # Create generator
            generator = RadiationAnomalyGenerator(
                base_temperature=st.session_state.get('base_temperature', 20.0),
                base_radiation=st.session_state.get('base_radiation', 10.0),
                radiation_spike_threshold=st.session_state.get('radiation_threshold', 500.0),
                random_seed=st.session_state.get('random_seed', 42)
            )
            
            # Generate data
            duration = st.session_state.get('duration', '1d')
            sampling_interval = st.session_state.get('sampling_interval', '1m')
            include_anomalies = st.session_state.get('include_anomalies', True)
            
            data = generator.generate(
                duration=duration,
                sampling_interval=sampling_interval,
                include_anomalies=include_anomalies,
                anomaly_rate=st.session_state.get('anomaly_rate', 0.05),
                include_mission_phases=st.session_state.get('include_mission_phases', True)
            )
            
            st.session_state.telemetry_data = data['telemetry']
            st.session_state.anomaly_labels = data['anomalies']
            st.session_state.regenerate_data = False
            
            # Set model features
            if 'model_features' not in st.session_state:
                st.session_state.model_features = data['telemetry'].select_dtypes(include=[np.number]).columns.tolist()
                # Filter out any columns that are not telemetry
                st.session_state.model_features = [f for f in st.session_state.model_features 
                                                  if f not in ['is_anomaly', 'anomaly_score']]
    
    return st.session_state.telemetry_data, st.session_state.anomaly_labels

def train_model():
    """Train the anomaly detection model."""
    if 'detector' not in st.session_state or st.session_state.retrain_model:
        with st.spinner("Training anomaly detection model..."):
            # Create detector
            detector = AnomalyDetector(
                model_type=st.session_state.model_type,
                input_features=st.session_state.model_features,
                sequence_length=st.session_state.sequence_length,
                threshold=None  # Auto-determine threshold
            )
            
            # Get data
            telemetry_data, anomaly_labels = load_or_simulate_data()
            
            # Train model
            data_dict = {
                'telemetry': telemetry_data,
                'anomalies': anomaly_labels
            }
            
            results = detector.train(data_dict)
            
            st.session_state.detector = detector
            st.session_state.training_results = results
            st.session_state.retrain_model = False
            
            # Save model
            os.makedirs("models", exist_ok=True)
            model_path = f"models/{uuid.uuid4()}"
            detector.save(model_path)
            st.session_state.model_path = model_path
    
    return st.session_state.detector

def detect_anomalies():
    """Detect anomalies in the telemetry data."""
    if 'predictions' not in st.session_state or st.session_state.redetect_anomalies:
        with st.spinner("Detecting anomalies..."):
            # Get detector and data
            detector = train_model()
            telemetry_data, _ = load_or_simulate_data()
            
            # Detect anomalies
            results = detector.detect(telemetry_data, explain=True)
            
            st.session_state.predictions = results['predictions']
            st.session_state.anomaly_timestamps = results['anomaly_timestamps']
            st.session_state.explanations = results.get('explanations', {})
            st.session_state.redetect_anomalies = False
    
    return st.session_state.predictions, st.session_state.anomaly_timestamps, st.session_state.explanations

def plot_telemetry_with_anomalies(telemetry_data, predictions, features_to_plot=None):
    """Create interactive plot of telemetry data with anomalies."""
    if features_to_plot is None:
        features_to_plot = st.session_state.model_features[:3]  # Default to first 3 features
    
    # Create figure
    fig = go.Figure()
    
    # Add trace for each feature
    for feature in features_to_plot:
        fig.add_trace(go.Scatter(
            x=telemetry_data.index,
            y=telemetry_data[feature],
            mode='lines',
            name=feature
        ))
        
        # Add markers for anomalies if predictions exist
        if predictions is not None and not predictions.empty:
            anomaly_data = predictions[predictions['is_anomaly']]
            if not anomaly_data.empty and feature in anomaly_data:
                fig.add_trace(go.Scatter(
                    x=anomaly_data.index,
                    y=anomaly_data[feature],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='red',
                        symbol='x'
                    ),
                    name=f'{feature} anomalies'
                ))
    
    # Update layout
    fig.update_layout(
        title='Spacecraft Telemetry with Detected Anomalies',
        xaxis_title='Time',
        yaxis_title='Value',
        legend_title='Features',
        height=500,
        hovermode='closest'
    )
    
    return fig

def initialize_session_state():
    """Initialize session state variables."""
    defaults = {
        'regenerate_data': False,
        'retrain_model': False,
        'redetect_anomalies': False,
        'model_type': 'lstm',
        'sequence_length': 10,
        'base_temperature': 20.0,
        'base_radiation': 10.0,
        'radiation_threshold': 500.0,
        'random_seed': 42,
        'duration': '1d',
        'sampling_interval': '1m',
        'include_anomalies': True,
        'anomaly_rate': 0.05,
        'include_mission_phases': True,
        'selected_features': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Initialize session state
initialize_session_state()

# Define sidebar
with st.sidebar:
    st.title("MakarSpace")
    st.image("https://img.icons8.com/fluency/96/000000/satellite.png")
    st.subheader("Spacecraft Anomaly Detection")
    
    # Navigation
    page = st.radio("Navigation", ["Dashboard", "Simulation Settings", "Model Settings", "Anomaly Analysis"])
    
    st.markdown("---")
    
    # Common actions
    if st.button("Generate New Data"):
        st.session_state.regenerate_data = True
        st.session_state.retrain_model = True
        st.session_state.redetect_anomalies = True
    
    if st.button("Retrain Model"):
        st.session_state.retrain_model = True
        st.session_state.redetect_anomalies = True
    
    if st.button("Re-detect Anomalies"):
        st.session_state.redetect_anomalies = True

# Main content based on selected page
if page == "Dashboard":
    st.title("Mission Control Dashboard")
    
    # Load data and train model if needed
    telemetry_data, _ = load_or_simulate_data()
    predictions, anomaly_timestamps, explanations = detect_anomalies()
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Data Points", 
            value=len(telemetry_data),
            delta=None
        )
    
    with col2:
        st.metric(
            label="Anomalies Detected", 
            value=len(anomaly_timestamps),
            delta=f"{len(anomaly_timestamps)/len(telemetry_data)*100:.2f}%"
        )
    
    with col3:
        if 'training_results' in st.session_state and 'f1_score' in st.session_state.training_results:
            f1 = st.session_state.training_results['f1_score']
            st.metric(
                label="Model F1 Score", 
                value=f"{f1:.3f}",
                delta=None
            )
        else:
            st.metric(
                label="Model F1 Score", 
                value="N/A",
                delta=None
            )
    
    with col4:
        if 'detector' in st.session_state:
            st.metric(
                label="Model Type", 
                value=st.session_state.detector.model_type.upper(),
                delta=None
            )
        else:
            st.metric(
                label="Model Type", 
                value="N/A",
                delta=None
            )
    
    # Feature selection for plot
    available_features = st.session_state.model_features
    selected_features = st.multiselect(
        "Select features to display", 
        available_features,
        default=available_features[:3] if available_features else []
    )
    
    if not selected_features:
        selected_features = available_features[:3] if available_features else []
    
    # Plot telemetry data with anomalies
    fig = plot_telemetry_with_anomalies(telemetry_data, predictions, selected_features)
    st.plotly_chart(fig, use_container_width=True)
    
    # Anomaly table
    if len(anomaly_timestamps) > 0:
        st.subheader("Detected Anomalies")
        
        anomaly_df = pd.DataFrame({
            'Timestamp': [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in anomaly_timestamps],
            'Anomaly Score': [predictions.loc[ts, 'anomaly_score'] for ts in anomaly_timestamps]
        })
        
        # Add primary cause if explanations exist
        if explanations:
            causes = []
            for ts in anomaly_timestamps:
                if ts in explanations:
                    top_feature = explanations[ts]['top_features'][0][0] if explanations[ts]['top_features'] else "Unknown"
                    causes.append(top_feature)
                else:
                    causes.append("Unknown")
            
            anomaly_df['Primary Cause'] = causes
        
        st.dataframe(anomaly_df)
        
        # Detailed explanation for selected anomaly
        if anomaly_timestamps:
            selected_anomaly = st.selectbox(
                "Select anomaly for detailed explanation", 
                [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in anomaly_timestamps]
            )
            
            # Convert back to datetime for lookup
            selected_ts = None
            for ts in anomaly_timestamps:
                if ts.strftime('%Y-%m-%d %H:%M:%S') == selected_anomaly:
                    selected_ts = ts
                    break
            
            if selected_ts and selected_ts in explanations:
                exp = explanations[selected_ts]
                
                st.subheader("Anomaly Explanation")
                st.markdown(f"**Explanation:** {exp['explanation_text']}")
                
                # Feature contributions
                st.subheader("Feature Contributions")
                
                contrib_df = pd.DataFrame({
                    'Feature': [f[0] for f in exp['top_features']],
                    'Contribution': [f[1] for f in exp['top_features']]
                })
                
                contrib_fig = px.bar(
                    contrib_df,
                    x='Contribution',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance',
                    labels={'Contribution': 'Importance Score', 'Feature': 'Feature'}
                )
                
                contrib_fig.update_layout(height=300)
                st.plotly_chart(contrib_fig, use_container_width=True)
    else:
        st.info("No anomalies detected in the current data.")

elif page == "Simulation Settings":
    st.title("Simulation Settings")
    st.markdown("Configure the synthetic data generation parameters.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Temporal Settings")
        
        duration = st.text_input(
            "Simulation Duration", 
            value=st.session_state.duration,
            help="Format: '30d', '12h', '45m', etc."
        )
        
        sampling_interval = st.text_input(
            "Sampling Interval", 
            value=st.session_state.sampling_interval,
            help="Format: '1m', '10s', etc."
        )
        
        include_mission_phases = st.checkbox(
            "Include Mission Phases", 
            value=st.session_state.include_mission_phases,
            help="Simulate different mission phases (launch, orbit, etc.)"
        )
        
        random_seed = st.number_input(
            "Random Seed", 
            value=st.session_state.random_seed,
            help="Seed for reproducible simulations"
        )
    
    with col2:
        st.subheader("Anomaly Settings")
        
        include_anomalies = st.checkbox(
            "Include Anomalies", 
            value=st.session_state.include_anomalies
        )
        
        anomaly_rate = st.slider(
            "Anomaly Rate", 
            min_value=0.01, 
            max_value=0.2, 
            value=st.session_state.anomaly_rate,
            step=0.01,
            format="%.2f",
            help="Percentage of data points marked as anomalies"
        )
        
        base_temperature = st.number_input(
            "Base Temperature (°C)", 
            value=st.session_state.base_temperature,
            step=1.0
        )
        
        base_radiation = st.number_input(
            "Base Radiation (rads)", 
            value=st.session_state.base_radiation,
            step=1.0
        )
        
        radiation_threshold = st.number_input(
            "Radiation Threshold (rads)", 
            value=st.session_state.radiation_threshold,
            step=10.0,
            help="Threshold for radiation spike anomalies"
        )
    
    # Save button
    if st.button("Save Simulation Settings"):
        st.session_state.duration = duration
        st.session_state.sampling_interval = sampling_interval
        st.session_state.include_mission_phases = include_mission_phases
        st.session_state.random_seed = random_seed
        st.session_state.include_anomalies = include_anomalies
        st.session_state.anomaly_rate = anomaly_rate
        st.session_state.base_temperature = base_temperature
        st.session_state.base_radiation = base_radiation
        st.session_state.radiation_threshold = radiation_threshold
        
        st.session_state.regenerate_data = True
        st.session_state.retrain_model = True
        st.session_state.redetect_anomalies = True
        
        st.success("Settings saved! Click on Dashboard to see the results.")

elif page == "Model Settings":
    st.title("Model Settings")
    st.markdown("Configure the anomaly detection model parameters.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Configuration")
        
        model_type = st.selectbox(
            "Model Type", 
            options=["lstm", "transformer", "hybrid"],
            index=["lstm", "transformer", "hybrid"].index(st.session_state.model_type),
            help="LSTM: Good for short sequences, Transformer: Better for long-range dependencies, Hybrid: Combines both"
        )
        
        sequence_length = st.slider(
            "Sequence Length", 
            min_value=5, 
            max_value=100, 
            value=st.session_state.sequence_length,
            step=5,
            help="Number of time steps to consider for each prediction"
        )
    
    with col2:
        st.subheader("Feature Selection")
        
        # Load data if not already loaded
        telemetry_data, _ = load_or_simulate_data()
        
        # Get available features
        available_features = telemetry_data.select_dtypes(include=[np.number]).columns.tolist()
        available_features = [f for f in available_features if f not in ['is_anomaly', 'anomaly_score']]
        
        selected_features = st.multiselect(
            "Features to Use", 
            available_features,
            default=st.session_state.model_features if st.session_state.model_features else available_features
        )
        
        if not selected_features:
            selected_features = available_features
    
    # Save button
    if st.button("Save Model Settings and Retrain"):
        st.session_state.model_type = model_type
        st.session_state.sequence_length = sequence_length
        st.session_state.model_features = selected_features
        
        st.session_state.retrain_model = True
        st.session_state.redetect_anomalies = True
        
        st.success("Settings saved! Model will be retrained. Click on Dashboard to see the results.")
        
    # Display current model information if available
    if 'detector' in st.session_state:
        st.markdown("---")
        st.subheader("Current Model Information")
        
        detector = st.session_state.detector
        
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.markdown(f"**Model Type:** {detector.model_type.upper()}")
            st.markdown(f"**Sequence Length:** {detector.sequence_length}")
            
            if 'training_results' in st.session_state:
                results = st.session_state.training_results
                if 'f1_score' in results:
                    st.markdown(f"**F1 Score:** {results['f1_score']:.3f}")
                if 'precision' in results:
                    st.markdown(f"**Precision:** {results['precision']:.3f}")
                if 'recall' in results:
                    st.markdown(f"**Recall:** {results['recall']:.3f}")
        
        with info_col2:
            st.markdown("**Input Features:**")
            for feature in detector.input_features:
                st.markdown(f"- {feature}")
            
            if detector.threshold is not None:
                st.markdown(f"**Anomaly Threshold:** {detector.threshold:.6f}")

elif page == "Anomaly Analysis":
    st.title("Anomaly Analysis")
    
    # Make sure we have predictions
    predictions, anomaly_timestamps, explanations = detect_anomalies()
    
    if len(anomaly_timestamps) > 0:
        # Anomaly timeline
        st.subheader("Anomaly Timeline")
        
        anomaly_df = pd.DataFrame({
            'timestamp': anomaly_timestamps,
            'anomaly_score': [predictions.loc[ts, 'anomaly_score'] for ts in anomaly_timestamps]
        })
        
        timeline_fig = px.scatter(
            anomaly_df,
            x='timestamp',
            y='anomaly_score',
            size='anomaly_score',
            color='anomaly_score',
            title='Anomaly Timeline',
            labels={'timestamp': 'Time', 'anomaly_score': 'Anomaly Score'}
        )
        
        timeline_fig.update_layout(height=400)
        st.plotly_chart(timeline_fig, use_container_width=True)
        
        # Feature correlation analysis
        st.subheader("Feature Correlation Analysis")
        
        # Calculate correlation matrix
        telemetry_data, _ = load_or_simulate_data()
        numeric_data = telemetry_data.select_dtypes(include=[np.number])
        
        # Add anomaly score if available
        if 'anomaly_score' in predictions.columns:
            # Align indices
            common_idx = numeric_data.index.intersection(predictions.index)
            numeric_data = numeric_data.loc[common_idx]
            numeric_data['anomaly_score'] = predictions.loc[common_idx, 'anomaly_score']
        
        corr = numeric_data.corr()
        
        # Plot correlation heatmap
        corr_fig = px.imshow(
            corr,
            labels=dict(color="Correlation"),
            x=corr.columns,
            y=corr.columns,
            title="Feature Correlation Matrix"
        )
        
        corr_fig.update_layout(height=500)
        st.plotly_chart(corr_fig, use_container_width=True)
        
        # Anomaly distribution by feature
        st.subheader("Anomaly Distribution by Feature")
        
        # Get feature to analyze
        feature_to_analyze = st.selectbox(
            "Select feature for distribution analysis",
            options=st.session_state.model_features
        )
        
        # Create distribution plot
        normal_data = telemetry_data[~telemetry_data.index.isin(anomaly_timestamps)][feature_to_analyze]
        anomaly_data = telemetry_data[telemetry_data.index.isin(anomaly_timestamps)][feature_to_analyze]
        
        dist_fig = go.Figure()
        
        # Add normal distribution
        dist_fig.add_trace(go.Histogram(
            x=normal_data,
            name='Normal',
            opacity=0.75,
            marker_color='blue'
        ))
        
        # Add anomaly distribution
        dist_fig.add_trace(go.Histogram(
            x=anomaly_data,
            name='Anomaly',
            opacity=0.75,
            marker_color='red'
        ))
        
        dist_fig.update_layout(
            title=f"{feature_to_analyze} Distribution: Normal vs. Anomaly",
            xaxis_title=feature_to_analyze,
            yaxis_title="Count",
            barmode='overlay',
            height=400
        )
        
        st.plotly_chart(dist_fig, use_container_width=True)
        
        # Mission phase analysis if available
        if 'mission_phase' in telemetry_data.columns:
            st.subheader("Anomalies by Mission Phase")
            
            # Count anomalies by phase
            phase_counts = pd.DataFrame({
                'phase': telemetry_data.loc[anomaly_timestamps, 'mission_phase'].value_counts().index,
                'count': telemetry_data.loc[anomaly_timestamps, 'mission_phase'].value_counts().values
            })
            
            # Add total counts for percentage calculation
            total_by_phase = telemetry_data['mission_phase'].value_counts()
            percentages = []
            
            for phase in phase_counts['phase']:
                anomaly_count = phase_counts.loc[phase_counts['phase'] == phase, 'count'].iloc[0]
                total_count = total_by_phase[phase]
                percentages.append(anomaly_count / total_count * 100)
            
            phase_counts['percentage'] = percentages
            
            # Plot
            phase_fig = px.bar(
                phase_counts,
                x='phase',
                y='percentage',
                color='phase',
                text='count',
                title='Anomaly Percentage by Mission Phase',
                labels={'phase': 'Mission Phase', 'percentage': 'Anomaly Percentage (%)'}
            )
            
            phase_fig.update_layout(height=400)
            st.plotly_chart(phase_fig, use_container_width=True)
    else:
        st.info("No anomalies detected in the current data. Try generating new data with anomalies or adjusting the model settings.")

st.markdown("---")
st.caption("© 2025 MakarSpace - AI-Driven Anomaly Detection for Spacecraft")
