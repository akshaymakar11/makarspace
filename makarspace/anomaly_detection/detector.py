import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import json
from datetime import datetime

from makarspace.anomaly_detection.models.base_model import BaseAnomalyModel
from makarspace.anomaly_detection.models.lstm_model import LSTMAnomalyDetector
from makarspace.anomaly_detection.models.transformer_model import TransformerAnomalyDetector
from makarspace.anomaly_detection.visualization.explainer import AnomalyExplainer

class AnomalyDetector:
    """
    Main interface for spacecraft anomaly detection.
    
    This class serves as the primary interface for users to interact with
    the anomaly detection system. It provides methods for training models,
    detecting anomalies, and explaining results.
    """
    
    def __init__(
        self,
        model_type: str = 'lstm',
        input_features: List[str] = None,
        sequence_length: int = None,
        normal_ranges: Dict[str, Tuple[float, float]] = None,
        threshold: float = None,
        device: str = None
    ):
        """
        Initialize the anomaly detector.
        
        Args:
            model_type: Type of model to use ('lstm', 'transformer', or 'hybrid')
            input_features: List of feature names to use as input
            sequence_length: Length of input sequences (if None, use model defaults)
            normal_ranges: Dictionary mapping feature names to normal range tuples (min, max)
            threshold: Anomaly threshold (if None, set automatically during training)
            device: Device to use (e.g., 'cuda', 'cpu')
        """
        self.model_type = model_type.lower()
        self.input_features = input_features
        self.normal_ranges = normal_ranges or {}
        
        # Set model-specific defaults for sequence length
        if sequence_length is None:
            if self.model_type == 'lstm':
                sequence_length = 10
            elif self.model_type == 'transformer':
                sequence_length = 50
            elif self.model_type == 'hybrid':
                sequence_length = 30
                
        self.sequence_length = sequence_length
        self.threshold = threshold
        self.device = device
        
        # Initialize models based on type
        if self.model_type == 'lstm':
            self.model = LSTMAnomalyDetector(
                input_features=input_features,
                sequence_length=sequence_length,
                threshold=threshold,
                device=device
            )
        elif self.model_type == 'transformer':
            self.model = TransformerAnomalyDetector(
                input_features=input_features,
                sequence_length=sequence_length,
                threshold=threshold,
                device=device
            )
        elif self.model_type == 'hybrid':
            # Initialize both models for hybrid approach
            self.lstm_model = LSTMAnomalyDetector(
                input_features=input_features,
                sequence_length=sequence_length,
                threshold=threshold,
                device=device
            )
            self.transformer_model = TransformerAnomalyDetector(
                input_features=input_features,
                sequence_length=sequence_length,
                threshold=threshold,
                device=device
            )
            # Primary model for interface consistency
            self.model = self.lstm_model
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        # Initialize explainer (will be fully set up after fitting)
        self.explainer = None
        
        # Track training and prediction history
        self.history = {
            'training': {},
            'predictions': {},
            'anomalies': {}
        }
        
    def train(
        self, 
        data: Dict[str, pd.DataFrame],
        validation_split: float = 0.2,
        update_normal_ranges: bool = True
    ) -> Dict[str, Any]:
        """
        Train the anomaly detection model.
        
        Args:
            data: Dictionary with 'telemetry' and optional 'anomalies' DataFrames
            validation_split: Fraction of data to use for validation
            update_normal_ranges: Whether to update normal ranges based on training data
            
        Returns:
            Dictionary with training results
        """
        telemetry_data = data['telemetry']
        
        # Extract feature names if not provided
        if self.input_features is None:
            self.input_features = telemetry_data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Exclude any columns that might be labels
            exclude_cols = ['is_anomaly', 'anomaly_score', 'mission_phase']
            self.input_features = [f for f in self.input_features if f not in exclude_cols]
        
        # Update normal ranges if requested
        if update_normal_ranges:
            for feature in self.input_features:
                feature_data = telemetry_data[feature]
                q_low, q_high = feature_data.quantile([0.05, 0.95])
                self.normal_ranges[feature] = (q_low, q_high)
        
        # Split data into training and validation
        n_samples = len(telemetry_data)
        split_idx = int(n_samples * (1 - validation_split))
        
        train_data = telemetry_data.iloc[:split_idx]
        val_data = telemetry_data.iloc[split_idx:]
        
        # Train model based on type
        if self.model_type == 'hybrid':
            # Train both models
            print("Training LSTM model...")
            self.lstm_model.fit(train_data)
            
            print("Training Transformer model...")
            self.transformer_model.fit(train_data)
            
            # Validate both models
            lstm_results = self._validate_model(self.lstm_model, val_data, data.get('anomalies'))
            transformer_results = self._validate_model(self.transformer_model, val_data, data.get('anomalies'))
            
            # Use model with better F1 score as primary
            if lstm_results.get('f1_score', 0) >= transformer_results.get('f1_score', 0):
                self.model = self.lstm_model
                training_results = lstm_results
                print("LSTM model selected as primary based on validation performance.")
            else:
                self.model = self.transformer_model
                training_results = transformer_results
                print("Transformer model selected as primary based on validation performance.")
        else:
            # Train single model
            print(f"Training {self.model_type} model...")
            self.model.fit(train_data)
            
            # Validate model
            training_results = self._validate_model(self.model, val_data, data.get('anomalies'))
        
        # Initialize explainer with trained model
        self.explainer = AnomalyExplainer(
            model=self.model,
            feature_names=self.input_features,
            normal_ranges=self.normal_ranges
        )
        
        # Update history
        self.history['training'] = {
            'timestamp': datetime.now(),
            'data_size': len(telemetry_data),
            'results': training_results
        }
        
        return training_results
    
    def _validate_model(
        self, 
        model: BaseAnomalyModel,
        val_data: pd.DataFrame,
        true_anomalies: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Validate model performance.
        
        Args:
            model: Model to validate
            val_data: Validation data
            true_anomalies: Ground truth anomaly labels (if available)
            
        Returns:
            Dictionary of validation metrics
        """
        # Make predictions
        predictions = model.predict(val_data)
        
        # If true anomalies are provided, compute metrics
        if true_anomalies is not None:
            # Align predictions with true anomalies
            aligned_true = true_anomalies.loc[predictions.index, 'is_anomaly']
            results = model.evaluate(val_data, aligned_true)
        else:
            # Without ground truth, just return basic statistics
            anomaly_rate = predictions['is_anomaly'].mean()
            mean_score = predictions['anomaly_score'].mean()
            
            results = {
                'anomaly_rate': anomaly_rate,
                'mean_anomaly_score': mean_score,
                'threshold': model.threshold
            }
            
        return results
    
    def detect(
        self, 
        data: pd.DataFrame,
        explain: bool = False,
        ensemble_method: str = 'vote'
    ) -> Dict[str, Any]:
        """
        Detect anomalies in the given data.
        
        Args:
            data: DataFrame with telemetry data
            explain: Whether to generate explanations for detected anomalies
            ensemble_method: Method for combining predictions in hybrid mode ('vote' or 'avg')
            
        Returns:
            Dictionary with detection results
        """
        if self.model_type == 'hybrid':
            # Get predictions from both models
            lstm_predictions = self.lstm_model.predict(data)
            transformer_predictions = self.transformer_model.predict(data)
            
            # Combine predictions based on ensemble method
            if ensemble_method == 'vote':
                # Logical OR of anomaly flags
                combined_anomalies = (
                    lstm_predictions['is_anomaly'] | 
                    transformer_predictions['is_anomaly']
                )
                
                # Average of anomaly scores
                combined_scores = (
                    lstm_predictions['anomaly_score'] + 
                    transformer_predictions['anomaly_score']
                ) / 2
                
            elif ensemble_method == 'avg':
                # Average anomaly scores and apply threshold
                combined_scores = (
                    lstm_predictions['anomaly_score'] + 
                    transformer_predictions['anomaly_score']
                ) / 2
                
                # Use primary model's threshold
                combined_anomalies = combined_scores > self.model.threshold
                
            else:
                raise ValueError(f"Unsupported ensemble method: {ensemble_method}")
                
            # Create combined results DataFrame
            predictions = data.copy()
            predictions['anomaly_score'] = combined_scores
            predictions['is_anomaly'] = combined_anomalies
            
            # Also store individual model predictions
            predictions['lstm_anomaly_score'] = lstm_predictions['anomaly_score']
            predictions['lstm_is_anomaly'] = lstm_predictions['is_anomaly']
            predictions['transformer_anomaly_score'] = transformer_predictions['anomaly_score']
            predictions['transformer_is_anomaly'] = transformer_predictions['is_anomaly']
            
        else:
            # Single model prediction
            predictions = self.model.predict(data)
        
        # Get anomaly timestamps
        anomaly_timestamps = predictions.index[predictions['is_anomaly']].tolist()
        
        results = {
            'predictions': predictions,
            'anomaly_timestamps': anomaly_timestamps,
            'anomaly_count': len(anomaly_timestamps),
            'data_points': len(predictions)
        }
        
        # Generate explanations if requested
        if explain and anomaly_timestamps and self.explainer is not None:
            explanations = {}
            
            for timestamp in anomaly_timestamps:
                anomaly_point = predictions.loc[timestamp]
                
                # Prepare data for explainer
                anomaly_df = pd.DataFrame([anomaly_point])
                anomaly_df.index = [timestamp]
                
                # Generate explanation
                explanation = self.explainer.explain_anomaly(
                    anomaly_point=anomaly_point,
                    normal_data=data[~predictions['is_anomaly']],
                    method='shap',
                    top_k=3
                )
                
                explanations[timestamp] = explanation
                
            results['explanations'] = explanations
        
        # Update history
        self.history['predictions'][datetime.now()] = {
            'data_size': len(data),
            'anomaly_count': len(anomaly_timestamps)
        }
        
        return results
    
    def explain(
        self,
        anomaly_timestamp: datetime,
        telemetry_data: pd.DataFrame,
        predictions: pd.DataFrame = None,
        method: str = 'shap',
        output_format: str = 'text'
    ) -> Dict[str, Any]:
        """
        Generate explanation for a specific anomaly.
        
        Args:
            anomaly_timestamp: Timestamp of the anomaly to explain
            telemetry_data: Full telemetry data
            predictions: Prediction results (if None, will be computed)
            method: Explanation method ('shap' or 'lime')
            output_format: Format for explanation ('text', 'html', or 'dict')
            
        Returns:
            Dictionary with explanation information
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Train the model first.")
            
        # Get predictions if not provided
        if predictions is None:
            detect_results = self.detect(telemetry_data)
            predictions = detect_results['predictions']
            
        # Verify that timestamp is an anomaly
        if anomaly_timestamp not in predictions.index or not predictions.loc[anomaly_timestamp, 'is_anomaly']:
            raise ValueError(f"No anomaly detected at timestamp: {anomaly_timestamp}")
            
        # Get anomaly point
        anomaly_point = predictions.loc[anomaly_timestamp]
        
        # Create DataFrame with anomaly point
        anomaly_df = pd.DataFrame([anomaly_point])
        anomaly_df.index = [anomaly_timestamp]
        
        # Generate explanation
        explanation = self.explainer.explain_anomaly(
            anomaly_point=anomaly_point,
            normal_data=telemetry_data[~predictions['is_anomaly']],
            method=method,
            top_k=3
        )
        
        # Generate report if requested
        if output_format in ['text', 'html']:
            report = self.explainer.generate_report(
                anomaly_data=anomaly_df,
                explanation=explanation,
                include_plots=True,
                output_format=output_format
            )
            explanation['report'] = report
            
        # Update history
        self.history['anomalies'][anomaly_timestamp] = {
            'explanation_method': method,
            'top_features': [f[0] for f in explanation['top_features']]
        }
        
        return explanation
    
    def visualize(
        self,
        telemetry_data: pd.DataFrame,
        predictions: pd.DataFrame = None,
        features_to_plot: List[str] = None,
        output_format: str = 'plotly',
        save_path: Optional[str] = None
    ) -> Any:
        """
        Visualize telemetry data with anomalies.
        
        Args:
            telemetry_data: Telemetry data to visualize
            predictions: Prediction results (if None, will be computed)
            features_to_plot: List of features to plot (if None, use input features)
            output_format: Output format ('plotly' or 'matplotlib')
            save_path: Path to save visualization (if None, display only)
            
        Returns:
            Visualization figure
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Train the model first.")
            
        # Get predictions if not provided
        if predictions is None:
            detect_results = self.detect(telemetry_data)
            predictions = detect_results['predictions']
            anomaly_timestamps = detect_results['anomaly_timestamps']
        else:
            anomaly_timestamps = predictions.index[predictions['is_anomaly']].tolist()
            
        # Use input features if not specified
        if features_to_plot is None:
            features_to_plot = self.input_features
            
        # Generate visualization
        fig = self.explainer.visualize_anomaly(
            telemetry_data=telemetry_data,
            anomaly_timestamps=anomaly_timestamps,
            features_to_plot=features_to_plot,
            output_format=output_format,
            save_path=save_path
        )
        
        return fig
    
    def save(self, directory: str) -> None:
        """
        Save the anomaly detector to disk.
        
        Args:
            directory: Directory to save to
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save configuration
        config = {
            'model_type': self.model_type,
            'input_features': self.input_features,
            'sequence_length': self.sequence_length,
            'normal_ranges': self.normal_ranges,
            'history': self.history
        }
        
        with open(os.path.join(directory, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
            
        # Save models
        if self.model_type == 'hybrid':
            os.makedirs(os.path.join(directory, 'lstm_model'), exist_ok=True)
            os.makedirs(os.path.join(directory, 'transformer_model'), exist_ok=True)
            
            self.lstm_model.save(os.path.join(directory, 'lstm_model'))
            self.transformer_model.save(os.path.join(directory, 'transformer_model'))
        else:
            os.makedirs(os.path.join(directory, 'model'), exist_ok=True)
            self.model.save(os.path.join(directory, 'model'))
            
        print(f"Anomaly detector saved to {directory}")
        
    @classmethod
    def load(cls, directory: str) -> 'AnomalyDetector':
        """
        Load an anomaly detector from disk.
        
        Args:
            directory: Directory to load from
            
        Returns:
            Loaded AnomalyDetector instance
        """
        # Load configuration
        with open(os.path.join(directory, 'config.json'), 'r') as f:
            config = json.load(f)
            
        # Create instance
        detector = cls(
            model_type=config['model_type'],
            input_features=config['input_features'],
            sequence_length=config['sequence_length'],
        )
        
        # Restore normal ranges and history
        detector.normal_ranges = config['normal_ranges']
        detector.history = config['history']
        
        # Load models
        if detector.model_type == 'hybrid':
            detector.lstm_model = LSTMAnomalyDetector.load(
                os.path.join(directory, 'lstm_model')
            )
            detector.transformer_model = TransformerAnomalyDetector.load(
                os.path.join(directory, 'transformer_model')
            )
            
            # Set primary model
            detector.model = detector.lstm_model
        else:
            if detector.model_type == 'lstm':
                detector.model = LSTMAnomalyDetector.load(
                    os.path.join(directory, 'model')
                )
            elif detector.model_type == 'transformer':
                detector.model = TransformerAnomalyDetector.load(
                    os.path.join(directory, 'model')
                )
                
        # Initialize explainer
        detector.explainer = AnomalyExplainer(
            model=detector.model,
            feature_names=detector.input_features,
            normal_ranges=detector.normal_ranges
        )
        
        print(f"Anomaly detector loaded from {directory}")
        return detector
