import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

class AnomalyExplainer:
    """
    Explainability engine for spacecraft anomaly detection.
    
    This class provides methods to explain model predictions using SHAP and LIME,
    visualize feature importance, and generate natural language explanations
    for detected anomalies.
    """
    
    def __init__(
        self, 
        model: Any,
        feature_names: List[str],
        normal_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        """
        Initialize the anomaly explainer.
        
        Args:
            model: The anomaly detection model to explain
            feature_names: List of feature names
            normal_ranges: Dictionary mapping feature names to normal range tuples (min, max)
        """
        self.model = model
        self.feature_names = feature_names
        self.normal_ranges = normal_ranges or {}
        
    def _get_feature_contributions(
        self, 
        anomaly_data: pd.DataFrame,
        background_data: pd.DataFrame,
        method: str = 'shap'
    ) -> Dict[str, float]:
        """
        Get feature contributions to the anomaly using SHAP or LIME.
        
        Args:
            anomaly_data: DataFrame with anomalous data points
            background_data: DataFrame with normal data (for background distribution)
            method: Explanation method ('shap' or 'lime')
            
        Returns:
            Dictionary mapping feature names to contribution scores
        """
        if method.lower() == 'shap':
            # Create explainer
            explainer = shap.Explainer(self.model, background_data)
            
            # Compute SHAP values
            shap_values = explainer(anomaly_data)
            
            # Get mean absolute SHAP values across samples
            mean_shap = np.abs(shap_values.values).mean(axis=0)
            
            # Create dictionary of feature contributions
            contributions = dict(zip(self.feature_names, mean_shap))
            
        elif method.lower() == 'lime':
            # Create LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                background_data.values,
                feature_names=self.feature_names,
                class_names=['normal', 'anomaly'],
                discretize_continuous=True
            )
            
            # Explain first anomaly (for simplicity)
            exp = explainer.explain_instance(
                anomaly_data.iloc[0].values,
                self.model.predict_proba,
                num_features=len(self.feature_names)
            )
            
            # Get feature contributions
            contributions = {}
            for feature, importance in exp.as_list():
                feature_name = feature.split(' ')[0]  # Extract feature name
                contributions[feature_name] = abs(importance)
                
        else:
            raise ValueError(f"Unsupported explanation method: {method}")
        
        return contributions
    
    def explain_anomaly(
        self, 
        anomaly_point: pd.Series,
        normal_data: pd.DataFrame,
        method: str = 'shap',
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Generate an explanation for an anomalous data point.
        
        Args:
            anomaly_point: Series with anomalous data point
            normal_data: DataFrame with normal data for reference
            method: Explanation method ('shap' or 'lime')
            top_k: Number of top contributing features to include
            
        Returns:
            Dictionary with explanation information
        """
        # Get feature contributions
        contributions = self._get_feature_contributions(
            pd.DataFrame([anomaly_point]), normal_data, method
        )
        
        # Sort features by contribution
        sorted_features = sorted(
            contributions.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Get top-k contributing features
        top_features = sorted_features[:top_k]
        
        # Find deviations from normal ranges
        deviations = {}
        for feature, _ in top_features:
            if feature in self.normal_ranges:
                normal_min, normal_max = self.normal_ranges[feature]
                actual_value = anomaly_point[feature]
                
                if actual_value < normal_min:
                    deviation = (actual_value - normal_min) / normal_min * 100
                    deviations[feature] = {
                        'direction': 'below',
                        'normal_range': (normal_min, normal_max),
                        'actual_value': actual_value,
                        'percent_deviation': deviation
                    }
                elif actual_value > normal_max:
                    deviation = (actual_value - normal_max) / normal_max * 100
                    deviations[feature] = {
                        'direction': 'above',
                        'normal_range': (normal_min, normal_max),
                        'actual_value': actual_value,
                        'percent_deviation': deviation
                    }
        
        # Generate natural language explanation
        explanation_text = self._generate_explanation_text(top_features, deviations)
        
        return {
            'top_features': top_features,
            'deviations': deviations,
            'explanation_text': explanation_text,
            'all_contributions': contributions
        }
    
    def _generate_explanation_text(
        self, 
        top_features: List[Tuple[str, float]],
        deviations: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Generate natural language explanation for the anomaly.
        
        Args:
            top_features: List of tuples (feature_name, contribution_score)
            deviations: Dictionary of feature deviations from normal ranges
            
        Returns:
            Natural language explanation string
        """
        if not top_features:
            return "No significant anomalies detected."
        
        # Start with main anomaly cause
        primary_feature, _ = top_features[0]
        
        if primary_feature in deviations:
            dev_info = deviations[primary_feature]
            direction = dev_info['direction']
            pct_dev = abs(dev_info['percent_deviation'])
            
            explanation = f"Anomaly primarily caused by {primary_feature} reading {direction} " \
                          f"normal range by {pct_dev:.1f}%. "
            
            # Physics-based explanations for common spacecraft anomalies
            if primary_feature == 'radiation' and direction == 'above':
                explanation += "This is likely due to a solar flare or passing through a radiation belt. "
                
                # Check if temperature is also affected (with delay)
                if 'temperature' in [f for f, _ in top_features]:
                    explanation += "The elevated radiation has also caused a temperature increase. "
                    
            elif primary_feature == 'temperature' and direction == 'above':
                explanation += "This could be caused by excessive heat generation from electronics "
                
                # Check related features
                if 'voltage' in [f for f, _ in top_features]:
                    explanation += "or power system stress. "
                else:
                    explanation += "or insufficient thermal dissipation. "
                    
            elif primary_feature == 'voltage' and direction == 'below':
                explanation += "This indicates a power system issue, possibly due to "
                
                # Check if current is also affected
                if 'current' in [f for f, _ in top_features]:
                    explanation += "increased power draw or a partial short circuit. "
                else:
                    explanation += "battery degradation or solar panel underperformance. "
                    
            elif primary_feature == 'pressure' and direction != 'normal':
                explanation += "This could indicate a potential leak or valve malfunction "
                if direction == 'below':
                    explanation += "resulting in pressure loss. "
                else:
                    explanation += "causing pressurization beyond normal limits. "
                    
        else:
            # Generic explanation if no deviation info
            explanation = f"Anomaly primarily caused by unusual {primary_feature} behavior. "
            
        # Add secondary factors
        if len(top_features) > 1:
            secondary_features = [f for f, _ in top_features[1:]]
            explanation += f"Other contributing factors include {', '.join(secondary_features)}."
            
        # Add suggested action
        explanation += "\n\nRecommended action: "
        
        if primary_feature == 'radiation' and 'radiation' in deviations:
            if deviations['radiation']['percent_deviation'] > 200:
                explanation += "Immediately power down non-essential systems and initiate radiation protection protocols."
            else:
                explanation += "Monitor radiation levels and prepare to shield sensitive equipment if levels continue to rise."
                
        elif primary_feature == 'temperature':
            explanation += "Check cooling systems and reduce computational load to allow for thermal dissipation."
            
        elif primary_feature == 'voltage' or primary_feature == 'current':
            explanation += "Perform power system diagnostics and potentially switch to backup power sources."
            
        elif primary_feature == 'pressure':
            explanation += "Initiate leak detection protocols and prepare for pressure containment procedures."
            
        else:
            explanation += "Monitor the system closely and collect additional diagnostic data."
            
        return explanation
    
    def visualize_anomaly(
        self, 
        telemetry_data: pd.DataFrame, 
        anomaly_timestamps: List[datetime],
        features_to_plot: Optional[List[str]] = None,
        output_format: str = 'plotly',
        save_path: Optional[str] = None
    ) -> Any:
        """
        Visualize anomalies in telemetry data.
        
        Args:
            telemetry_data: DataFrame with telemetry data
            anomaly_timestamps: List of timestamps where anomalies were detected
            features_to_plot: List of features to include in visualization (if None, use all)
            output_format: 'plotly' or 'matplotlib'
            save_path: Path to save visualization (if None, display instead)
            
        Returns:
            Figure object
        """
        if features_to_plot is None:
            # Use all numeric features except anomaly columns
            features_to_plot = telemetry_data.select_dtypes(include=[np.number]).columns.tolist()
            features_to_plot = [f for f in features_to_plot if 'anomaly' not in f.lower()]
        
        if output_format.lower() == 'plotly':
            # Create subplots
            fig = go.Figure()
            
            for feature in features_to_plot:
                # Add trace for each feature
                fig.add_trace(go.Scatter(
                    x=telemetry_data.index,
                    y=telemetry_data[feature],
                    mode='lines',
                    name=feature
                ))
                
                # Add markers for anomalies
                if anomaly_timestamps:
                    anomaly_values = telemetry_data.loc[anomaly_timestamps, feature]
                    fig.add_trace(go.Scatter(
                        x=anomaly_values.index,
                        y=anomaly_values.values,
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
                hovermode='closest'
            )
            
            # Save if path provided
            if save_path:
                fig.write_html(save_path)
                
            return fig
            
        elif output_format.lower() == 'matplotlib':
            # Create subplots
            n_features = len(features_to_plot)
            fig, axes = plt.subplots(n_features, 1, figsize=(12, n_features * 3), sharex=True)
            
            if n_features == 1:
                axes = [axes]
                
            for i, feature in enumerate(features_to_plot):
                # Plot time series
                axes[i].plot(telemetry_data.index, telemetry_data[feature], label=feature)
                
                # Mark anomalies
                if anomaly_timestamps:
                    anomaly_values = telemetry_data.loc[anomaly_timestamps, feature]
                    axes[i].scatter(
                        anomaly_values.index, 
                        anomaly_values.values, 
                        color='red', 
                        marker='x', 
                        s=100, 
                        label='Anomaly'
                    )
                    
                axes[i].set_ylabel(feature)
                axes[i].legend()
                
                # Add normal range if available
                if feature in self.normal_ranges:
                    normal_min, normal_max = self.normal_ranges[feature]
                    axes[i].axhspan(normal_min, normal_max, alpha=0.2, color='green', label='Normal Range')
            
            # Set titles and format
            axes[0].set_title('Spacecraft Telemetry with Detected Anomalies')
            axes[-1].set_xlabel('Time')
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path)
                
            return fig
            
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
            
    def feature_importance_plot(
        self, 
        contributions: Dict[str, float],
        output_format: str = 'plotly',
        save_path: Optional[str] = None
    ) -> Any:
        """
        Create feature importance plot.
        
        Args:
            contributions: Dictionary mapping feature names to importance scores
            output_format: 'plotly' or 'matplotlib'
            save_path: Path to save visualization (if None, display instead)
            
        Returns:
            Figure object
        """
        # Sort features by importance
        sorted_items = sorted(contributions.items(), key=lambda x: x[1])
        features = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        
        if output_format.lower() == 'plotly':
            fig = px.bar(
                x=values,
                y=features,
                orientation='h',
                title='Feature Importance for Anomaly Detection',
                labels={'x': 'Importance Score', 'y': 'Feature'}
            )
            
            # Save if path provided
            if save_path:
                fig.write_html(save_path)
                
            return fig
            
        elif output_format.lower() == 'matplotlib':
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(features, values)
            ax.set_xlabel('Importance Score')
            ax.set_ylabel('Feature')
            ax.set_title('Feature Importance for Anomaly Detection')
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path)
                
            return fig
            
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
            
    def generate_report(
        self,
        anomaly_data: pd.DataFrame,
        explanation: Dict[str, Any],
        include_plots: bool = True,
        output_format: str = 'html',
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate an anomaly report for engineers.
        
        Args:
            anomaly_data: DataFrame with the anomalous data points
            explanation: Explanation dictionary from explain_anomaly method
            include_plots: Whether to include visualizations in the report
            output_format: 'html' or 'text'
            save_path: Path to save report (if None, return as string)
            
        Returns:
            Report as string
        """
        timestamp = anomaly_data.index[0]
        
        if output_format.lower() == 'html':
            report = f"""
            <html>
            <head>
                <title>Spacecraft Anomaly Report - {timestamp}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #2c3e50; }}
                    .alert {{ background-color: #f8d7da; border: 1px solid #f5c6cb; padding: 10px; border-radius: 5px; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                    .metric {{ font-weight: bold; margin-right: 10px; }}
                    .good {{ color: green; }}
                    .warning {{ color: orange; }}
                    .critical {{ color: red; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Spacecraft Anomaly Report</h1>
                    <p><strong>Timestamp:</strong> {timestamp}</p>
                    
                    <div class="alert">
                        <h2>Anomaly Explanation</h2>
                        <p>{explanation['explanation_text']}</p>
                    </div>
                    
                    <h2>Contributing Factors</h2>
                    <table>
                        <tr>
                            <th>Feature</th>
                            <th>Contribution</th>
                            <th>Value</th>
                            <th>Normal Range</th>
                            <th>Deviation</th>
                        </tr>
            """
            
            # Add rows for top features
            for feature, importance in explanation['top_features']:
                feature_value = anomaly_data[feature].iloc[0]
                normal_range = self.normal_ranges.get(feature, ('Unknown', 'Unknown'))
                
                if feature in explanation['deviations']:
                    dev_info = explanation['deviations'][feature]
                    deviation = f"{dev_info['percent_deviation']:.1f}% {dev_info['direction']}"
                    css_class = 'critical' if abs(dev_info['percent_deviation']) > 50 else 'warning'
                else:
                    deviation = "Within normal range"
                    css_class = 'good'
                
                report += f"""
                        <tr>
                            <td>{feature}</td>
                            <td>{importance:.4f}</td>
                            <td>{feature_value:.4f}</td>
                            <td>{normal_range[0]} - {normal_range[1]}</td>
                            <td class="{css_class}">{deviation}</td>
                        </tr>
                """
            
            report += """
                    </table>
                    
                    <h2>Telemetry Data</h2>
                    <table>
                        <tr>
                            <th>Feature</th>
                            <th>Value</th>
                        </tr>
            """
            
            # Add all telemetry values
            for column in anomaly_data.columns:
                if column not in ['is_anomaly', 'anomaly_score']:
                    value = anomaly_data[column].iloc[0]
                    report += f"""
                            <tr>
                                <td>{column}</td>
                                <td>{value:.4f}</td>
                            </tr>
                    """
            
            report += """
                    </table>
                </div>
            </body>
            </html>
            """
            
        elif output_format.lower() == 'text':
            report = f"Spacecraft Anomaly Report - {timestamp}\n"
            report += "=" * 50 + "\n\n"
            
            report += "ANOMALY EXPLANATION:\n"
            report += explanation['explanation_text'] + "\n\n"
            
            report += "CONTRIBUTING FACTORS:\n"
            report += f"{'Feature':<15} {'Contribution':<15} {'Value':<15} {'Normal Range':<20} {'Deviation':<15}\n"
            report += "-" * 80 + "\n"
            
            for feature, importance in explanation['top_features']:
                feature_value = anomaly_data[feature].iloc[0]
                normal_range = self.normal_ranges.get(feature, ('Unknown', 'Unknown'))
                
                if feature in explanation['deviations']:
                    dev_info = explanation['deviations'][feature]
                    deviation = f"{dev_info['percent_deviation']:.1f}% {dev_info['direction']}"
                else:
                    deviation = "Within normal range"
                    
                report += f"{feature:<15} {importance:<15.4f} {feature_value:<15.4f} "
                report += f"{normal_range[0]} - {normal_range[1]:<10} {deviation:<15}\n"
            
            report += "\nTELEMETRY DATA:\n"
            report += f"{'Feature':<15} {'Value':<15}\n"
            report += "-" * 30 + "\n"
            
            for column in anomaly_data.columns:
                if column not in ['is_anomaly', 'anomaly_score']:
                    value = anomaly_data[column].iloc[0]
                    report += f"{column:<15} {value:<15.4f}\n"
                    
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
            
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
                
        return report
