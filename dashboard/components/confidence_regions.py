"""
Confidence regions visualization component with scatter plots and threshold visualization.
"""

from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from dashboard.data_loader import get_data_loader

data_loader = get_data_loader()
available_datasets = data_loader.get_available_datasets()


def create_confidence_page():
    """Create the confidence regions page with visualizations."""
    return html.Div([
        html.H1("Confidence Regions", className="mb-4"),
        
        # Controls
        dbc.Row([
            dbc.Col([
                dbc.Label("Dataset:", className="fw-bold"),
                dcc.Dropdown(
                    id="confidence-dataset-selector",
                    options=[{"label": ds, "value": ds} for ds in available_datasets],
                    value=available_datasets[0] if available_datasets else None,
                    clearable=False,
                ),
            ], width=4),
            dbc.Col([
                dbc.Label("Trust Model:", className="fw-bold"),
                dcc.Dropdown(
                    id="confidence-model-selector",
                    options=[
                        {"label": "Logistic Regression", "value": "lr"},
                        {"label": "Random Forest", "value": "rf"},
                    ],
                    value="lr",
                    clearable=False,
                ),
            ], width=4),
        ], className="mb-4"),
        
        # Confidence scatter plot
        html.Div(id="confidence-scatter-plot", className="mb-4"),
        
        # Policy thresholds visualization
        html.Div(id="confidence-policy-thresholds", className="mb-4"),
        
        # Coverage vs Accuracy curve
        html.Div(id="confidence-coverage-curve", className="mb-4"),
        
        # Calibration plot
        html.Div(id="confidence-calibration-plot", className="mb-4"),
    ])


@callback(
    [Output("confidence-scatter-plot", "children"),
     Output("confidence-policy-thresholds", "children"),
     Output("confidence-coverage-curve", "children"),
     Output("confidence-calibration-plot", "children")],
    [Input("confidence-dataset-selector", "value"),
     Input("confidence-model-selector", "value")]
)
def update_confidence_visualizations(dataset_name, model_type):
    """Update confidence visualizations based on selected dataset and model."""
    if not dataset_name:
        return html.Div(), html.Div(), html.Div(), html.Div()
    
    # Load joined data
    df = data_loader.get_joined_data(dataset_name, include_trust=True, model_type=model_type)
    
    if df.empty:
        return html.Div("No data available."), html.Div(), html.Div(), html.Div()
    
    # Create scatter plot
    scatter_plot = create_scatter_plot(df)
    
    # Create policy thresholds visualization
    policy_thresholds = create_policy_thresholds_viz(dataset_name, model_type, df)
    
    # Create coverage vs accuracy curve
    coverage_curve = create_coverage_accuracy_curve(df, model_type)
    
    # Create calibration plot
    calibration_plot = create_calibration_plot(df, model_type)
    
    return scatter_plot, policy_thresholds, coverage_curve, calibration_plot


def create_scatter_plot(df):
    """Create scatter plot of LLM confidence vs DeBERTa confidence."""
    if df.empty or 'llm_confidence_numeric' not in df.columns:
        return html.Div("Data not available for scatter plot.")
    
    # Get DeBERTa max probability
    prob_cols = [col for col in df.columns if col.startswith('prob_class')]
    if not prob_cols:
        return html.Div("DeBERTa predictions not available.")
    
    deberta_probs = df[prob_cols].values
    df_plot = df.copy()
    df_plot['deberta_p_max'] = deberta_probs.max(axis=1)
    
    # Color by agreement status
    df_plot['color'] = df_plot['is_correct'].map({1: 'Correct', 0: 'Incorrect'})
    
    fig = px.scatter(
        df_plot,
        x='llm_confidence_numeric',
        y='deberta_p_max',
        color='color',
        color_discrete_map={'Correct': '#2ca02c', 'Incorrect': '#d62728'},
        hover_data=['row_id', 'gold_label_str', 'llm_label_str'],
        title="LLM Confidence vs DeBERTa Confidence",
        labels={
            'llm_confidence_numeric': 'LLM Confidence',
            'deberta_p_max': 'DeBERTa Max Probability',
            'color': 'Agreement Status'
        }
    )
    
    # Add confidence threshold lines
    fig.add_hline(y=0.9, line_dash="dash", line_color="blue", 
                  annotation_text="High Confidence (0.9)")
    fig.add_hline(y=0.6, line_dash="dash", line_color="orange", 
                  annotation_text="Medium Confidence (0.6)")
    fig.add_hline(y=0.3, line_dash="dash", line_color="red", 
                  annotation_text="Low Confidence (0.3)")
    
    fig.add_vline(x=0.9, line_dash="dash", line_color="blue", 
                  annotation_text="High (0.9)")
    fig.add_vline(x=0.6, line_dash="dash", line_color="orange", 
                  annotation_text="Medium (0.6)")
    fig.add_vline(x=0.3, line_dash="dash", line_color="red", 
                  annotation_text="Low (0.3)")
    
    fig.update_layout(height=500, hovermode='closest')
    
    return dbc.Card([
        dbc.CardHeader("Confidence Comparison"),
        dbc.CardBody([dcc.Graph(figure=fig)])
    ])


def create_policy_thresholds_viz(dataset_name, model_type, df):
    """Create visualization of policy thresholds."""
    if df.empty or 'trust_score_calibrated' not in df.columns:
        return html.Div("Trust scores not available.")
    
    # Load trust config
    trust_config = data_loader.load_trust_config(dataset_name)
    if not trust_config or 'policy_thresholds' not in trust_config:
        return html.Div("Policy thresholds not available.")
    
    thresholds = trust_config['policy_thresholds']
    
    # Create histogram of trust scores with threshold lines
    fig = go.Figure()
    
    # Histogram of trust scores
    fig.add_trace(go.Histogram(
        x=df['trust_score_calibrated'],
        nbinsx=50,
        name='Trust Score Distribution',
        marker_color='lightblue',
        opacity=0.7
    ))
    
    # Add threshold lines
    colors = {'high_precision': 'red', 'balanced': 'orange', 'high_coverage': 'green'}
    for policy_name, threshold in thresholds.items():
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color=colors.get(policy_name, 'gray'),
            annotation_text=f"{policy_name.replace('_', ' ').title()}: {threshold:.3f}",
            annotation_position="top"
        )
    
    fig.update_layout(
        title="Trust Score Distribution with Policy Thresholds",
        xaxis_title="Trust Score",
        yaxis_title="Count",
        height=400,
        hovermode='x unified'
    )
    
    # Create table of policy metrics
    trust_metrics = data_loader.load_trust_metrics(dataset_name, model_type)
    policy_metrics = trust_metrics.get('policy_metrics', {})
    
    rows = []
    for policy_name, metrics in policy_metrics.items():
        rows.append(html.Tr([
            html.Td(policy_name.replace('_', ' ').title()),
            html.Td(f"{metrics.get('threshold', 0):.3f}"),
            html.Td(f"{metrics.get('coverage', 0):.2%}"),
            html.Td(f"{metrics.get('accepted_accuracy', 0):.2%}"),
            html.Td(f"{metrics.get('f1_score', 0):.3f}"),
        ]))
    
    table = dbc.Table([
        html.Thead([
            html.Tr([
                html.Th("Policy"),
                html.Th("Threshold"),
                html.Th("Coverage"),
                html.Th("Accepted Accuracy"),
                html.Th("F1 Score"),
            ])
        ]),
        html.Tbody(rows),
    ], bordered=True, hover=True, responsive=True, className="mt-3")
    
    return dbc.Card([
        dbc.CardHeader("Policy Thresholds"),
        dbc.CardBody([
            dcc.Graph(figure=fig),
            table
        ])
    ])


def create_coverage_accuracy_curve(df, model_type):
    """Create coverage vs accuracy trade-off curve."""
    if df.empty or 'trust_score_calibrated' not in df.columns or 'is_correct' not in df.columns:
        return html.Div("Data not available for coverage curve.")
    
    # Sort by trust score descending
    df_sorted = df.sort_values('trust_score_calibrated', ascending=False).copy()
    
    # Calculate coverage and accuracy at different thresholds
    n_total = len(df_sorted)
    thresholds = np.linspace(0, 1, 100)
    coverages = []
    accuracies = []
    
    for threshold in thresholds:
        accepted = df_sorted[df_sorted['trust_score_calibrated'] >= threshold]
        if len(accepted) > 0:
            coverage = len(accepted) / n_total
            accuracy = accepted['is_correct'].mean()
            coverages.append(coverage)
            accuracies.append(accuracy)
        else:
            coverages.append(0)
            accuracies.append(0)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=coverages,
        y=accuracies,
        mode='lines',
        name='Coverage vs Accuracy',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title="Coverage vs Accuracy Trade-off",
        xaxis_title="Coverage",
        yaxis_title="Accepted Accuracy",
        height=400,
        hovermode='x unified'
    )
    
    return dbc.Card([
        dbc.CardHeader("Coverage vs Accuracy Curve"),
        dbc.CardBody([dcc.Graph(figure=fig)])
    ])


def create_calibration_plot(df, model_type):
    """Create calibration plot (reliability diagram)."""
    if df.empty or 'trust_score_calibrated' not in df.columns or 'is_correct' not in df.columns:
        return html.Div("Data not available for calibration plot.")
    
    # Bin the predictions
    n_bins = 10
    df['bin'] = pd.cut(df['trust_score_calibrated'], bins=n_bins, labels=False)
    
    # Calculate mean predicted probability and mean actual accuracy per bin
    bin_stats = df.groupby('bin').agg({
        'trust_score_calibrated': 'mean',
        'is_correct': 'mean',
        'row_id': 'count'
    }).reset_index()
    bin_stats.columns = ['bin', 'mean_predicted', 'mean_actual', 'count']
    
    fig = go.Figure()
    
    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='gray', dash='dash')
    ))
    
    # Actual calibration curve
    fig.add_trace(go.Scatter(
        x=bin_stats['mean_predicted'],
        y=bin_stats['mean_actual'],
        mode='lines+markers',
        name='Calibration Curve',
        line=dict(color='blue', width=2),
        marker=dict(size=8),
        text=[f"Count: {c}" for c in bin_stats['count']],
        hovertemplate='Predicted: %{x:.3f}<br>Actual: %{y:.3f}<br>%{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Calibration Plot (Reliability Diagram)",
        xaxis_title="Mean Predicted Trust Score",
        yaxis_title="Mean Actual Accuracy",
        height=400,
        hovermode='closest',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1])
    )
    
    return dbc.Card([
        dbc.CardHeader("Calibration Plot"),
        dbc.CardBody([dcc.Graph(figure=fig)])
    ])

