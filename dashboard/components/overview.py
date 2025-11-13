"""
Overview page component with high-level statistics and dataset selector.
"""

from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from dashboard.data_loader import get_data_loader

data_loader = get_data_loader()
available_datasets = data_loader.get_available_datasets()


def create_overview_page():
    """Create the overview page with statistics and dataset selector."""
    return html.Div([
        html.H1("Overview", className="mb-4"),
        
        # Dataset selector
        dbc.Row([
            dbc.Col([
                dbc.Label("Select Dataset:", className="fw-bold"),
                dcc.Dropdown(
                    id="overview-dataset-selector",
                    options=[{"label": ds, "value": ds} for ds in available_datasets],
                    value=available_datasets[0] if available_datasets else None,
                    clearable=False,
                ),
            ], width=4),
        ], className="mb-4"),
        
        # Statistics cards
        html.Div(id="overview-stats-cards", className="mb-4"),
        
        # Metrics table
        html.Div(id="overview-metrics-table", className="mb-4"),
        
        # Dataset comparison chart
        html.Div(id="overview-comparison-chart", className="mb-4"),
    ])


@callback(
    [Output("overview-stats-cards", "children"),
     Output("overview-metrics-table", "children"),
     Output("overview-comparison-chart", "children")],
    Input("overview-dataset-selector", "value")
)
def update_overview(dataset_name):
    """Update overview content based on selected dataset."""
    if not dataset_name:
        return html.Div(), html.Div(), html.Div()
    
    # Load metrics
    metrics = data_loader.load_agreement_metrics(dataset_name)
    dataset_info = data_loader.get_dataset_info(dataset_name)
    
    # Create statistics cards
    stats_cards = create_stats_cards(metrics, dataset_info)
    
    # Create metrics table
    metrics_table = create_metrics_table(metrics, dataset_info)
    
    # Create comparison chart
    comparison_chart = create_comparison_chart()
    
    return stats_cards, metrics_table, comparison_chart


def create_stats_cards(metrics, dataset_info):
    """Create statistics cards for key metrics."""
    if not metrics:
        return html.Div("No metrics available for this dataset.")
    
    cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Accuracy", className="card-title"),
                    html.H2(f"{metrics.get('accuracy', 0):.2%}", className="text-primary"),
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("F1 Score", className="card-title"),
                    html.H2(f"{metrics.get('f1_score', 0):.3f}", className="text-success"),
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Cohen's Kappa", className="card-title"),
                    html.H2(f"{metrics.get('cohen_kappa', 0):.3f}", className="text-info"),
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Avg Confidence", className="card-title"),
                    html.H2(f"{metrics.get('average_confidence', 0):.3f}", className="text-warning"),
                ])
            ])
        ], width=3),
    ])
    
    return cards


def create_metrics_table(metrics, dataset_info):
    """Create detailed metrics table."""
    if not metrics:
        return html.Div()
    
    # Extract classification report data
    class_report = metrics.get('classification_report', {})
    
    rows = []
    if isinstance(class_report, dict):
        for class_name, class_metrics in class_report.items():
            if isinstance(class_metrics, dict) and 'precision' in class_metrics:
                rows.append(html.Tr([
                    html.Td(class_name.title()),
                    html.Td(f"{class_metrics.get('precision', 0):.3f}"),
                    html.Td(f"{class_metrics.get('recall', 0):.3f}"),
                    html.Td(f"{class_metrics.get('f1-score', 0):.3f}"),
                    html.Td(f"{int(class_metrics.get('support', 0))}"),
                ]))
    
    table = dbc.Table([
        html.Thead([
            html.Tr([
                html.Th("Class"),
                html.Th("Precision"),
                html.Th("Recall"),
                html.Th("F1-Score"),
                html.Th("Support"),
            ])
        ]),
        html.Tbody(rows),
    ], bordered=True, hover=True, responsive=True, className="mt-3")
    
    return dbc.Card([
        dbc.CardHeader("Per-Class Performance Metrics"),
        dbc.CardBody([table])
    ])


def create_comparison_chart():
    """Create comparison chart across all datasets."""
    if not available_datasets:
        return html.Div()
    
    # Collect metrics for all datasets
    dataset_names = []
    accuracies = []
    f1_scores = []
    kappas = []
    
    for ds in available_datasets:
        metrics = data_loader.load_agreement_metrics(ds)
        if metrics:
            dataset_names.append(ds.upper())
            accuracies.append(metrics.get('accuracy', 0))
            f1_scores.append(metrics.get('f1_score', 0))
            kappas.append(metrics.get('cohen_kappa', 0))
    
    if not dataset_names:
        return html.Div()
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Accuracy',
        x=dataset_names,
        y=accuracies,
        marker_color='#1f77b4'
    ))
    
    fig.add_trace(go.Bar(
        name='F1 Score',
        x=dataset_names,
        y=f1_scores,
        marker_color='#2ca02c'
    ))
    
    fig.add_trace(go.Bar(
        name="Cohen's Kappa",
        x=dataset_names,
        y=kappas,
        marker_color='#ff7f0e'
    ))
    
    fig.update_layout(
        title="Performance Comparison Across Datasets",
        xaxis_title="Dataset",
        yaxis_title="Score",
        barmode='group',
        height=400,
        hovermode='x unified'
    )
    
    return dbc.Card([
        dbc.CardHeader("Dataset Comparison"),
        dbc.CardBody([
            dcc.Graph(figure=fig)
        ])
    ])

