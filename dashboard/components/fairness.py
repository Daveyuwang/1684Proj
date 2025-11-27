import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from dashboard.data_loader import get_data_loader

data_loader = get_data_loader()
available_datasets = data_loader.get_available_datasets()

def create_fairness_page():
    return html.Div([
        html.H1("Fairness Analysis", className="mb-4"),
        
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Select Dataset:", className="fw-bold"),
                        dcc.Dropdown(
                            id="fairness-dataset-selector",
                            options=[{"label": ds, "value": ds} for ds in available_datasets],
                            value=available_datasets[0] if available_datasets else None,
                            clearable=False,
                        ),
                    ], width=4),
                ]),
            ])
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Agreement Rate by Gold Label Class"),
                    dbc.CardBody(dcc.Graph(id="fairness-agreement-chart"))
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("LLM Confidence Distribution by Class"),
                    dbc.CardBody(dcc.Graph(id="fairness-confidence-chart"))
                ])
            ], width=6),
        ]),
    ])

@callback(
    [Output("fairness-agreement-chart", "figure"),
     Output("fairness-confidence-chart", "figure")],
    Input("fairness-dataset-selector", "value")
)
def update_fairness_charts(dataset_name):
    if not dataset_name:
        return {}, {}
    
    df = data_loader.load_llm_annotations(dataset_name)
    
    # Agreement by Class
    agreement_by_class = df.groupby('gold_label_str')['is_correct'].mean().reset_index()
    agreement_by_class.columns = ['Class', 'Accuracy']
    
    fig_agreement = px.bar(
        agreement_by_class, 
        x='Class', 
        y='Accuracy',
        color='Accuracy',
        range_y=[0, 1],
        title="Accuracy per Label Class",
        color_continuous_scale="Viridis"
    )

    # Confidence by Class
    fig_confidence = px.box(
        df,
        x='gold_label_str',
        y='llm_confidence_numeric',
        color='gold_label_str',
        title="Confidence Distribution per Label Class",
        points="outliers"
    )
    
    return fig_agreement, fig_confidence
