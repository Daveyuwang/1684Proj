import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
from dashboard.data_loader import get_data_loader

data_loader = get_data_loader()
available_datasets = data_loader.get_available_datasets()

def create_export_page():
    return html.Div([
        html.H1("Data Export", className="mb-4"),
        
        dbc.Card([
            dbc.CardHeader("Export Configuration"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Select Dataset:", className="fw-bold"),
                        dcc.Dropdown(
                            id="export-dataset-selector",
                            options=[{"label": ds, "value": ds} for ds in available_datasets],
                            value=available_datasets[0] if available_datasets else None,
                            clearable=False,
                        ),
                    ], width=6),
                ], className="mb-4"),
                
                dbc.Label("Include Data:", className="fw-bold"),
                dbc.Checklist(
                    id="export-options",
                    options=[
                        {"label": "LLM Annotations (Labels, Confidence, Rationale)", "value": "annotations"},
                        {"label": "Trust Scores", "value": "trust"},
                        {"label": "DeBERTa Predictions", "value": "deberta"},
                        {"label": "Hard Cases Flag", "value": "hard_cases"},
                    ],
                    value=["annotations", "trust"],
                    switch=True,
                ),
                
                html.Hr(),
                
                dbc.Button("Generate & Download CSV", id="export-btn", color="success", size="lg"),
                dcc.Download(id="export-download")
            ])
        ])
    ])

@callback(
    Output("export-download", "data"),
    Input("export-btn", "n_clicks"),
    [State("export-dataset-selector", "value"),
     State("export-options", "value")],
    prevent_initial_call=True
)
def export_data(n_clicks, dataset_name, options):
    if not dataset_name:
        return None
    
    # Start with base annotations
    df = data_loader.load_llm_annotations(dataset_name)
    
    # Merge DeBERTa
    if "deberta" in options:
        deberta_df = data_loader.load_deberta_predictions(dataset_name)
        if not deberta_df.empty:
            merge_col = 'example_id' if 'example_id' in df.columns else 'row_id'
            if merge_col in deberta_df.columns:
                df = df.merge(deberta_df, on=merge_col, how='left', suffixes=('', '_deberta'))

    # Merge Trust Scores
    if "trust" in options:
        trust_df = data_loader.load_trust_scores(dataset_name)
        if not trust_df.empty:
            merge_col = 'example_id' if 'example_id' in df.columns else 'row_id'
            if merge_col in trust_df.columns:
                df = df.merge(trust_df, on=merge_col, how='left', suffixes=('', '_trust'))
                
    # Flag Hard Cases
    if "hard_cases" in options:
        hard_df = data_loader.load_hard_cases(dataset_name)
        if not hard_df.empty:
            # Assuming hard_cases has row_id or example_id
            id_col = 'row_id' if 'row_id' in hard_df.columns else 'example_id'
            hard_ids = set(hard_df[id_col])
            df['is_hard_case'] = df[id_col].apply(lambda x: x in hard_ids)
    
    return dcc.send_data_frame(df.to_csv, f"{dataset_name}_export.csv", index=False)
