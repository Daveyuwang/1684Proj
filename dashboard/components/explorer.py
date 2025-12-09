"""
Annotation explorer component with filterable table and export functionality.
"""

from dash import dcc, html, Input, Output, State, callback
from dash.dash_table import DataTable
import dash_bootstrap_components as dbc
import pandas as pd
from dashboard.data_loader import get_data_loader

data_loader = get_data_loader()
available_datasets = data_loader.get_available_datasets()


def create_explorer_page():
    """Create the annotation explorer page with filters and table."""
    return html.Div([
        html.H1("Annotation Explorer", className="mb-4"),
        
        # Filters
        dbc.Card([
            dbc.CardHeader("Filters"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Dataset:", className="fw-bold"),
                        dcc.Dropdown(
                            id="explorer-dataset-selector",
                            options=[{"label": ds, "value": ds} for ds in available_datasets],
                            value=available_datasets[0] if available_datasets else None,
                            clearable=False,
                        ),
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Confidence Level:", className="fw-bold"),
                        dcc.Dropdown(
                            id="explorer-confidence-filter",
                            options=[
                                {"label": "All", "value": "all"},
                                {"label": "High", "value": "high"},
                                {"label": "Medium", "value": "medium"},
                                {"label": "Low", "value": "low"},
                            ],
                            value="all",
                            clearable=False,
                        ),
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Agreement Status:", className="fw-bold"),
                        dcc.Dropdown(
                            id="explorer-agreement-filter",
                            options=[
                                {"label": "All", "value": "all"},
                                {"label": "Correct", "value": "correct"},
                                {"label": "Incorrect", "value": "incorrect"},
                            ],
                            value="all",
                            clearable=False,
                        ),
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Filter Type:", className="fw-bold"),
                        dcc.Dropdown(
                            id="explorer-subset-filter",
                            options=[
                                {"label": "All Data", "value": "all"},
                                {"label": "Hard Cases Only", "value": "hard_cases"},
                            ],
                            value="all",
                            clearable=False,
                        ),
                    ], width=3),
                ], className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Label Class:", className="fw-bold"),
                        dcc.Dropdown(
                            id="explorer-label-filter",
                            options=[{"label": "All", "value": "all"}],
                            value="all",
                            clearable=False,
                        ),
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Text Length Range:", className="fw-bold"),
                        dcc.RangeSlider(
                            id="explorer-length-slider",
                            min=0,
                            max=10000,
                            step=100,
                            value=[0, 10000],
                            marks={i: str(i) for i in range(0, 10001, 2000)},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ], width=12),
                ]),
            ])
        ], className="mb-4"),
        
        # Results info and export
        dbc.Row([
            dbc.Col([
                html.Div(id="explorer-results-info", className="mb-2"),
            ], width=8),
            dbc.Col([
                dbc.Button("Export to CSV", id="explorer-export-btn", color="primary", className="float-end"),
                dcc.Download(id="explorer-download-csv"),
            ], width=4),
        ], className="mb-3"),
        
        # Data table
        html.Div(id="explorer-table-container"),
        html.Hr(),
        html.Div(id="explorer-detail-view", className="p-3 border rounded bg-light")
    ])


@callback(
    Output("explorer-label-filter", "options"),
    Input("explorer-dataset-selector", "value")
)
def update_label_filter(dataset_name):
    """Update label filter options based on selected dataset."""
    if not dataset_name:
        return [{"label": "All", "value": "all"}]
    
    dataset_info = data_loader.get_dataset_info(dataset_name)
    classes = dataset_info.get('classes', [])
    
    options = [{"label": "All", "value": "all"}]
    options.extend([{"label": cls.title(), "value": cls} for cls in classes])
    
    return options


@callback(
    [Output("explorer-table-container", "children"),
     Output("explorer-results-info", "children"),
     Output("explorer-length-slider", "max"),
     Output("explorer-length-slider", "value")],
    [Input("explorer-dataset-selector", "value"),
     Input("explorer-confidence-filter", "value"),
     Input("explorer-agreement-filter", "value"),
     Input("explorer-label-filter", "value"),
     Input("explorer-length-slider", "value"),
     Input("explorer-subset-filter", "value")]
)
def update_explorer_table(dataset_name, confidence_filter, agreement_filter, 
                          label_filter, length_range, subset_filter):
    """Update the explorer table based on filters."""
    if not dataset_name:
        return html.Div(), html.Div(), 10000, [0, 10000]
    
    # Load data
    df = data_loader.load_llm_annotations(dataset_name)
    
    if df.empty:
        return html.Div("No data available."), html.Div(), 10000, [0, 10000]
    
    # Apply filters
    filtered_df = df.copy()
    
    # Confidence filter
    if confidence_filter != "all":
        filtered_df = filtered_df[filtered_df['confidence_level'] == confidence_filter]
    
    # Agreement filter
    if agreement_filter != "all":
        filtered_df = filtered_df[filtered_df['agreement_status'] == agreement_filter]
    
    # Label filter
    if label_filter != "all":
        filtered_df = filtered_df[filtered_df['gold_label_str'] == label_filter]
    
    # Length filter
    if 'text_len' in filtered_df.columns:
        min_len, max_len = length_range
        filtered_df = filtered_df[
            (filtered_df['text_len'] >= min_len) & 
            (filtered_df['text_len'] <= max_len)
        ]
        max_length = int(df['text_len'].max()) if 'text_len' in df.columns else 10000
    else:
        max_length = 10000
    
    # Apply Hard Cases Filter
    if subset_filter == "hard_cases":
        hard_cases_df = data_loader.load_hard_cases(dataset_name)
        if not hard_cases_df.empty:
            # Filter df to only include rows present in hard_cases_df
            # Assuming 'row_id' is the common key
            hard_ids = hard_cases_df['row_id'].unique()
            filtered_df = filtered_df[filtered_df['row_id'].isin(hard_ids)]

    # Prepare columns for display
    display_columns = [
        {'name': 'Row ID', 'id': 'row_id', 'type': 'numeric'},
        {'name': 'Text Preview', 'id': 'text_preview', 'type': 'text'},
        {'name': 'Gold Label', 'id': 'gold_label_str', 'type': 'text'},
        {'name': 'LLM Label', 'id': 'llm_label_str', 'type': 'text'},
        {'name': 'Confidence', 'id': 'confidence_level', 'type': 'text'},
        {'name': 'Confidence (Numeric)', 'id': 'llm_confidence_numeric', 'type': 'numeric', 'format': {'specifier': '.2f'}},
        {'name': 'Agreement', 'id': 'agreement_status', 'type': 'text'},
    ]
    
    # Add text_len if available
    if 'text_len' in filtered_df.columns:
        display_columns.insert(2, {'name': 'Text Length', 'id': 'text_len', 'type': 'numeric'})
    
    # Update Display Columns to include Rationale if it exists
    if 'rationale' in filtered_df.columns:
         display_columns.append({'name': 'Rationale', 'id': 'rationale', 'type': 'text'})

    # Prepare data for table
    table_data = filtered_df[['row_id', 'gold_label_str', 'llm_label_str', 
                              'confidence_level', 'llm_confidence_numeric', 
                              'agreement_status']].copy()
    
    # Add text preview
    if 'text_preview' in filtered_df.columns:
        table_data['text_preview'] = filtered_df['text_preview']
    elif 'text' in filtered_df.columns:
        table_data['text_preview'] = filtered_df['text'].str[:100] + "..."
    else:
        table_data['text_preview'] = "N/A"
    
    if 'text_len' in filtered_df.columns:
        table_data['text_len'] = filtered_df['text_len']
    
    # Create data table
    table = DataTable(
        id="explorer-data-table",
        columns=display_columns,
        data=table_data.to_dict('records'),
        page_size=25,
        page_action="native",
        sort_action="native",
        filter_action="native",
        row_selectable='single', # Enable single row selection
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'fontFamily': 'Arial',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'maxWidth': 0,
        },
        style_cell_conditional=[
            {'if': {'column_id': 'text_preview'}, 'width': '30%'},
            {'if': {'column_id': 'row_id'}, 'width': '8%'},
        ],
        style_data_conditional=[
            {
                'if': {'filter_query': '{agreement_status} = incorrect'},
                'backgroundColor': '#ffcccc',
            },
            {
                'if': {'filter_query': '{confidence_level} = low'},
                'color': '#cc0000',
                'fontWeight': 'bold',
            },
        ],
        tooltip_data=[
            {
                column: {'value': str(value), 'type': 'markdown'}
                for column, value in row.items()
            } for row in table_data.to_dict('records')
        ],
        tooltip_duration=None,
    )
    
    # Results info
    results_info = html.P([
        f"Showing {len(filtered_df)} of {len(df)} annotations",
        html.Br(),
        f"Correct: {(filtered_df['is_correct'] == 1).sum()}, ",
        f"Incorrect: {(filtered_df['is_correct'] == 0).sum()}",
    ])
    
    return html.Div([table]), results_info, max_length, length_range


@callback(
    Output("explorer-download-csv", "data"),
    Input("explorer-export-btn", "n_clicks"),
    State("explorer-data-table", "data"),
    prevent_initial_call=True
)
def export_to_csv(n_clicks, table_data):
    """Export filtered table data to CSV."""
    if not table_data:
        return None
    
    df = pd.DataFrame(table_data)
    return dcc.send_data_frame(df.to_csv, "annotations_export.csv", index=False)


@callback(
    Output("explorer-detail-view", "children"),
    [Input("explorer-data-table", "selected_rows"),
     Input("explorer-data-table", "data")]
)
def display_selected_details(selected_rows, rows):
    if not selected_rows or not rows:
        return html.Div("Select a row to view details.", className="text-muted")
    
    row_data = rows[selected_rows[0]]
    
    return html.Div([
        html.H4(f"Details for Row {row_data.get('row_id', 'N/A')}"),
        html.H5("Full Text", className="mt-3"),
        html.P(row_data.get('text', 'Text not available in table view (ensure "text" is in data)')),
        
        html.H5("Rationale", className="mt-3"),
        html.P(row_data.get('rationale', 'No rationale available.')),
        
        dbc.Row([
            dbc.Col([
                html.Strong("Gold Label: "), html.Span(row_data.get('gold_label_str')),
            ], width=4),
            dbc.Col([
                html.Strong("LLM Label: "), html.Span(row_data.get('llm_label_str')),
            ], width=4),
            dbc.Col([
                html.Strong("Confidence: "), html.Span(f"{row_data.get('llm_confidence_numeric', 0):.2f}"),
            ], width=4),
        ], className="mt-3")
    ])

