"""
Main Plotly Dash application for LLM annotation reliability analysis dashboard.
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from pathlib import Path
import config

from dashboard.data_loader import get_data_loader
from dashboard.components.overview import create_overview_page
from dashboard.components.explorer import create_explorer_page
from dashboard.components.confidence_regions import create_confidence_page
from dashboard.components.fairness import create_fairness_page
from dashboard.components.export import create_export_page


# Initialize Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="LLM Annotation Reliability Dashboard"
)

# Initialize data loader
data_loader = get_data_loader()

# Get available datasets
available_datasets = data_loader.get_available_datasets()

# Define navigation sidebar
def create_sidebar():
    """Create the navigation sidebar."""
    return html.Div(
        [
            html.Div(
                [
                    html.H4("LLM Annotation Dashboard", className="text-white mb-4"),
                    html.Hr(className="text-white"),
                    dbc.Nav(
                        [
                            dbc.NavLink("Overview", href="/", active="exact", className="text-white"),
                            dbc.NavLink("Annotation Explorer", href="/explorer", active="exact", className="text-white"),
                            dbc.NavLink("Confidence Regions", href="/confidence", active="exact", className="text-white"),
                            dbc.NavLink("Fairness Analysis", href="/fairness", active="exact", className="text-white"),
                            dbc.NavLink("Export", href="/export", active="exact", className="text-white"),
                        ],
                        vertical=True,
                        pills=True,
                    ),
                ],
                className="p-4"
            ),
        ],
        style={
            "position": "fixed",
            "top": 0,
            "left": 0,
            "bottom": 0,
            "width": "250px",
            "background-color": "#343a40",
            "padding": "0",
        },
    )


# Main layout
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    create_sidebar(),
    html.Div(
        id="page-content",
        style={"margin-left": "250px", "padding": "20px"}
    ),
])


# Page components are imported from components module


# Callback for page routing
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def render_page_content(pathname):
    """Render the appropriate page based on URL pathname."""
    if pathname == "/":
        return create_overview_page()
    elif pathname == "/explorer":
        return create_explorer_page()
    elif pathname == "/confidence":
        return create_confidence_page()
    elif pathname == "/fairness":
        return create_fairness_page()
    elif pathname == "/export":
        return create_export_page()
    else:
        return html.Div([
            html.H1("404: Page not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ])


# Server instance for production deployment
server = app.server

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    app.run(debug=debug, host="0.0.0.0", port=port)

