import base64
import io

import dash
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, State, dash_table, ctx
import pandas as pd
import plotly.express as px
import numpy as np
from voa.copula import create_Q_plot

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# ---------------
# Helper Functions
# ---------------

def parse_contents(contents: str) -> pd.DataFrame:
    """
    Expects base64-encoded CSV data. Returns a DataFrame
    with a single column of numbers.
    """
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    # Load CSV data into a DataFrame
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), header=None)
    df.columns = ["Values"]
    return df

# ---------------
# Layout
# ---------------

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Upload File 1 (CSV)"),
            dcc.Upload(
                id='upload-data-1',
                children=html.Div(['Drag and Drop or ', html.A('Select File')]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center'
                },
                multiple=False
            ),
        ], width=6),
        dbc.Col([
            html.H2("Upload File 2 (CSV)"),
            dcc.Upload(
                id='upload-data-2',
                children=html.Div(['Drag and Drop or ', html.A('Select File')]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center'
                },
                multiple=False
            ),
        ], width=6),
    ]),

    html.Hr(),

    dbc.Row([
        # 1) New "Generate Plot" button
        dbc.Col([
            html.Button("Generate Plot", id="btn_generate_plot", n_clicks=0, className="btn btn-primary"),
        ], width="auto"),
    ]),

    html.Br(),

    dbc.Row([
        dbc.Col([
            # 2) dcc.Loading wrapper provides a spinner while the figure is updating
            dcc.Loading(
                id="loading-plot",
                type="default",
                children=[
                    # 3) Graph initially hidden via style={'display': 'none'}
                    dcc.Graph(id='my-plot', style={'display': 'none'})
                ],
            )
        ])
    ]),

    html.Hr(),

    dbc.Row([
        dbc.Col([
            html.Button("Download Results as CSVs", id="btn_download", n_clicks=0, className="btn btn-primary")
        ], width="auto"),
    ]),

    # Hidden dcc.Download components for separate CSV downloads
    dcc.Download(id="download-1"),
    dcc.Download(id="download-2"),

    # Stores to keep the uploaded data & result in memory
    dcc.Store(id='store-data-1'),
    dcc.Store(id='store-data-2'),
    dcc.Store(id='store-results-1'),
    dcc.Store(id='store-results-2'),
], fluid=True)

# ---------------
# Callbacks
# ---------------

@app.callback(
    Output('store-data-1', 'data'),
    Input('upload-data-1', 'contents'),
    prevent_initial_call=True
)
def update_data_1(contents):
    """
    Parses the contents of the first uploaded file
    and stores the data in dcc.Store.
    """
    if contents is not None:
        df1 = parse_contents(contents)
        return df1['Values']  # convert to dictionary for storing
    return dash.no_update

@app.callback(
    Output('store-data-2', 'data'),
    Input('upload-data-2', 'contents'),
    prevent_initial_call=True
)
def update_data_2(contents):
    """
    Parses the contents of the second uploaded file
    and stores the data in dcc.Store.
    """
    if contents is not None:
        df2 = parse_contents(contents)
        return df2['Values']
    return dash.no_update

# -----------------------------
# Plot generation on button click
# -----------------------------
@app.callback(
    # 4) We now have TWO outputs for the graph: the figure itself AND its style
    Output('my-plot', 'figure'),
    Output('my-plot', 'style'),
    Output('store-results-1', 'data'),
    Output('store-results-2', 'data'),
    Input('btn_generate_plot', 'n_clicks'),
    State('store-data-1', 'data'),
    State('store-data-2', 'data'),
    prevent_initial_call=True
)
def generate_plot_and_results(n_clicks, data1, data2):
    """
    Once both data sets are available and the user clicks "Generate Plot",
    generate the Q-plot and store the results.
    """
    if not data1 or not data2:
        # If either data is missing, don't show anything
        return px.scatter(), {'display': 'none'}, dash.no_update, dash.no_update

    # Do your calculations here (example dummy data):
    # X = np.array(list(range(100)))
    # Y = np.array([x ** 2 for x in X])

    # Create Q plot via your existing function
    result = create_Q_plot(data1, data2, k_plot_grid=100, MC=100, display=False)
    fig = result['Q_plot']

    # Return the figure, make it visible, and store the results
    return (
        fig,                       # The new figure
        {'display': 'block'},      # Reveal the plot
        pd.DataFrame(result['C_grid']).to_dict('records'),  # store-results-1
        pd.DataFrame(result['Q_grid']).to_dict('records')   # store-results-2
    )

@app.callback(
    Output("download-1", "data"),
    Output("download-2", "data"),
    Input("btn_download", "n_clicks"),
    State("store-results-1", "data"),
    State("store-results-2", "data"),
    prevent_initial_call=True
)
def download_csvs(n_clicks, data_sum, data_diff):
    """
    When the user clicks the download button, provide two CSV files.
    """
    if not n_clicks:
        return dash.no_update, dash.no_update

    if data_sum is None or data_diff is None:
        return dash.no_update, dash.no_update

    df_sum_result = pd.DataFrame(data_sum)
    df_diff_result = pd.DataFrame(data_diff)

    # Return two separate CSV files
    download1 = dict(
        content=df_sum_result.to_csv(index=False),
        filename="C_grid.csv",
        type="text/csv"
    )
    download2 = dict(
        content=df_diff_result.to_csv(index=False),
        filename="Q_grid.csv",
        type="text/csv"
    )

    return download1, download2

# ---------------
# Main
# ---------------
if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8051)
