#! /usr/bin/env python3

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import re
import webbrowser
import copy
import traceback
import numpy as np
import dash
from scipy.optimize import curve_fit
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dash.dependencies import Input, Output, State
import os
from scipy import stats
import plotly.io as pio
import orjson
import json
import glob
# import db_scan

def plot_df(df, x, y, df_name, append_to_file="", include_curve=False):
    add_fieldsheet = None
    graph_title = f'<b>{df_name}</b> {append_to_file}'
    num_na = df[y].isna().sum()
    if num_na == df.shape[0]:
        return {}
    try:
        for k, v in FIELDSHEET_DICT.items():
            if k in y and 'fieldsheet' not in y:
                add_fieldsheet = v
                break

        fig = px.scatter(color=df[y],
                         x=df[x],
                         y=df[y],
                         hover_name=df['index'],
                         opacity=.6,
                         color_continuous_scale='viridis',
                         title=graph_title)



        if add_fieldsheet:
            fig.add_trace(
                go.Scatter(
                    x=df[x],
                    y=df[add_fieldsheet],
                    mode='markers',
                    marker=dict(color='tomato'),
                    hoverinfo='skip',
                    name=add_fieldsheet,
                    showlegend=True,
                )
            )

        # if include_curve:
        #     fig.add_trace(
        #         go.Scatter(
        #             x=df[x],
        #             y=df[f"{y}_curve"],
        #             mode="lines",
        #             line=go.scatter.Line(color="gray"),
        #             showlegend=False)
        #     )

        fig.update_layout(font_family="proxima-nova", margin={'l': 40, 'b': 40, 't': 40, 'r': 0},
                          hovermode='closest', clickmode='event+select', dragmode='select')
        fig.update_yaxes(title=f'{y}')
        fig.update_xaxes(title=f'{x}')
        fig.update_traces(marker=dict(size=5))
    except:
        print(traceback.format_exc())
    return fig

def open_df(df_name):
    append_to_file = ""
    try:
        # else, if corrected file exists, use that
        return_df = pd.read_csv(f'{df_name.split(".csv")[0]}_corrected.csv')
        append_to_file = "with edits"
    except:
        # all else failing, use the vanilla df csv
        return_df = pd.read_csv(df_name)

    if return_df['datetime'].dtypes == 'datetime64[ns]':
        pass
    else:
        return_df['datetime'] = return_df['datetime'].apply(lambda x: " ".join(
            ["-".join(list(map(lambda y: y.zfill(2), x.split(" ")[0].split("-")))),
             ":".join(list(map(lambda y: y.zfill(2), x.split(" ")[1].split(":"))))]))
        return_df['datetime'] = pd.to_datetime(return_df.datetime, format='%y-%m-%d %H:%M:%S')

    return return_df, append_to_file

def init():
    # gets all site csv files and sorts splices from originals
    site_names_all = glob.glob('*.csv')
    site_names = [x for x in site_names_all if len(x.split("_")) == 2]

    #Creates a dictionary pairing non-empty column names to corresponding data site
    options = {}
    n = 10000  # chunk row size
    for site in site_names:
        print("now working on", site)

        # df = pd.read_csv(os.getcwd() + "/" + site)
        df, append_to_file = open_df(site)
        print(append_to_file)

        cols = []
        for col in df.columns:
            if df[col].dropna().empty:
                pass
            else:
                cols.append(col)

        # gets all splices for current site (excluding any final corrected versions as well)
        site_names_temp = [x for x in site_names_all if site[:-4] in x and not (len(x.split("_")) == 3 and "corrected" in x)]
        if len(site_names_temp) <= 1:
            # find first none null entry and start df there
            check_for_na = df.drop(columns=['index', 'indexInWaterYear', 'waterYear', 'datetime'])
            check_for_na = check_for_na.dropna(how="all")
            df = df[check_for_na.index[0]:]

            # mark where to splice df
            n_step = range(0, df.shape[0], n)

            # reformat dates if they aren't already of type datetime
            if df['datetime'].dtypes == 'datetime64[ns]':
                pass
            else:
                df['datetime'] = df['datetime'].apply(lambda x: " ".join(
                    ["-".join(list(map(lambda y: y.zfill(2), x.split(" ")[0].split("-")))),
                     ":".join(list(map(lambda y: y.zfill(2), x.split(" ")[1].split(":"))))]))
                df['datetime'] = pd.to_datetime(df.datetime, format='%y-%m-%d %H:%M:%S')

            # splice df
            list_df = [df[i:i + n] for i in n_step]

            # mark lines for generating image
            n_lines = df['datetime'][::n].tolist()
            n_lines.append(df['datetime'].iloc[-1])

            # save each df splice
            for i in range(len(list_df)):
                list_df[i].to_csv(f'{site[:-4]}_{i}.csv', index=False, date_format='%y-%m-%d %H:%M:%S')
                options[f'{site[:-4]}_{i}.csv'] = cols
                print(f'now printing {site[:-4]}_{i}.csv')

            # save image of entire df for each yaxis
            for yaxis in cols:
                try:
                    if yaxis != "datetime" and yaxis != "device":
                        image_path = f"assets/{site.split('_')[1].split('.csv')[0]}_images"

                        if not os.path.exists(image_path):
                            os.mkdir(image_path)

                        fig = px.scatter(color=df[yaxis],
                                         x=df['datetime'],
                                         y=df[yaxis],
                                         opacity=.6,
                                         color_continuous_scale='viridis',
                                         )

                        for i in range(len(n_lines)):
                            if i != len(n_lines) - 1:
                                if i % 2 == 0:
                                    fig.add_vrect(x0=n_lines[i], x1=n_lines[i + 1],
                                                  annotation_text=f"{i}", annotation_position="bottom right",
                                                  fillcolor="#b4b8b6", opacity=0.15, line_width=0, annotation=dict(font_size=50))
                                else:
                                    fig.add_vrect(x0=n_lines[i], x1=n_lines[i + 1],
                                                  annotation_text=f"{i}", annotation_position="bottom right",
                                                  fillcolor="#b4b8b6", opacity=0, line_width=0, annotation=dict(font_size=50))

                        fig.update_traces(marker=dict(size=4))
                        fig.update_layout(margin={'l': 40, 'b': 40, 't': 40, 'r': 0})
                        fig.write_image(f"{image_path}/{site.split('_')[1].split('.csv')[0]}_{yaxis}.png", width=3980, height=1080, scale=3)
                        print(f"now rendering {image_path}/{site.split('_')[1].split('.csv')[0]}_{yaxis}.png")
                    else:
                        print(f'not rendering an image for {yaxis}')
                except:
                    print(yaxis)
                    print(traceback.format_exc())

        else:
            # sort existing dfs
            spliced_dfs = sorted(list(set(site_names_temp) - set(site_names)))
            for df_splice in spliced_dfs:
                if 'corrected' not in df_splice:
                    options[df_splice] = cols
    return options

FIELDSHEET_DICT = {
    'temperature':'temperature_fieldsheet',
    'pH':'pH_fieldsheet',
    'orp':'orpMV_fieldsheet',
    'electrical':'electricalConductivity_fieldsheet',
    'pressure':'barometricPressure_fieldsheet',
    'dissolvedOxygenPercent':'dissolvedOxygenPercent_fieldsheet',
    'dissolvedOxygen_mgL':'dissolvedOxygen_mgL_fieldsheet',
    'dissolvedOxygen_mgl':'dissolvedOxygen_mgL_fieldsheet',
}

pio.templates.default = "plotly_white"
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

options = init()

config = dict({
    'displaylogo': False,
    'toImageButtonOptions': {
        'format': 'png', # one of png, svg, jpeg, webp
        'filename': 'custom_image',
        'height': 500,
        'width': 900,
        'scale': 3 # Multiply title/legend/axis/canvas sizes by this factor
      },
    'modeBarButtonsToAdd': [
        'drawopenpath',
        'eraseshape'
       ]})

colors = {
    'background': '#F8F9F9',
    'text': '#0C0C0B',
    'center': '#D1F2EB'
}

selectedPoints = []

app.layout = html.Div([
    #Header
    html.H1(
        children='site data visualization',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    #Subheader
    html.Div(children='''
        an interactive way to see and eliminate outliers!
    ''',style={
        'textAlign': 'center',
        'color': colors['text'],
    },
    id="wide"),

    html.Div([
        #This is the dropdown menu for individual site .csv files
        html.Label(["Site - ",

            dcc.Dropdown(
                id='df-name',
                options=[{'label': i, 'value': i} for i in options.keys()],
                value=f'{list(options.keys())[0]}'
            )
        ], style={'width': '48%', 'display': 'inline-block',
                  'justify-content': 'right'}),

    ]),

    html.Div([

        #X-axis drop down menu
        html.Label(["X-axis - ",
            dcc.Dropdown(id='xaxis-column')
        ], style={'width': '48%', 'display': 'inline-block'}),

        #Y-axis drop down menu
        html.Label(["Y-axis -",
            dcc.Dropdown(id='yaxis-column')
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),

    #Graph object
    dcc.Graph(id='indicator-graphic', config=config),

    html.Button('Remove', id='remove-val', n_clicks=0),
    html.Button('Increase data', id='up-val', n_clicks=0),
    html.Button('Decrease data', id='down-val', n_clicks=0),
    dcc.Input(id='input_number', type='number', placeholder=1, value=1, size="3"),
    html.Button('Correct Drift - start at left', id='drift-val-l', n_clicks=0),
    html.Button('Correct Drift - start at right', id='drift-val-r', n_clicks=0),
    # html.Button('Show curve for values', id='curve-val', n_clicks=0),
    html.Button('Revert edits for this splice', id='revert-val', n_clicks=0),
    html.Button('Export all splices for this site', id='export-val', n_clicks=0),
    # dcc.Checklist(
    #     options=[
    #         {'label': 'Toggle fieldsheet', 'value': "True"},
    #     ],
    #     value=['False'], id="toggle-fieldsheet"
    # ),

    html.Div([
        dcc.Markdown(id='test-text')]),

    dbc.Alert(
            "Error - No points for removal selected",
            color='primary',
            id="alert-auto",
            is_open=True,
            duration=4000,
    ),
    dbc.Alert(
            "export successful!",
            color='primary',
            id="alert-auto-export",
            is_open=False,
            duration=4000,
    ),

    html.Img(src='', id='image'),

    dcc.Store(id='json-progress', storage_type='local'),
    html.Div(id='intermediate-val', style={'display': 'none'}),

    html.Div([
    html.H1(
        children='instructions etc',
        id="instructions"),
    html.Div(children=''''''),
    html.Div(children='''Please don't rename any files or everything breaks!! :)''',id="wide6"),
    html.Div(children=''''''),
    html.Div(children='''Graph axes can be changed by dragging up/down or left/right on either of them. You can zoom in by using the magnifying glass tool. Select data points using either the box select (default option) or lasso select. You can add to your selection by holding down the 'shift' key as you make a new selection and remove from it by holding down the 'option' key. You can increase/decrease the step by which data is increased/decreased using the number input section. The correct drift buttons should only be used if you've identified drift in the data that needs to be corrected. Double clicking on the graph resets a selection/coloring, sometimes helpful.''', id="wide3"),
    html.Div(children=''''''),
    html.Div(children='''\n\nMy understanding is that aside from deleting obviously faulty data we won't be doing too much editing! The program also stores a copy of any data you delete, which we will later tag red when we are double checking with Ben/whoever is in charge so they can okay all edits before the final delete.''', id="wide4"),
    html.Div(children=''''''),
    html.Div(children='''\n\nThe overall graph at the bottom is a static image that displays your place in the current splice of the timeseries. Each entire timeseries is too large to load on the page at once so that is why we have to deal with the splices. As the bottom graph is a static image, it will not update as you edit the timeseries. If you really need to see an updated overall graph (as in the case of a couple outliers squashing the shape of the rest of the graph so as to make it not helpful), first delete the outliers, then export the data, and then go into the folder '~/Documents/timeseries-corrections-tool/' and delete every timeSeriesReport .csv file with a number in the filename (including those with 'corrected' at the end). Rerunning the program will take a bit to render each of the static images but that should give you an updated graph for each variable. You may also need to clear your browser cache to display the new image.''', id="wide2"),
    html.Div(children=''''''),
    html.Div(children='''\n\nIf there are red dots on the graph, those are the fieldsheet values for reference. (Don't rely on them too much)''', id="wide7"),
    html.Div(children=''''''),
    html.Div(children='''\n\nIf anything breaks/you have any questions feel free to reach out to me! (Zach) :)''', id="wide5")
], id='constrain-width'),
])

"""This callback updates options for the x and y axes based on the user's selection of data splice."""
@app.callback(
    Output('xaxis-column', 'options'),
    Output('yaxis-column', 'options'),
    Input('df-name', 'value'))
def update_df(df_name):
    x_opts = [{'label': i, 'value': i} for i in options[df_name]]
    y_opts = [{'label': i, 'value': i} for i in options[df_name]]
    return x_opts, y_opts

"""This callback updates the image path based on the user's selection of data splice and y-axis."""
@app.callback(
    Output('image', 'src'),
    Input('df-name', 'value'),
    Input('yaxis-column', 'value'))
def set_image(df_name, yaxis_val):
    image_path = f'assets/{df_name.split("_")[1].split(".csv")[0]}_images/{df_name.split("_")[1].split(".csv")[0]}_{yaxis_val}.png'
    return image_path

"""Initializes the values of the x and y axes based on the updated option list."""
@app.callback(
    Output('xaxis-column', 'value'),
    Output('yaxis-column', 'value'),
    Input('xaxis-column', 'options'),
    Input('yaxis-column', 'options'),
    Input('yaxis-column', 'value'))
def set_axis_values(xoptions,yoptions, yval):
    yval = yoptions[4]['value'] if yval is None else yval
    return 'datetime', yval

'''handles all data manipulation and graph updates'''
@app.callback(
    Output('indicator-graphic', 'figure'),
    Output('alert-auto', 'is_open'),
    Output('intermediate-val', 'children'),
    # Output('json-progress', 'data'),
    # Input('json-progress', 'data'),
    Input('remove-val', 'n_clicks'),
    Input('up-val', 'n_clicks'),
    Input('down-val', 'n_clicks'),
    Input('drift-val-l', 'n_clicks'),
    Input('drift-val-r', 'n_clicks'),
    Input('revert-val', 'n_clicks'),
    Input('export-val', 'n_clicks'),
    Input('input_number', 'value'),
    Input('xaxis-column', 'value'),
    Input('yaxis-column', 'value'),
    State('indicator-graphic', 'selectedData'),
    State('df-name', 'value'),
    State('alert-auto', 'is_open'),
    State('intermediate-val', 'children'))
def manipulate_data(remove_clicks, up_clicks, down_clicks, drift_clicks_l, drift_clicks_r, revert_clicks, export_clicks, input_number, xaxis_column_name, yaxis_column_name, selectedData, df_name, is_open, selectedPoints):

    '''This callback is activated whenever a click is made on one of the input buttons or when a different input value is
    changed. Unbelievably, the best method for determining which button was pressed is to keep track of how many times
    each button has been clicked and store it in a JSON in the DOM. This JSON is then parsed and compared with the
    new, updated clicks here, determining which path of the tree the code should follow.'''

    try:
        with open("progress.json") as p:
            click_progress = p.read()
    except:
        click_progress = None

    click_progress = json.loads(click_progress) if click_progress is not None else {"remove_clicks": 0, "up_clicks": 0, "down_clicks": 0, "drift_clicks_l": 0, "drift_clicks_r": 0, "revert_clicks": 0, "export_clicks": 0, "curve_clicks": 0}

    ### entry point (when no clicks have been made yet)
    if remove_clicks == 0 and up_clicks == 0 and drift_clicks_l == 0 and down_clicks == 0 and drift_clicks_r == 0 and revert_clicks == 0 and export_clicks == 0 or (type(input_number) != int and type(input_number) != float):
        working_df, append_to_file = open_df(df_name)

        xaxis = xaxis_column_name if xaxis_column_name is not None else "datetime"
        yaxis = yaxis_column_name if yaxis_column_name is not None else "datetime"

        fig = plot_df(working_df, xaxis, yaxis, df_name, append_to_file)

        with open('progress.json', 'w') as p:
            p.write(json.dumps({'remove_clicks': remove_clicks, 'up_clicks': up_clicks, "drift_clicks_l": drift_clicks_l,
                        "down_clicks": down_clicks, "drift_clicks_r": drift_clicks_r, "revert_clicks": revert_clicks,
                        "export_clicks": export_clicks, }))

        return fig, not is_open, selectedPoints

    ### remove values code
    elif remove_clicks > click_progress["remove_clicks"]:
        try:
            working_df, append_to_file = open_df(df_name)

            # working_df = working_df[~working_df[yaxis_column_name].isna()]
            x = [selectedData['points'][i]['hovertext'] for i in range(len(selectedData['points'])) if 'hovertext' in selectedData['points'][i].keys()]
            working_df['mask'] = working_df['index'].isin(x)
            
            # save data to be removed in column for later review
            if f'{yaxis_column_name}_bad' in working_df.columns.tolist():
                working_df[f'{yaxis_column_name}_bad'] = working_df[f'{yaxis_column_name}_bad'].where(~working_df['mask'], working_df[f"{yaxis_column_name}"])
            else:
                working_df[f'{yaxis_column_name}_bad'] = [None] * working_df.shape[0] # working_df[yaxis_column_name]
                working_df[f'{yaxis_column_name}_bad'] = working_df[f'{yaxis_column_name}_bad'].where(~working_df['mask'], working_df[f"{yaxis_column_name}"])

            # remove data from main column
            working_df[yaxis_column_name] = working_df[yaxis_column_name].where(~working_df['mask'], np.nan)
            working_df = working_df.drop(columns={"mask"})

            selectedPoints = set() if selectedPoints is None else set(selectedPoints)
            selectedPoints.update(x)

            fig = plot_df(working_df, xaxis_column_name, yaxis_column_name, df_name, 'with edits')

            # save progress
            working_df.to_csv(f"{df_name[:-4]}_corrected.csv", index=False, date_format='%y-%m-%d %H:%M:%S')

            with open('progress.json', 'w') as p:
                p.write(json.dumps({'remove_clicks': remove_clicks, 'up_clicks': up_clicks, "drift_clicks_l": drift_clicks_l,
                            "down_clicks": down_clicks, "drift_clicks_r": drift_clicks_r,
                            "revert_clicks": revert_clicks,
                            "export_clicks": export_clicks, }))

            if not selectedPoints:
                return fig, is_open, x
            else:
                return fig, is_open, list(selectedPoints)
        except:
            print(traceback.format_exc())

    ### increase values code
    elif up_clicks > click_progress["up_clicks"]:
        try:
            working_df, append_to_file = open_df(df_name)
            # working_df = working_df[~working_df[yaxis_column_name].isna()]

            x = [selectedData['points'][i]['hovertext'] for i in range(len(selectedData['points'])) if 'hovertext' in selectedData['points'][i].keys()]
            working_df['mask'] = working_df['index'].isin(x)

            try:
                working_df["add"] = [input_number] * working_df.shape[0]
                working_df["add"] = working_df["add"].where(working_df['mask'], 0)
                working_df[yaxis_column_name] = working_df[yaxis_column_name] + working_df["add"]
                working_df = working_df.drop(columns={"mask", "add"})
            except:
                print(traceback.format_exc())

            selectedPoints = set() if selectedPoints is None else set(selectedPoints)
            selectedPoints.update(x)

            fig = plot_df(working_df, xaxis_column_name, yaxis_column_name, df_name, 'with edits')

            # save progress
            working_df.to_csv(f"{df_name[:-4]}_corrected.csv", index=False, date_format='%y-%m-%d %H:%M:%S')

            with open('progress.json', 'w') as p:
                p.write(json.dumps({'remove_clicks': remove_clicks, 'up_clicks': up_clicks, "drift_clicks_l": drift_clicks_l,
                            "down_clicks": down_clicks, "drift_clicks_r": drift_clicks_r,
                            "revert_clicks": revert_clicks,
                            "export_clicks": export_clicks, }))

            if not selectedPoints:
                return fig, is_open, x
            else:
                return fig, is_open, list(selectedPoints)
        except:
            print(traceback.format_exc())

    ### decrease values code
    elif down_clicks > click_progress["down_clicks"]:
        try:
            working_df, append_to_file = open_df(df_name)

            # working_df = working_df[~working_df[yaxis_column_name].isna()]
            x = [selectedData['points'][i]['hovertext'] for i in range(len(selectedData['points'])) if 'hovertext' in selectedData['points'][i].keys()]
            working_df['mask'] = working_df['index'].isin(x)
            try:
                working_df["add"] = [-input_number] * working_df.shape[0]
                working_df["add"] = working_df["add"].where(working_df['mask'], 0)
                working_df[yaxis_column_name] = working_df[yaxis_column_name] + working_df["add"]
                working_df = working_df.drop(columns={"mask", "add"})
            except:
                print(traceback.format_exc())

            selectedPoints = set() if selectedPoints is None else set(selectedPoints)
            selectedPoints.update(x)

            fig = plot_df(working_df, xaxis_column_name, yaxis_column_name, df_name, 'with edits')

            # save progress
            working_df.to_csv(f"{df_name[:-4]}_corrected.csv", index=False, date_format='%y-%m-%d %H:%M:%S')

            with open('progress.json', 'w') as p:
                p.write(json.dumps({'remove_clicks': remove_clicks, 'up_clicks': up_clicks, "drift_clicks_l": drift_clicks_l,
                            "down_clicks": down_clicks, "drift_clicks_r": drift_clicks_r,
                            "revert_clicks": revert_clicks,
                            "export_clicks": export_clicks, }))

            if not selectedPoints:
                return fig, is_open, x
            else:
                return fig, is_open, list(selectedPoints)
        except:
            print(traceback.format_exc())

    ### correct points for drift starting at right
    elif drift_clicks_r > click_progress["drift_clicks_r"]:
        try:
            working_df, append_to_file = open_df(df_name)

            # working_df = working_df[~working_df[yaxis_column_name].isna()]
            x = [selectedData['points'][i]['hovertext'] for i in range(len(selectedData['points'])) if 'hovertext' in selectedData['points'][i].keys()]
            working_df['mask'] = working_df['index'].isin(x)
            try:
                x_line = working_df[working_df['mask']]['index']
                y_line = working_df[working_df['mask']][yaxis_column_name]
                slope, intercept, r, p, se = stats.linregress(x_line, y_line)

                working_df["drift_correct"] = (slope * max(x)) + (-slope * working_df["index"])
                working_df["drift_correct"] = working_df["drift_correct"].where((min(x) <= working_df["index"]) & (working_df["index"] <= max(x)), 0)
                working_df[yaxis_column_name] = working_df[yaxis_column_name] + working_df["drift_correct"]
                working_df = working_df.drop(columns={"mask", "drift_correct"})
            except:
                print(traceback.format_exc())

            selectedPoints = set() if selectedPoints is None else set(selectedPoints)
            selectedPoints.update(x)

            fig = plot_df(working_df, xaxis_column_name, yaxis_column_name, df_name, 'with edits')

            # save progress
            working_df.to_csv(f"{df_name[:-4]}_corrected.csv", index=False, date_format='%y-%m-%d %H:%M:%S')

            with open('progress.json', 'w') as p:
                p.write(json.dumps({'remove_clicks': remove_clicks, 'up_clicks': up_clicks, "drift_clicks_l": drift_clicks_l,
                            "down_clicks": down_clicks, "drift_clicks_r": drift_clicks_r,
                            "revert_clicks": revert_clicks,
                            "export_clicks": export_clicks, }))

            if not selectedPoints:
                return fig, is_open, x
            else:
                return fig, is_open, list(selectedPoints)
        except:
            print(traceback.format_exc())

    ### correct points for drift starting at left
    elif drift_clicks_l > click_progress["drift_clicks_l"]:
        try:
            working_df, append_to_file = open_df(df_name)

            # working_df = working_df[~working_df[yaxis_column_name].isna()]
            x = [selectedData['points'][i]['hovertext'] for i in range(len(selectedData['points'])) if 'hovertext' in selectedData['points'][i].keys()]
            working_df['mask'] = working_df['index'].isin(x)
            try:
                x_line = working_df[working_df['mask']]['index']
                y_line = working_df[working_df['mask']][yaxis_column_name]
                slope, intercept, r, p, se = stats.linregress(x_line, y_line)
                working_df["drift_correct"] = (slope * min(x)) + (-slope * working_df["index"])
                working_df["drift_correct"] = working_df["drift_correct"].where((min(x) <= working_df["index"]) & (working_df["index"] <= max(x)), 0)
                working_df[yaxis_column_name] = working_df[yaxis_column_name] + working_df["drift_correct"]
                working_df = working_df.drop(columns={"mask", "drift_correct"})
            except:
                print(traceback.format_exc())

            selectedPoints = set() if selectedPoints is None else set(selectedPoints)
            selectedPoints.update(x)

            fig = plot_df(working_df, xaxis_column_name, yaxis_column_name, df_name, 'with edits')

            # save progress
            working_df.to_csv(f"{df_name[:-4]}_corrected.csv", index=False, date_format='%y-%m-%d %H:%M:%S')

            with open('progress.json', 'w') as p:
                p.write(json.dumps({'remove_clicks': remove_clicks, 'up_clicks': up_clicks, "drift_clicks_l": drift_clicks_l,
                            "down_clicks": down_clicks, "drift_clicks_r": drift_clicks_r,
                            "revert_clicks": revert_clicks,
                            "export_clicks": export_clicks, }))

            if not selectedPoints:
                return fig, is_open, x
            else:
                return fig, is_open, list(selectedPoints)
        except:
            print(traceback.format_exc())

    # ### show curve
    # elif curve_clicks > click_progress["curve_clicks"]:
    #     try:
    #         working_df, append_to_file = open_df(df_name)
    #
    #         x = [selectedData['points'][i]['hovertext'] for i in range(len(selectedData['points'])) if 'hovertext' in selectedData['points'][i].keys()]
    #         working_df['mask'] = working_df['index'].isin(x)
    #
    #         try:
    #             x = working_df[working_df["mask"]][xaxis_column_name].tolist()
    #             y = working_df[working_df["mask"]][f"{yaxis_column_name}"].tolist()
    #             min_ind = x[0]
    #             max_ind = x[-1]
    #
    #             popt, _ = curve_fit(objective, x, y)
    #
    #             # define curve fit to YSI points
    #             working_df[f"{yaxis_column_name}_curve"] = objective(working_df[xaxis_column_name], *popt)
    #
    #             # cut off errant tails of curve
    #             working_df[f"{yaxis_column_name}_curve"] = working_df[f"{yaxis_column_name}_curve"].where((working_df[xaxis_column_name] > min_ind), None)
    #             working_df[f"{yaxis_column_name}_curve"] = working_df[f"{yaxis_column_name}_curve"].where((working_df[xaxis_column_name] < max_ind), None)
    #         except:
    #             print(traceback.format_exc())
    #
    #         selectedPoints = set() if selectedPoints is None else set(selectedPoints)
    #         selectedPoints.update(x)
    #
    #         fig = plot_df(working_df, xaxis_column_name, yaxis_column_name, df_name, 'with edits', True)
    #
    #         working_df = working_df.drop(columns={f"{yaxis_column_name}_curve"})
    #
    #         # save progress
    #         working_df.to_csv(f"{df_name[:-4]}_corrected.csv", index=False, date_format='%y-%m-%d %H:%M:%S')
    #
    #         if not selectedPoints:
    #             return fig, is_open, x, json.dumps({'remove_clicks':remove_clicks, 'up_clicks': up_clicks, "drift_clicks_l": drift_clicks_l, "down_clicks": down_clicks, "drift_clicks_r": drift_clicks_r, "revert_clicks": revert_clicks, "export_clicks": export_clicks, })
    #         else:
    #             return fig, is_open, list(selectedPoints), json.dumps({'remove_clicks':remove_clicks, 'up_clicks': up_clicks, "drift_clicks_l": drift_clicks_l, "down_clicks": down_clicks, "drift_clicks_r": drift_clicks_r, "revert_clicks": revert_clicks, "export_clicks": export_clicks, })
    #     except:
    #         print(traceback.format_exc())

    ### revert progress!
    elif revert_clicks > click_progress["revert_clicks"]:
        try:
            if os.path.exists(f"{df_name[:-4]}_corrected.csv"):
                os.remove(f"{df_name[:-4]}_corrected.csv")
            else:
                print("The file does not exist")

            working_df, append_to_file = open_df(df_name)
            selectedPoints = set() if selectedPoints is None else set(selectedPoints)
            fig = plot_df(working_df, xaxis_column_name, yaxis_column_name, df_name, append_to_file)
        except:
            print(traceback.format_exc())

        with open('progress.json', 'w') as p:
            p.write(json.dumps({'remove_clicks': remove_clicks, 'up_clicks': up_clicks, "drift_clicks_l": drift_clicks_l,
                        "down_clicks": down_clicks, "drift_clicks_r": drift_clicks_r, "revert_clicks": revert_clicks,
                        "export_clicks": export_clicks, }))

        return fig, is_open, list(selectedPoints)

    ### export df!
    elif export_clicks > click_progress["export_clicks"]:
        # save current progress on df, if any
        working_df, append_to_file = open_df(df_name)
        working_df.to_csv(f"{df_name[:-4]}_corrected.csv", index=False, date_format='%y-%m-%d %H:%M:%S')

        # strip df_name for values
        prepend = df_name.split("_")[0]
        site = df_name.split("_")[1]

        # get list of files
        splices = [x for x in glob.glob('*.csv') if len(x.split("_")) > 2]
        splices = [x for x in splices if not (len(x.split("_")) == 3 and "corrected" in x)]
        splices = sorted([x[:-4] for x in splices if site in x])
        delete_later = copy.copy(splices)

        # find which files have corrected versions so we use those if available
        has_corrected = {}
        for i in range(len(splices)):
            has_corrected[splices[i].split("_")[2]] = True if "corrected" in splices[i] else False
            splices[i] = splices[i].split("_")[2]

        unique_splices = sorted(list({f"{prepend}_{site}_{csv}_corrected.csv" if has_corrected[csv] else f"{prepend}_{site}_{csv}.csv" for csv in splices}))
        df = pd.concat((pd.read_csv(csv) for csv in unique_splices), axis=0, ignore_index=True)
        df.to_csv(f"{prepend}_{site}_corrected.csv", index=False, date_format='%y-%m-%d %H:%M:%S')

        working_df, append_to_file = open_df(f"{prepend}_{site}_corrected.csv")
        selectedPoints = set() if selectedPoints is None else set(selectedPoints)
        fig = plot_df(working_df, xaxis_column_name, yaxis_column_name, df_name, 'completed edits - restart the app to continue editing<br>(may need to clear browser cache to refresh image)')

        # delete all splices
        for splice in delete_later:
            os.remove(f"{splice}.csv")

        with open('progress.json', 'w') as p:
            p.write(json.dumps({'remove_clicks': remove_clicks, 'up_clicks': up_clicks, "drift_clicks_l": drift_clicks_l,
             "down_clicks": down_clicks, "drift_clicks_r": drift_clicks_r, "revert_clicks": revert_clicks,
             "export_clicks": export_clicks, }))

        return fig, is_open, list(selectedPoints)

    elif selectedData:
        working_df, append_to_file = open_df(df_name)

        xaxis = xaxis_column_name if xaxis_column_name is not None else "pressure_hobo"
        yaxis = yaxis_column_name if yaxis_column_name is not None else "index"

        fig = plot_df(working_df, xaxis, yaxis, df_name, append_to_file)

        return fig, not is_open, selectedPoints, json.dumps({'remove_clicks': remove_clicks, 'up_clicks': up_clicks, 'down_clicks': down_clicks, "drift_clicks_l": drift_clicks_l, 'drift_clicks_r': drift_clicks_r, "revert_clicks": revert_clicks, "export_clicks": export_clicks, })

    elif not selectedData:
        try:
            working_df, append_to_file = open_df(df_name)

            xaxis = xaxis_column_name if xaxis_column_name is not None else "pressure_hobo"
            yaxis = yaxis_column_name if yaxis_column_name is not None else "index"

            fig = plot_df(working_df, xaxis, yaxis, df_name, append_to_file)

            return fig, not is_open, selectedPoints, json.dumps({'remove_clicks':remove_clicks, 'up_clicks': up_clicks, 'down_clicks': down_clicks, "drift_clicks_l": drift_clicks_l, "drift_clicks_r": drift_clicks_r, "revert_clicks": revert_clicks, "export_clicks": export_clicks, })
        except:
            print(traceback.format_exc())

def objective(x, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, y, z):
    return (a * x) + (b * x ** 2) + (c * x ** 3) + (d * x ** 4) + (e * x ** 5) + (f * x ** 6) + (g * x ** 7) + (h * x ** 8) + (i * x ** 9) + (j * x ** 10) + (k * x ** 11) + (l * x ** 12) + (m * x ** 13) + (n * x ** 14) + (o * x ** 15) + (p * x ** 16) + (q * x ** 17) + (r * x ** 18) + (s * x ** 19) + (t * x ** 20) + (u * x ** 21) + (v * x ** 22) + (w * x ** 23) + (y * x ** 24) + z

if __name__ == '__main__':
    webbrowser.open('http://127.0.0.1:8050')
    app.run_server(debug=True)
