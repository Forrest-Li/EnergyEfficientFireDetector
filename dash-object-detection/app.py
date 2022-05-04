from textwrap import dedent
from time import time, ctime
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_player as player
import dash_daq as daq
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import pathlib

FRAMERATE = 25.0

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Energy Efficient Fire Detector"
server = app.server
app.config.suppress_callback_exceptions = True

BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("data").resolve()


def load_data(path):
    """Load og_data about a specific footage (given by the path). It returns a dictionary of useful variables such as
    the dataframe containing all the detection and bounds localization, the number of classes inside that footage,
    the matrix of all the classes in string, the given class with padding, and the root of the number of classes,
    rounded."""

    # Load the dataframe containing all the processed object detections inside the video
    video_info_df = pd.read_csv(DATA_PATH.joinpath(path))

    # # The list of classes, and the number of classes
    # classes_list = video_info_df["class_str"].value_counts().index.tolist()
    # n_classes = len(classes_list)
    #
    # # Gets the smallest value needed to add to the end of the classes list to get a square matrix
    # root_round = np.ceil(np.sqrt(len(classes_list)))
    # total_size = root_round ** 2
    # padding_value = int(total_size - n_classes)
    # classes_padded = np.pad(classes_list, (0, padding_value), mode="constant")
    #
    # # The padded matrix containing all the classes inside a matrix
    # classes_matrix = np.reshape(classes_padded, (int(root_round), int(root_round)))
    #
    # # Flip it for better looks
    # classes_matrix = np.flip(classes_matrix, axis=0)

    data_dict = {
        "video_info_df": video_info_df,
        # "n_classes": n_classes,
        # "classes_matrix": classes_matrix,
        # "classes_padded": classes_padded,
        # "root_round": root_round,
    }

    if True:
        print(f"{path} loaded.")

    return data_dict


def markdown_popup():
    return html.Div(
        id="markdown",
        className="modal",
        style={"display": "none"},
        children=(
            html.Div(
                className="markdown-container",
                children=[
                    html.Div(
                        className="markdown-text",
                        children=[
                            dcc.Markdown(
                                children=dedent(
                                    """
                                ## Confirmation
                                
                                In event of fires, before notifying the fire units, it is suggested to double-check in 
                                case of a false alarm. To continue, confirm if alerting fire units by calling 911: 
                                
                                > (the function is not implemented yet and can be tested without concerns)
                                
                                """
                                )
                            )
                        ],
                    ),
                    html.Div(
                        className="close-container",
                        children=[
                            html.Button(
                                "Confirm",
                                id="markdown_confirm",
                                n_clicks=0,
                                className="confirmButton",
                            ),
                            html.Button(
                                "Close",
                                id="markdown_close",
                                n_clicks=0,
                                className="closeButton",
                            ),
                        ],
                    ),
                ],
            )
        ),
    )


# Main App
app.layout = html.Div(
    children=[
        dcc.Interval(id="interval-updating-graphs", interval=1000, n_intervals=0),
        html.Div(id="top-bar", className="row"),
        html.Div(
            className="container",
            children=[
                html.Div(
                    id="left-side-column",
                    className="nine columns",
                    children=[
                        html.Div(
                            className="stats-section",
                            children=[
                                html.Div(
                                    id="card-1",
                                    className="stats-element",
                                    children=[
                                        html.H2("Time of day"),
                                        daq.LEDDisplay(
                                            id="LED_HHMM",
                                            value="10:04",
                                            color="#7DCEA0",
                                            backgroundColor="#1e2130",
                                            size=100,
                                        ),
                                        html.Div(
                                            id="MMDDYYYY",
                                            children=["May/03/2022"],
                                            style={
                                                'textAlign': 'center',
                                                'color': "#7DCEA0",
                                                'font-style': 'italic',
                                                'font-size': '30px',
                                            }
                                        ),
                                    ],
                                ),
                                html.Div(
                                    id="card-2",
                                    className="stats-element",
                                    children=[
                                        html.H2("Severity of fire"),
                                        daq.Gauge(
                                            id="progress-gauge",
                                            # color={"gradient": True, "ranges": {"green": [0, 33], "yellow": [33, 67], "darkred": [67, 100]}},
                                            max=100,
                                            min=0,
                                            showCurrentValue=True,  # default size 200 pixel
                                            # size=100,
                                            style={'display': 'block'},
                                            labelPosition='bottom',
                                            label={
                                                'label': '%',
                                                'style': {
                                                    'color': "white",
                                                    'fontSize': 30
                                                }
                                            }
                                        ),
                                    ],
                                ),
                                html.Div(
                                    id="card-3",
                                    className="stats-element",
                                    children=[
                                        html.H2("Area status"),
                                        html.Div(
                                            id="fire-status",
                                            children='FIRE HAPPENING',
                                            style={
                                                'textAlign': 'center',
                                                'color': "#E74C3C",
                                                'font-weight': 'bold',
                                                'font-size': '50px',
                                            }),
                                        html.H2("Alarm fire units"),
                                        html.Div(
                                            id="utility-card",
                                            children=[daq.StopButton(id="stop-button", size=200, n_clicks=0)],
                                        ),
                                    ],
                                ),
                            ]
                        ),
                        html.Div(
                            className="video-outer-container",
                            children=html.Div(
                                className="video-container",
                                children=player.DashPlayer(
                                    id="video-display",
                                    url="https://www.youtube.com/watch?v=gPtn6hD7o8g",
                                    controls=True,
                                    playing=False,
                                    volume=1,
                                    width="100%",
                                    height="100%",
                                ),
                            ),
                        ),
                    ],
                ),
                html.Div(
                    id="right-side-column",
                    className="three columns",
                    children=[
                        html.Div(
                            id="header-section",
                            children=[
                                html.H1("Energy Efficient Fire Detector"),
                                html.P(
                                    "Students: Tianyi (Kenny) Chen, Yuanxi (Forrest) Li; Instructor: Dr. Lino Coria Mendoza"
                                ),
                                html.P(
                                    "Course: CS5330 Pattern Recognition Computer Vision"
                                ),
                                html.P(
                                    "This is a GUI representation of our project presenting a computer vision based "
                                    "method to detect fires at early stages."
                                ),
                                html.P(
                                    "Our approach combines traditional computer vision algorithms and deep learning "
                                    "models that is accurate while being fast & energy efficient."
                                ),
                            ],
                        ),
                        html.Div(
                            className="control-section",
                            children=[
                                html.Div(
                                    className="control-element",
                                    children=[
                                        html.Div(children=["Footage Selection:"]),
                                        dcc.Dropdown(
                                            id="dropdown-footage-selection",
                                            options=[
                                                {
                                                    "label": "Wildfire 1",
                                                    "value": "nest_1",
                                                },
                                                {
                                                    "label": "Wildfire 2",
                                                    "value": "nest_2",
                                                },
                                                {
                                                    "label": "Plane fire",
                                                    "value": "plane",
                                                },
                                            ],
                                            value="nest_1",
                                            clearable=False,
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control-element",
                                    children=[
                                        html.Div(children=["Video Display Mode:"]),
                                        dcc.Dropdown(
                                            id="dropdown-video-display-mode",
                                            options=[
                                                {
                                                    "label": "Raw Video",
                                                    "value": "regular",
                                                },
                                                {
                                                    "label": "Bounding Boxes",
                                                    "value": "bounding_box",
                                                },
                                            ],
                                            value="bounding_box",
                                            searchable=False,
                                            clearable=False,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(id="sep-bar", className="row"),
                        html.Div(
                            className="img-container",
                            children=[
                                html.Img(id="logo-web", src=app.get_asset_url("dash-logo.png")),
                            ]
                        ),
                        html.Div(
                            className="img-container",
                            children=[
                                html.Img(id="logo-neu", src=app.get_asset_url("northeastern.png")),
                            ]
                        ),
                        html.Div(
                            className="imgs-container",
                            children=[
                                html.Div(
                                    className="img-container",
                                    children=[
                                        html.Img(className="logo", src=app.get_asset_url("Tianyi-LinkedIn.png")),
                                    ]
                                ),
                                html.Div(
                                    className="img-container",
                                    children=[
                                        html.Img(className="logo", src=app.get_asset_url("Yuanxi-LinkedIn.png")),
                                    ]
                                ),
                            ]
                        ),
                    ],
                ),
            ],
        ),
        markdown_popup(),
    ]
)


# Data Loading
@app.server.before_first_request
def load_all_footage():
    global data_dict, url_dict

    # Load the dictionary containing all the variables needed for analysis
    data_dict = {
        "nest_1": load_data("nest_1.csv"),
        "nest_2": load_data("nest_2.csv"),
        "plane": load_data("plane.csv"),
    }

    url_dict = {
        "regular": {
            "nest_1": "https://youtu.be/O53Yj4sYkjM",
            "nest_2": "https://youtu.be/bShWt8XbD_4",
            "plane": "https://youtu.be/kGhcrLI3TWk",
        },
        "bounding_box": {
            "nest_1": "https://youtu.be/FySbN7znB10",
            "nest_2": "https://youtu.be/7-3-UPfRKrc",
            "plane": "https://youtu.be/a0oIAF_oUUk",
        },
    }


# Callback for alarming button
@app.callback(
    [
        Output("stop-button", "buttonText"),
        Output("stop-button", "disabled")
    ],
    [
        Input("stop-button", "n_clicks"),
        Input("markdown_confirm", "n_clicks"),
    ],
)
def start_alert(ncStop, ncConfirm):
    if ncConfirm == 0:
        return "Alarm", False
    return "Alerted", True


# Footage Selection
@app.callback(
    Output("video-display", "url"),
    [
        Input("dropdown-footage-selection", "value"),
        Input("dropdown-video-display-mode", "value"),
    ],
)
def select_footage(footage, display_mode):
    global FRAMERATE
    # Find desired footage and update player video
    url = url_dict[display_mode][footage]
    if footage == "plane":
        FRAMERATE = 25
    else:
        FRAMERATE = 30
    return url


# Alarm button popup
@app.callback(
    Output("markdown", "style"),
    [
        Input("stop-button", "n_clicks"),
        Input("markdown_close", "n_clicks"),
        Input("markdown_confirm", "n_clicks"),
    ],
)
def update_click_output(stop_clicks, close_clicks, confirm_clicks):
    ctx = dash.callback_context
    prop_id = ""
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if prop_id == "stop-button":
        return {"display": "block"}
    else:
        return {"display": "none"}


# Time of day update
@app.callback(
    [
        Output("LED_HHMM", "value"),
        Output("MMDDYYYY", "children"),
    ],
    Input("interval-updating-graphs", "n_intervals"),
)
def update_clock(n):
    current_datetime = ctime(time()).split(" ")
    current_time = current_datetime[-2].split(":")
    year = current_datetime[-1]
    month = current_datetime[1]
    day = current_datetime[-3]
    hour = current_time[0]
    minute = current_time[1]
    return f"{hour}:{minute}", \
           f"{month}/{day}/{year}"


# Severity of fire update
@app.callback(
    Output("progress-gauge", "value"),
    [Input("interval-updating-graphs", "n_intervals")],
    [
        State("video-display", "currentTime"),
        State("dropdown-footage-selection", "value"),
    ],
)
def update_dashboard(n, current_time, footage):
    if current_time is not None:
        current_frame = int(current_time * FRAMERATE)

        if n > 0 and current_frame > 0:
            video_info_df = data_dict[footage]["video_info_df"]
            last_frame = video_info_df["frame"].iloc[-1]

            # Select the subset of the dataset that correspond to the current frame
            current_frame = min(current_frame, last_frame)
            frame_df = video_info_df[video_info_df["frame"] == current_frame]

            ratio = frame_df["ratio"].tolist()[0]
            return int(ratio)
    return 0


# Area status update
@app.callback(
    [
        Output("fire-status", "children"),
        Output("fire-status", "style"),
    ],
    [Input("progress-gauge", "value"),],
)
def update_status(value):
    if int(value) == 0:
        return "SAFE", \
               {
                   'textAlign': 'center',
                   'color': "#145A32",
                   'font-weight': 'bold',
                   'font-size': '50px',
               }
    else:
        return "FIRE HAPPENING", \
               {
                   'textAlign': 'center',
                   'color': "#E74C3C",
                   'font-weight': 'bold',
                   'font-size': '50px',
               }

# Running the server
if __name__ == "__main__":
    app.run_server(debug=True, port=8053)
