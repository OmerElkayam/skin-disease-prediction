import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import base64
import io

try:
    model = tf.keras.models.load_model(r"C:\Users\HAMMAD\Desktop\scrape\full_model.h5")
except Exception as e:
    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H1("Error Loading Model", style={'color': 'red', 'text-align': 'center'}),
        html.P(f"Failed to load the model: {str(e)}", style={'text-align': 'center'}),
        html.P("Please check the model file and try again.", style={'text-align': 'center'})
    ], style={'padding': '50px'})
    app.run(debug=True)
    raise SystemExit

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css",
        "https://fonts.googleapis.com/css2?family=Montserrat:wght@700&family=Roboto:wght@400;500&display=swap",
        "/assets/custom.css"
    ],
    suppress_callback_exceptions=True,
    meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}]
)
server = app.server


app.layout = dbc.Container([

    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H1("Skin Cancer Analyzer", className="display-4 mb-1", style={'color': 'white', 'text-align': 'left', 'text-shadow': '0 3px 6px rgba(0, 0, 0, 0.5)'}),
                            html.P("A deep learning-powered tool for preliminary skin lesion analysis designed to assist in early detection of conditions such as benign and malignant tumors. It aids clinicians by providing fast, AI-driven insights to support diagnosis and prioritize care.", 
                                   className="lead text-muted mt-0", 
                                   style={'fontSize': '1.2rem', 'lineHeight': '1.8', 'color': 'rgba(255, 255, 255, 0.98)', 'text-align': 'left', 'fontWeight': '600'})
                        ], md=8, className="d-flex flex-column justify-content-center"),
                        dbc.Col([
                            html.Div([
                                html.Img(
                                    src="/assets/3.png",
                                    className="img-fluid",
                                    style={'borderRadius': '10px'}
                                )
                            ], className="header-right-animation")
                        ], md=4, className="position-relative")
                    ]),
                    html.Hr(className="my-4", style={'borderColor': 'rgba(255, 255, 255, 0.2)'}),
                ])
            ], className="header-card")
        ], width=12)
    ], className="mb-5"),

    
    dbc.Tabs([
        
        dbc.Tab(
            label="Image Upload",
            children=[
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Upload Skin Image", className="card-title mb-4", style={'color': 'var(--dark)'}),
                        html.P("Select a clear image of your skin lesion for analysis", 
                               className="card-text text-muted mb-4"),
                        
                        
                        dbc.Row([
                            dbc.Col([
                                html.Label("Choose an image file:", className="form-label", style={'color': 'var(--dark)'}),
                                dcc.Upload(
                                    id='upload-image',
                                    children=dbc.Button([
                                        html.I(className="fas fa-cloud-upload-alt me-2"),
                                        "Select Image File"
                                    ], color="primary", className="w-100"),
                                    multiple=False
                                ),
                                html.Div(id='output-filename', className="mt-2 text-muted")
                            ], width=6),
                            
                            dbc.Col([
                                html.Div(id='upload-preview', className="text-center")
                            ], width=6)
                        ], className="mb-4"),
                        
                        html.Div(id='upload-loading', children=[
                            dcc.Loading(type="circle", children="Analyzing...", color="#2196F3")
                        ], style={'display': 'none'}),
                        html.Div(id='upload-prediction-result')
                    ])
                ], className="upload-card")
            ],
            tabClassName="fw-bold"
        ),
        
        
        dbc.Tab(
            label="Webcam Analysis",
            children=[
                dbc.Card(
                    dbc.CardBody([
                        html.H4("Live Skin Analysis", className="card-title mb-4", style={'color': 'var(--dark)'}),
                        html.P("Position your skin lesion in front of your camera and click capture", 
                               className="card-text text-muted mb-4"),
                        
                        
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.Video(
                                        id="video-element",
                                        autoPlay=True,
                                        muted=True,
                                        className="rounded border",
                                        style={
                                            'width': '100%',
                                            'maxWidth': '500px',
                                            'backgroundColor': '#E8ECEF'
                                        }
                                    ),
                                    html.Div(id="webcam-error", className="text-danger text-center mt-2")
                                ], className="text-center mb-3"),
                                
                                dbc.Button([
                                    html.I(className="fas fa-camera me-2"),
                                    "Capture Image"
                                ], id="capture-btn", color="primary", className="mb-3"),
                                
                                
                                html.Canvas(id="canvas", style={"display": 'none'}),
                                dcc.Store(id="captured-image-store"),
                            ], md=6),
                            
                            dbc.Col([
                                html.Div([
                                    html.H5("Captured Image:", className="mb-3", style={'color': 'var(--dark)'}),
                                    html.Img(
                                        id="captured-image",
                                        className="img-thumbnail",
                                        style={
                                            'maxWidth': '100%',
                                            'display': 'none',
                                            'borderRadius': '10px',
                                            'borderColor': '#2196F3'
                                        }
                                    ),
                                ], id="webcam-preview", className="text-center")
                            ], md=6)
                        ]),
                        
                        html.Div(id='webcam-loading', children=[
                            dcc.Loading(type="circle", children="Analyzing...", color="#2196F3")
                        ], style={'display': 'none'}),
                        html.Div(id='webcam-prediction-result', className="mt-4")
                    ]), className="webcam-card"
                )
            ],
            tabClassName="fw-bold"
        ),
        
        
        dbc.Tab(
            label="Learn About Skin Cancer",
            children=[
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Types of Skin Cancer", className="card-title mb-4", style={'color': 'var(--dark)'}),
                        html.P("Learn about common types of Skin Cancer.", 
                               className="card-text text-muted mb-4"),
                        
                        
                        dbc.Row([
                            
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardImg(
                                        src="/assets/benign.jpg",
                                        top=True,
                                        className="img-fluid",
                                        style={'borderRadius': '10px 10px 0 0'},
                                        alt="Benign Skin Lesion"
                                    ),
                                    dbc.CardBody([
                                        html.H5("Benign Skin Lesion", className="card-title", style={'color': 'var(--dark)'}),
                                        html.P(
                                            "Non-cancerous growths that usually do not spread or pose a serious health risk"
                                            "Common types include moles, seborrheic keratosis, and skin tags.",
                                            className="card-text text-muted"
                                        ),
                                        html.Ul([
                                            html.Li("Typically uniform in color and shape"),
                                            html.Li("May be flat or raised"),
                                            html.Li("Generally slow-growing"),
                                            html.Li("Often soft or smooth to the touch")
                                        ], className="text-muted")
                                    ])
                                ], className="learn-card mb-4")
                            ], md=4),
                            
                            
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardImg(
                                        src="/assets/malignant.jpg",
                                        top=True,
                                        className="img-fluid",
                                        style={'borderRadius': '10px 10px 0 0'},
                                        alt="Malignant Skin Lesion example"
                                    ),
                                    dbc.CardBody([
                                        html.H5("Malignant Skin Lesion", className="card-title", style={'color': 'var(--dark)'}),
                                        html.P(
                                            "Cancerous growth that can spread to other parts of the body if left untreated."
                                            "Includes Melanoma and Others.",
                                            className="card-text text-muted"
                                        ),
                                        html.Ul([
                                            html.Li("Irregular borders and uneven coloring"),
                                            html.Li("May change in size, shape, or color over time"),
                                            html.Li("Can itch, bleed, or become ulcerated"),
                                            html.Li("Often found in sun-exposed areas but can appear anywhere")
                                        ], className="text-muted")
                                    ])
                                ], className="learn-card mb-4")
                            ], md=4),
                            
                            
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardImg(
                                        src="/assets/Melanoma.jpg",
                                        top=True,
                                        className="img-fluid",
                                        style={'borderRadius': '10px 10px 0 0'},
                                        alt="Melanoma example"
                                    ),
                                    dbc.CardBody([
                                        html.H5("Melanoma", className="card-title", style={'color': 'var(--dark)'}),
                                        html.P(
                                            "A serious and potentially life-threatening form of skin cancer that develops from melanocytes.",
                                            className="card-text text-muted"
                                        ),
                                        html.Ul([
                                            html.Li("Often appears as a new mole or a change in an existing one"),
                                            html.Li("May grow rapidly, itch, or bleed"),
                                            html.Li("Common on the back, legs, arms, and face"),
                                            html.Li("Can spread to other parts of the body if not treated early")
                                        ], className="text-muted")
                                    ])
                                ], className="learn-card mb-4")
                            ], md=4)
                        ])
                    ])
                ], className="learn-tab-card")
            ],
            tabClassName="fw-bold"
        )
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.Footer([
                html.Hr(className="footer-divider"),
                html.Div([
                    html.P([
                        "Disclaimer: This tool provides preliminary information Only. ",
                    ], className="footer-text"),
                    html.P("© 2025 Deep Learning Skin Lesion Classifier", className="footer-copyright")
                ], className="footer-content")
            ], className="footer")
        ], width=12)
    ], className="mt-5")

], fluid=True, className="py-4", style={'backgroundColor': '#F4F7F9'})


@app.callback(
    [Output('output-filename', 'children'),
     Output('upload-preview', 'children')],
    Input('upload-image', 'contents'),
    State('upload-image', 'filename')
)
def update_upload_preview(contents, filename):
    print("Upload callback triggered")
    if contents is None:
        return ["No file selected", html.Div([
            html.I(className="fas fa-image fa-5x text-muted"),
            html.P("Preview will appear here", className="text-muted mt-2")
        ])]
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    img = Image.open(io.BytesIO(decoded))
    
    return [
        f"Selected: {filename}",
        html.Img(
            src=contents,
            className="img-thumbnail",
            style={'maxWidth': '100%', 'maxHeight': '300px', 'borderRadius': '10px', 'borderColor': '#2196F3'}
        )
    ]


app.clientside_callback(
    """
    function(id) {
        console.log("Webcam initialization triggered for video-element ID:", id);
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: true })
            .then(function(stream) {
                var video = document.getElementById('video-element');
                if (video) {
                    video.srcObject = stream;
                    video.play().catch(function(err) {
                        console.error("Video playback error:", err);
                    });
                    console.log("Webcam stream started successfully");
                } else {
                    console.error("Video element not found");
                }
            })
            .catch(function(err) {
                console.error("Error accessing webcam:", err);
                var errorDiv = document.getElementById('webcam-error');
                if (errorDiv) {
                    errorDiv.innerText = "Could not access webcam. Please ensure you have granted camera permissions.";
                }
            });
        } else {
            console.error("Webcam access not supported in this browser");
            var errorDiv = document.getElementById('webcam-error');
            if (errorDiv) {
                errorDiv.innerText = "Webcam access not supported in this browser.";
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('video-element', 'id'),
    Input('video-element', 'id')
)


app.clientside_callback(
    """
    function(n_clicks) {
        console.log("Capture button clicked:", n_clicks);
        if (n_clicks > 0) {
            var video = document.getElementById('video-element');
            var canvas = document.getElementById('canvas');
            var context = canvas.getContext('2d');
            
            if (!video || !canvas) {
                console.error("Video or canvas element not found");
                return window.dash_clientside.no_update;
            }
            
            // Set canvas dimensions to match video
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            // Draw current video frame to canvas
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            // Get image data URL
            var dataURL = canvas.toDataURL('image/jpeg');
            
            // Display captured image
            var imgElement = document.getElementById('captured-image');
            if (imgElement) {
                imgElement.src = dataURL;
                imgElement.style.display = 'block';
                console.log("Captured image set successfully");
            } else {
                console.error("Captured image element not found");
            }
            
            return dataURL;
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('captured-image-store', 'data'),
    Input('capture-btn', 'n_clicks')
)


@app.callback(
    Output('captured-image', 'style'),
    Input('captured-image-store', 'data')
)
def show_captured_image(image_data):
    print("Show captured image callback triggered")
    if image_data:
        return {
            'maxWidth': '100%',
            'display': 'block',
            'border': '2px solid #2196F3',
            'borderRadius': '10px'
        }
    return {'display': 'none'}


@app.callback(
    [Output('upload-prediction-result', 'children'),
     Output('webcam-prediction-result', 'children'),
     Output('upload-loading', 'style'),
     Output('webcam-loading', 'style')],
    [Input('upload-image', 'contents'),
     Input('captured-image-store', 'data')],
    [State('upload-image', 'filename')]
)
def predict_image(upload_contents, webcam_data, upload_filename):
    print("Prediction callback triggered")
    ctx = callback_context
    
    if not ctx.triggered:
        print("No callback triggered")
        return [None, None, {'display': 'none'}, {'display': 'none'}]
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    print(f"Triggered ID: {triggered_id}")
    
    
    if triggered_id == 'upload-image':
        loading_styles = [{'display': 'block'}, {'display': 'none'}]
    else:
        loading_styles = [{'display': 'none'}, {'display': 'block'}]
    
    try:
        if triggered_id == 'upload-image' and upload_contents:
            print("Processing uploaded image")
            content_type, content_string = upload_contents.split(',')
            decoded = base64.b64decode(content_string)
            img = Image.open(io.BytesIO(decoded)).convert('RGB')
            source = "Uploaded Image"
            
        elif triggered_id == 'captured-image-store' and webcam_data:
            print("Processing webcam capture")
            content_string = webcam_data.split(',')[1]
            decoded = base64.b64decode(content_string)
            img = Image.open(io.BytesIO(decoded)).convert('RGB')
            source = "Webcam Capture"
        else:
            print("No valid image data")
            return [None, None, {'display': 'none'}, {'display': 'none'}]
        
        
        img = img.resize((128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        
        prediction = model.predict(img_array)[0][0]
        is_malignant = prediction > 0.5
        confidence = prediction * 100 if is_malignant else (1 - prediction) * 100
        print(f"Prediction: {prediction}, Malignant: {is_malignant}, Confidence: {confidence}")
        
        
        result_card = dbc.Card([
            dbc.CardHeader([
                html.H4("Analysis Results", className="mb-0", style={'color': 'var(--dark)'}),
                html.Small(source, className="text-muted")
            ], className="bg-light"),
            
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H2(
                                "🦠 Malignant" if is_malignant else "✅ Benign",
                                className='text-danger' if is_malignant else 'text-success'
                            ),
                            html.P(f"Confidence: {confidence:.1f}%", className="text-muted")
                        ], className="text-center mb-3")
                    ], md=4),
                    
                    dbc.Col([
                        dbc.Progress(
                            value=confidence,
                            striped=True,
                            animated=is_malignant,
                            color="danger" if is_malignant else "success",
                            style={"height": "20px"},
                            className="mb-3"
                        ),
                        ])
                    ],
                ),
                
                dbc.Alert([
                    html.I(className="fas fa-info-circle me-2"),
                    "This analysis is for informational purposes only."
                ], color="info", className="mt-3")
            ])
        ], className="result-card mt-4")
        
        
        if triggered_id == 'upload-image':
            return [result_card, None, {'display': 'none'}, {'display': 'none'}]
        else:
            return [None, result_card, {'display': 'none'}, {'display': 'none'}]
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        error_alert = dbc.Alert([
            html.H4("Error processing image", className="alert-heading", style={'color': 'var(--dark)'}),
            html.P(str(e), className="mb-0")
        ], color="danger")
        
        if triggered_id == 'upload-image':
            return [error_alert, None, {'display': 'none'}, {'display': 'none'}]
        else:
            return [None, error_alert, {'display': 'none'}, {'display': 'none'}]

if __name__ == '__main__':
    app.run(debug=True)