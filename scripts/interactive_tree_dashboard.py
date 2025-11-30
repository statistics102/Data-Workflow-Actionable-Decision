# interactive_tree_dashboard.py
import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
try:
    df = pd.read_csv('data/cleaned_analysis_data.csv')
    print("✅ Loaded data for interactive tree dashboard")
except:
    from generate_sample_data import create_sample_data
    from complete_analysis_pipeline import DataAnalysisPipeline
    create_sample_data()
    pipeline = DataAnalysisPipeline()
    pipeline.load_data().data_cleaning()
    df = pipeline.cleaned_df

# Prepare features
feature_names = ['age', 'income', 'spending_score', 'credit_score', 'satisfaction']
X = df[feature_names]
y = df['target']

# Train initial tree
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(X_train, y_train)

def create_tree_image(tree_model, feature_names, class_names):
    """Create a matplotlib tree plot and convert to base64 for HTML display"""
    plt.figure(figsize=(20, 10))
    tree.plot_tree(tree_model,
                  feature_names=feature_names,
                  class_names=class_names,
                  filled=True,
                  rounded=True,
                  fontsize=10,
                  proportion=True)
    plt.title('Decision Tree Visualization', fontsize=14, pad=20)
    
    # Convert plot to base64 for HTML display
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    plt.close()  # Close the figure to free memory
    
    encoded = base64.b64encode(image_png).decode('ascii')
    return f"data:image/png;base64,{encoded}"

def create_feature_importance_plot(tree_model, feature_names):
    """Create feature importance plot"""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': tree_model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    fig = px.bar(
        importance_df,
        x='importance',
        y='feature',
        orientation='h',
        title='📊 Feature Importance Ranking',
        color='importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        plot_bgcolor='#2D2D2D',
        paper_bgcolor='#2D2D2D',
        font_color='white',
        title_x=0.5,
        showlegend=False
    )
    
    return fig

def extract_business_rules_text(tree_model, feature_names, class_names=['Class 0', 'Class 1']):
    """Extract business rules as text from decision tree"""
    tree_rules = export_text(tree_model, feature_names=feature_names)
    
    # Parse and format the rules
    rules = []
    lines = tree_rules.split('\n')
    
    for line in lines:
        if 'class:' in line:
            # Extract class information
            class_part = line.split('class:')[-1].strip()
            rules.append(f"→ Predict: {class_names[int(class_part)]}")
        elif '|' in line and any(feature in line for feature in feature_names):
            # This is a condition line
            rules.append(line.replace('|', '│'))
    
    return rules[:15]  # Return top 15 rules

# Create Dash app
app = dash.Dash(__name__, title='Interactive Decision Tree Dashboard')
app.title = "Interactive Decision Tree Analysis"

# Color scheme
colors = {
    'background': '#1E1E1E',
    'text': '#FFFFFF',
    'card_background': '#2D2D2D',
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#4CAF50'
}

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🌳 Interactive Decision Tree Analysis", 
                style={'color': colors['text'], 'marginBottom': '10px'}),
        html.P("Explore Customer Segmentation Rules & Business Intelligence", 
               style={'color': '#CCCCCC', 'fontSize': '16px'}),
    ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': colors['card_background']}),
    
    # Controls Row
    html.Div([
        html.Div([
            html.Label("🌳 Tree Depth:", style={'color': colors['text'], 'marginRight': '10px', 'fontWeight': 'bold'}),
            dcc.Slider(
                id='depth-slider',
                min=2,
                max=5,
                step=1,
                value=3,
                marks={i: f'Depth {i}' for i in range(2, 6)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'width': '45%', 'margin': '10px', 'padding': '10px', 'backgroundColor': colors['card_background'], 'borderRadius': '10px'}),
        
        html.Div([
            html.Label("📊 Min Samples Split:", style={'color': colors['text'], 'marginRight': '10px', 'fontWeight': 'bold'}),
            dcc.Slider(
                id='split-slider',
                min=2,
                max=50,
                step=5,
                value=20,
                marks={i: str(i) for i in range(2, 51, 10)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'width': '45%', 'margin': '10px', 'padding': '10px', 'backgroundColor': colors['card_background'], 'borderRadius': '10px'})
    ], style={'display': 'flex', 'justifyContent': 'center', 'padding': '10px', 'flexWrap': 'wrap'}),
    
    # Feature Selection
    html.Div([
        html.Div([
            html.Label("🎯 Feature Selection:", style={'color': colors['text'], 'marginRight': '10px', 'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='feature-selector',
                options=[{'label': f, 'value': f} for f in feature_names],
                value=feature_names[:3],  # Default to first 3 features
                multi=True,
                style={'width': '100%', 'color': '#000', 'backgroundColor': '#FFF'}
            )
        ], style={'width': '90%', 'margin': '10px', 'padding': '15px', 'backgroundColor': colors['card_background'], 'borderRadius': '10px'})
    ], style={'display': 'flex', 'justifyContent': 'center'}),
    
    # KPI Cards
    html.Div([
        html.Div([
            html.Div([
                html.H4("📈 Accuracy", style={'color': colors['text'], 'margin': '0', 'fontSize': '14px'}),
                html.H3(id='accuracy-value', style={'color': colors['primary'], 'margin': '0', 'fontSize': '24px'})
            ], style={'padding': '15px', 'textAlign': 'center'})
        ], style={'backgroundColor': colors['card_background'], 'borderRadius': '10px', 'flex': '1', 'margin': '5px', 'border': f'2px solid {colors["primary"]}'}),
        
        html.Div([
            html.Div([
                html.H4("🌳 Tree Depth", style={'color': colors['text'], 'margin': '0', 'fontSize': '14px'}),
                html.H3(id='depth-value', style={'color': colors['accent'], 'margin': '0', 'fontSize': '24px'})
            ], style={'padding': '15px', 'textAlign': 'center'})
        ], style={'backgroundColor': colors['card_background'], 'borderRadius': '10px', 'flex': '1', 'margin': '5px', 'border': f'2px solid {colors["accent"]}'}),
        
        html.Div([
            html.Div([
                html.H4("📊 Nodes", style={'color': colors['text'], 'margin': '0', 'fontSize': '14px'}),
                html.H3(id='nodes-value', style={'color': colors['secondary'], 'margin': '0', 'fontSize': '24px'})
            ], style={'padding': '15px', 'textAlign': 'center'})
        ], style={'backgroundColor': colors['card_background'], 'borderRadius': '10px', 'flex': '1', 'margin': '5px', 'border': f'2px solid {colors["secondary"]}'}),
        
        html.Div([
            html.Div([
                html.H4("🎯 Leaves", style={'color': colors['text'], 'margin': '0', 'fontSize': '14px'}),
                html.H3(id='leaves-value', style={'color': colors['success'], 'margin': '0', 'fontSize': '24px'})
            ], style={'padding': '15px', 'textAlign': 'center'})
        ], style={'backgroundColor': colors['card_background'], 'borderRadius': '10px', 'flex': '1', 'margin': '5px', 'border': f'2px solid {colors["success"]}'})
    ], style={'display': 'flex', 'margin': '10px', 'gap': '5px'}),
    
    # Main Content
    html.Div([
        # Left Column - Tree Visualization
        html.Div([
            html.H3("Decision Tree Structure", 
                   style={'color': colors['text'], 'textAlign': 'center', 'marginBottom': '15px'}),
            html.Div([
                html.Img(id='tree-image', 
                        style={'width': '100%', 
                              'border': f'2px solid {colors["primary"]}',
                              'borderRadius': '10px',
                              'backgroundColor': 'white'})
            ], style={'textAlign': 'center'})
        ], style={'width': '60%', 'padding': '15px', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        # Right Column - Analytics
        html.Div([
            # Feature Importance
            html.Div([
                html.H3("Feature Importance", 
                       style={'color': colors['text'], 'textAlign': 'center', 'marginBottom': '15px'}),
                dcc.Graph(id='feature-importance-plot')
            ], style={'marginBottom': '20px'}),
            
            # Business Rules
            html.Div([
                html.H3("Business Rules & Insights", 
                       style={'color': colors['text'], 'textAlign': 'center', 'marginBottom': '15px'}),
                html.Div(id='business-rules', style={
                    'color': colors['text'],
                    'padding': '15px',
                    'backgroundColor': colors['card_background'],
                    'borderRadius': '10px',
                    'maxHeight': '400px',
                    'overflowY': 'auto',
                    'fontFamily': 'monospace',
                    'fontSize': '12px',
                    'border': f'1px solid {colors["accent"]}'
                })
            ])
        ], style={'width': '38%', 'padding': '15px', 'display': 'inline-block', 'verticalAlign': 'top'})
    ]),
    
    # Footer
    html.Div([
        html.P("🌳 Decision Tree Analytics | 🔍 Business Intelligence | 📊 Customer Segmentation", 
               style={'color': '#CCCCCC', 'textAlign': 'center', 'margin': '0', 'padding': '10px'}),
        html.P("Adjust sliders to explore different tree configurations and business rules", 
               style={'color': '#999999', 'textAlign': 'center', 'margin': '0', 'fontSize': '12px'})
    ], style={'padding': '10px', 'backgroundColor': colors['card_background'], 'marginTop': '20px'})
], style={'backgroundColor': colors['background'], 'minHeight': '100vh', 'fontFamily': 'Arial, sans-serif'})

@app.callback(
    [Output('tree-image', 'src'),
     Output('accuracy-value', 'children'),
     Output('depth-value', 'children'),
     Output('nodes-value', 'children'),
     Output('leaves-value', 'children'),
     Output('feature-importance-plot', 'figure'),
     Output('business-rules', 'children')],
    [Input('depth-slider', 'value'),
     Input('split-slider', 'value'),
     Input('feature-selector', 'value')]
)
def update_dashboard(depth, min_split, selected_features):
    try:
        # Validate selected features
        if not selected_features or len(selected_features) < 2:
            selected_features = feature_names[:3]  # Default to first 3 features
        
        # Prepare data with selected features
        X_filtered = X[selected_features]
        X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.2, random_state=42)
        
        # Train decision tree
        tree_model = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_split=min_split,
            random_state=42
        )
        tree_model.fit(X_train, y_train)
        
        # Calculate metrics
        accuracy = tree_model.score(X_test, y_test)
        n_nodes = tree_model.tree_.node_count
        n_leaves = tree_model.tree_.n_leaves
        
        # Create tree visualization
        class_names = ['Class 0', 'Class 1']
        tree_image_src = create_tree_image(tree_model, selected_features, class_names)
        
        # Create feature importance plot
        importance_fig = create_feature_importance_plot(tree_model, selected_features)
        
        # Extract business rules
        rules_text = extract_business_rules_text(tree_model, selected_features, class_names)
        
        # Format business rules for display
        rules_display = []
        for i, rule in enumerate(rules_text):
            if 'Predict' in rule:
                rules_display.append(html.P(rule, style={
                    'color': colors['success'], 
                    'fontWeight': 'bold',
                    'margin': '5px 0',
                    'padding': '5px',
                    'backgroundColor': '#1a1a1a',
                    'borderRadius': '5px'
                }))
            else:
                rules_display.append(html.P(rule, style={
                    'color': colors['text'],
                    'margin': '2px 0',
                    'paddingLeft': '10px'
                }))
        
        if not rules_display:
            rules_display = [html.P("No rules extracted. Try different parameters.", 
                                  style={'color': colors['accent'], 'fontStyle': 'italic'})]
        
        return (tree_image_src, 
                f"{accuracy:.3f}", 
                f"{depth}", 
                f"{n_nodes}", 
                f"{n_leaves}",
                importance_fig,
                rules_display)
                
    except Exception as e:
        print(f"Error in callback: {e}")
        # Return default values in case of error
        return ("", "0.000", "0", "0", "0", go.Figure(), [html.P("Error loading data. Please check parameters.")])

if __name__ == '__main__':
    print("🚀 Starting Interactive Decision Tree Dashboard...")
    print("🌐 Dashboard available at: http://127.0.0.1:8070/")
    print("🎯 Features:")
    print("   - Real-time tree visualization")
    print("   - Interactive parameter controls")
    print("   - Feature importance analysis")
    print("   - Business rules extraction")
    print("   - Professional dark theme")
    
    app.run(debug=False, host='127.0.0.1', port=8070)