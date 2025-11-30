# live_dashboard.py
import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
try:
    df = pd.read_csv('data/cleaned_analysis_data.csv')
    print("✅ Loaded cleaned data for dashboard")
except:
    # Generate sample data if cleaned data doesn't exist
    print("🔄 Generating sample data for dashboard...")
    from generate_sample_data import create_sample_data
    from complete_analysis_pipeline import DataAnalysisPipeline
    
    create_sample_data()
    pipeline = DataAnalysisPipeline()
    pipeline.load_data().data_cleaning()
    df = pipeline.cleaned_df

# Create additional features for enhanced analysis
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], 
                        labels=['18-30', '31-45', '46-60', '60+'])
df['income_segment'] = pd.cut(df['income'], bins=[0, 40000, 70000, 100000, np.inf],
                             labels=['Low', 'Medium', 'High', 'Very High'])

# Initialize Dash app
app = dash.Dash(__name__, title='Customer Analytics Live Dashboard')
app.title = "Live Customer Analytics Dashboard"

# Define color scheme
colors = {
    'background': '#1E1E1E',
    'text': '#FFFFFF',
    'card_background': '#2D2D2D',
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01'
}

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("📊 Live Customer Analytics Dashboard", 
                style={'color': colors['text'], 'marginBottom': '10px'}),
        html.P("Real-time Data Analysis & Business Intelligence", 
               style={'color': '#CCCCCC', 'fontSize': '16px'}),
    ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': colors['card_background']}),
    
    # KPI Cards Row
    html.Div([
        html.Div([
            html.Div([
                html.H3("👥 Total Customers", style={'color': colors['text'], 'margin': '0'}),
                html.H2(f"{len(df):,}", style={'color': colors['primary'], 'margin': '0'})
            ], style={'padding': '20px', 'textAlign': 'center'})
        ], style={'backgroundColor': colors['card_background'], 'borderRadius': '10px', 'flex': '1', 'margin': '10px'}),
        
        html.Div([
            html.Div([
                html.H3("💰 Avg Income", style={'color': colors['text'], 'margin': '0'}),
                html.H2(f"${df['income'].mean():,.0f}", style={'color': colors['accent'], 'margin': '0'})
            ], style={'padding': '20px', 'textAlign': 'center'})
        ], style={'backgroundColor': colors['card_background'], 'borderRadius': '10px', 'flex': '1', 'margin': '10px'}),
        
        html.Div([
            html.Div([
                html.H3("🎯 Target Ratio", style={'color': colors['text'], 'margin': '0'}),
                html.H2(f"{df['target'].value_counts().get(1, 0)/len(df)*100:.1f}%", 
                       style={'color': colors['secondary'], 'margin': '0'})
            ], style={'padding': '20px', 'textAlign': 'center'})
        ], style={'backgroundColor': colors['card_background'], 'borderRadius': '10px', 'flex': '1', 'margin': '10px'}),
        
        html.Div([
            html.Div([
                html.H3("📈 Avg Spending", style={'color': colors['text'], 'margin': '0'}),
                html.H2(f"{df['spending_score'].mean():.0f}", style={'color': '#4CAF50', 'margin': '0'})
            ], style={'padding': '20px', 'textAlign': 'center'})
        ], style={'backgroundColor': colors['card_background'], 'borderRadius': '10px', 'flex': '1', 'margin': '10px'})
    ], style={'display': 'flex', 'margin': '20px', 'gap': '10px'}),
    
    # Filters Row
    html.Div([
        html.Div([
            html.Label("Age Group Filter:", style={'color': colors['text'], 'marginRight': '10px'}),
            dcc.Dropdown(
                id='age-filter',
                options=[{'label': 'All Ages', 'value': 'all'}] + 
                        [{'label': age, 'value': age} for age in df['age_group'].cat.categories],
                value='all',
                style={'width': '200px', 'color': '#000'}
            )
        ], style={'margin': '10px'}),
        
        html.Div([
            html.Label("Income Segment:", style={'color': colors['text'], 'marginRight': '10px'}),
            dcc.Dropdown(
                id='income-filter',
                options=[{'label': 'All Incomes', 'value': 'all'}] + 
                        [{'label': seg, 'value': seg} for seg in df['income_segment'].cat.categories],
                value='all',
                style={'width': '200px', 'color': '#000'}
            )
        ], style={'margin': '10px'}),
        
        html.Div([
            html.Label("Target Class:", style={'color': colors['text'], 'marginRight': '10px'}),
            dcc.Dropdown(
                id='target-filter',
                options=[{'label': 'Both Classes', 'value': 'all'}] + 
                        [{'label': f'Class {i}', 'value': i} for i in [0, 1]],
                value='all',
                style={'width': '200px', 'color': '#000'}
            )
        ], style={'margin': '10px'})
    ], style={'display': 'flex', 'justifyContent': 'center', 'margin': '20px', 'flexWrap': 'wrap'}),
    
    # Main Charts Row 1
    html.Div([
        # Distribution Charts
        html.Div([
            dcc.Graph(id='target-distribution'),
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
        
        html.Div([
            dcc.Graph(id='income-distribution'),
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'})
    ]),
    
    # Main Charts Row 2
    html.Div([
        # Correlation and Scatter
        html.Div([
            dcc.Graph(id='correlation-heatmap'),
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
        
        html.Div([
            dcc.Graph(id='scatter-plot'),
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'})
    ]),
    
    # Main Charts Row 3
    html.Div([
        # Feature Importance and Clustering
        html.Div([
            dcc.Graph(id='feature-importance'),
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
        
        html.Div([
            dcc.Graph(id='clustering-analysis'),
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'})
    ]),
    
    # Real-time Updates Section
    html.Div([
        html.H3("🔄 Live Data Updates", style={'color': colors['text'], 'textAlign': 'center'}),
        html.Div(id='live-updates', style={'color': colors['text'], 'textAlign': 'center', 'padding': '20px'}),
        dcc.Interval(
            id='interval-component',
            interval=10*1000,  # Update every 10 seconds
            n_intervals=0
        )
    ], style={'backgroundColor': colors['card_background'], 'margin': '20px', 'padding': '20px', 'borderRadius': '10px'}),
    
    # Footer
    html.Div([
        html.P("📊 Professional Data Analytics Dashboard | Built with Plotly Dash", 
               style={'color': '#CCCCCC', 'textAlign': 'center', 'margin': '0'}),
        html.P("🔄 Real-time Updates | 🔍 Interactive Filters | 📈 Business Intelligence", 
               style={'color': '#CCCCCC', 'textAlign': 'center', 'margin': '0'})
    ], style={'padding': '20px', 'backgroundColor': colors['card_background'], 'marginTop': '20px'})
], style={'backgroundColor': colors['background'], 'minHeight': '100vh', 'fontFamily': 'Arial, sans-serif'})

# Callback for filtering data
@app.callback(
    [Output('target-distribution', 'figure'),
     Output('income-distribution', 'figure'),
     Output('correlation-heatmap', 'figure'),
     Output('scatter-plot', 'figure'),
     Output('feature-importance', 'figure'),
     Output('clustering-analysis', 'figure'),
     Output('live-updates', 'children')],
    [Input('age-filter', 'value'),
     Input('income-filter', 'value'),
     Input('target-filter', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_dashboard(age_filter, income_filter, target_filter, n_intervals):
    # Filter data based on selections
    filtered_df = df.copy()
    
    if age_filter != 'all':
        filtered_df = filtered_df[filtered_df['age_group'] == age_filter]
    
    if income_filter != 'all':
        filtered_df = filtered_df[filtered_df['income_segment'] == income_filter]
    
    if target_filter != 'all':
        filtered_df = filtered_df[filtered_df['target'] == target_filter]
    
    # 1. Target Distribution Pie Chart
    target_counts = filtered_df['target'].value_counts()
    target_fig = px.pie(
        values=target_counts.values,
        names=[f'Class {i}' for i in target_counts.index],
        title='🎯 Customer Target Distribution',
        color_discrete_sequence=[colors['primary'], colors['secondary']]
    )
    target_fig.update_layout(
        plot_bgcolor=colors['card_background'],
        paper_bgcolor=colors['card_background'],
        font_color=colors['text'],
        title_x=0.5
    )
    
    # 2. Income Distribution by Age Group
    income_fig = px.box(
        filtered_df, 
        x='age_group', 
        y='income', 
        color='target',
        title='💰 Income Distribution by Age Group',
        color_discrete_sequence=[colors['primary'], colors['secondary']]
    )
    income_fig.update_layout(
        plot_bgcolor=colors['card_background'],
        paper_bgcolor=colors['card_background'],
        font_color=colors['text'],
        title_x=0.5
    )
    
    # 3. Correlation Heatmap
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
    corr_matrix = filtered_df[numeric_cols].corr()
    
    correlation_fig = px.imshow(
        corr_matrix,
        title='📊 Feature Correlation Matrix',
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )
    correlation_fig.update_layout(
        plot_bgcolor=colors['card_background'],
        paper_bgcolor=colors['card_background'],
        font_color=colors['text'],
        title_x=0.5
    )
    
    # 4. Interactive Scatter Plot
    scatter_fig = px.scatter(
        filtered_df,
        x='income',
        y='spending_score',
        color='target',
        size='credit_score',
        hover_data=['age', 'satisfaction'],
        title='🎯 Income vs Spending Score (Size: Credit Score)',
        color_discrete_sequence=[colors['primary'], colors['secondary']]
    )
    scatter_fig.update_layout(
        plot_bgcolor=colors['card_background'],
        paper_bgcolor=colors['card_background'],
        font_color=colors['text'],
        title_x=0.5
    )
    
    # 5. Feature Importance
    if len(filtered_df) > 10:  # Only calculate if enough data
        X = filtered_df[['age', 'income', 'spending_score', 'credit_score', 'satisfaction']]
        y = filtered_df['target']
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=True)
        
        feature_fig = px.bar(
            feature_importance,
            x='importance',
            y='feature',
            orientation='h',
            title='🔍 Random Forest Feature Importance',
            color='importance',
            color_continuous_scale='Viridis'
        )
    else:
        feature_fig = go.Figure()
        feature_fig.add_annotation(text="Not enough data for feature importance analysis",
                                 x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    
    feature_fig.update_layout(
        plot_bgcolor=colors['card_background'],
        paper_bgcolor=colors['card_background'],
        font_color=colors['text'],
        title_x=0.5
    )
    
    # 6. Customer Clustering Analysis
    if len(filtered_df) > 10:
        # Use K-means clustering
        cluster_data = filtered_df[['income', 'spending_score', 'age']]
        kmeans = KMeans(n_clusters=3, random_state=42)
        filtered_df['cluster'] = kmeans.fit_predict(cluster_data)
        
        cluster_fig = px.scatter_3d(
            filtered_df,
            x='income',
            y='spending_score',
            z='age',
            color='cluster',
            title='👥 Customer Segmentation (3D Clustering)',
            hover_data=['credit_score', 'satisfaction']
        )
    else:
        cluster_fig = go.Figure()
        cluster_fig.add_annotation(text="Not enough data for clustering analysis",
                                 x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    
    cluster_fig.update_layout(
        plot_bgcolor=colors['card_background'],
        paper_bgcolor=colors['card_background'],
        font_color=colors['text'],
        title_x=0.5
    )
    
    # Live updates text
    from datetime import datetime
    update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    live_updates = [
        html.P(f"🕒 Last Updated: {update_time}", style={'fontSize': '16px', 'margin': '5px'}),
        html.P(f"📊 Displaying: {len(filtered_df):,} customers", style={'fontSize': '16px', 'margin': '5px'}),
        html.P(f"🎯 Target Class Ratio: {filtered_df['target'].mean()*100:.1f}%", style={'fontSize': '16px', 'margin': '5px'})
    ]
    
    return target_fig, income_fig, correlation_fig, scatter_fig, feature_fig, cluster_fig, live_updates

if __name__ == '__main__':
    print("🚀 Starting Live Professional Dashboard...")
    print("📊 Dashboard will be available at: http://127.0.0.1:8050/")
    print("🔄 Dashboard features real-time updates every 10 seconds")
    print("🎯 Use filters to interact with the data")
    app.run(debug=True, host='127.0.0.1', port=8050)