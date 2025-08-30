import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
import io
import base64
import os
import pickle
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Project Tracking Success Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .success-card {
        background: linear-gradient(135deg, #28a745, #20c997);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 8px 25px rgba(40, 167, 69, 0.3);
        transition: transform 0.3s ease;
    }
    
    .success-card:hover {
        transform: translateY(-5px);
    }
    
    .failure-card {
        background: linear-gradient(135deg, #dc3545, #fd7e14);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 8px 25px rgba(220, 53, 69, 0.3);
        transition: transform 0.3s ease;
    }
    
    .failure-card:hover {
        transform: translateY(-5px);
    }
    
    .medium-risk-card {
        background: linear-gradient(135deg, #ffc107, #fd7e14);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 8px 25px rgba(255, 193, 7, 0.3);
        transition: transform 0.3s ease;
    }
    
    .medium-risk-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #007bff;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .industry-card {
        background: linear-gradient(135deg, #6f42c1, #e83e8c);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 15px 0;
        box-shadow: 0 8px 25px rgba(111, 66, 193, 0.3);
    }
    
    .chart-container {
        background: white;
        border-radius: 15px;
        padding: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 10px;
        color: #000;
        font-weight: bold;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #007bff;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class StartupSuccessPredictor:
    def __init__(self, model_path="Project_Tracking_model.pkl"):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = model_path
        self.model_loaded = False
        self.feature_names = [
            'Industry_of_Project', 'Employee_Count', 'team_size_grown',
            'Number_of_Co_founders', 'Team_size_Senior_leadership',
            'customer_satisfaction_score', 'scalability_score', 'churn_rate',
            'burn_rate', 'customer_acquisition_cost', 'monthly_recurring_revenue',
            'business_sustainability_score', 'User_Adoption_Rate',
            'Revenue_Growth', 'Market_Competition'
        ]
        self.success_benchmarks = {
            'customer_satisfaction_score': 8.0,
            'churn_rate': 0.10,
            'scalability_score': 80.0,
            'team_size_grown': 2,
            'monthly_recurring_revenue': 50000,
            'burn_rate': 10.0,
            'User_Adoption_Rate': 0.65,
            'Revenue_Growth': 15.0,
            'Employee_Count': 30,
            'business_sustainability_score': 80.0,
            'Number_of_Co_founders': 2,
            'Team_size_Senior_leadership': 5,
            'customer_acquisition_cost': 100,
            'Market_Competition': 3
        }
        self.industry_success_rates = {
            'Technology': 0.72,
            'Healthcare': 0.68,
            'E-Commerce': 0.58,
            'Analytics': 0.65,
            'Marketing': 0.52,
            'Cloud Computing': 0.75,
            'Mobile': 0.48,
            'Finance': 0.62,
            'Food & Beverages': 0.45,
            'Education': 0.55,
            'Real Estate': 0.50
        }
        self.load_model()
    
    def _try_load_pickle(self, filepath):
        """Helper function to safely load a pickle file with error handling"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                # Verify the loaded data has the expected structure
                if isinstance(data, dict) and 'model' in data:
                    return data
                elif hasattr(data, 'predict'):  # It's a model object directly
                    return {'model': data}
                else:
                    return None
        except Exception:
            return None
    
    def load_model(self):
        """Load the pre-trained model from pickle file with silent error handling"""
        # Try to load the model file if it exists
        if os.path.exists(self.model_path):
            model_data = self._try_load_pickle(self.model_path)
            
            if model_data and 'model' in model_data and hasattr(model_data['model'], 'predict'):
                self.model = model_data['model']
                self.scaler = model_data.get('scaler')
                
        
        # If we couldn't load the model, create a fallback one
        if not hasattr(self, 'model') or self.model is None:
            self._create_fallback_model()
        else:
            # Initialize scaler if not provided
            if not hasattr(self, 'scaler') or self.scaler is None:
                self.scaler = StandardScaler()
                self._fit_default_scaler()
            
            # Verify the model has the required methods
            if not (hasattr(self.model, 'predict') and hasattr(self.model, 'predict_proba')):
                self._create_fallback_model()
            else:
                self.model_loaded = True
    
    def _fit_default_scaler(self):
        """Fit scaler with default parameters if not provided"""
        # Create dummy data with realistic ranges for fitting the scaler
        dummy_data = {
            'Industry_of_Project': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            'Employee_Count': [10, 20, 30, 50, 100, 200, 300],
            'team_size_grown': [0, 1, 2, 3, 5],
            'Number_of_Co_founders': [1, 2, 3, 4],
            'Team_size_Senior_leadership': [2, 3, 5, 8, 12],
            'customer_satisfaction_score': [5, 6, 7, 8, 9, 10],
            'scalability_score': [40, 50, 60, 70, 80, 90],
            'churn_rate': [0.05, 0.1, 0.15, 0.2, 0.3],
            'burn_rate': [5, 10, 15, 20, 25],
            'customer_acquisition_cost': [50, 100, 150, 200, 300],
            'monthly_recurring_revenue': [10000, 30000, 50000, 100000, 200000],
            'business_sustainability_score': [50, 60, 70, 80, 90],
            'User_Adoption_Rate': [0.3, 0.5, 0.6, 0.7, 0.8],
            'Revenue_Growth': [-10, 0, 10, 20, 30],
            'Market_Competition': [1, 2, 3, 4, 5]
        }
        
        # Create dummy dataframe and fit scaler
        max_len = max(len(v) for v in dummy_data.values())
        for key in dummy_data:
            dummy_data[key] = (dummy_data[key] * (max_len // len(dummy_data[key]) + 1))[:max_len]
        
        dummy_df = pd.DataFrame(dummy_data)
        self.scaler.fit(dummy_df[self.feature_names])
    
    def _create_fallback_model(self):
        """Create a fallback model if pickle loading fails"""
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        )
        
        # Create training data for fallback model
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'Industry_of_Project': np.random.randint(1, 12, n_samples),
            'Employee_Count': np.random.lognormal(3, 0.8, n_samples).astype(int),
            'team_size_grown': np.random.poisson(2, n_samples),
            'Number_of_Co_founders': np.random.choice([1, 2, 3, 4], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
            'Team_size_Senior_leadership': np.random.randint(2, 15, n_samples),
            'customer_satisfaction_score': np.random.beta(2, 1, n_samples) * 7 + 3,
            'scalability_score': np.random.beta(2, 2, n_samples) * 70 + 30,
            'churn_rate': np.random.exponential(0.15, n_samples),
            'burn_rate': np.random.lognormal(2, 0.5, n_samples),
            'customer_acquisition_cost': np.random.lognormal(4, 0.6, n_samples),
            'monthly_recurring_revenue': np.random.lognormal(10, 1, n_samples),
            'business_sustainability_score': np.random.beta(2, 1, n_samples) * 60 + 40,
            'User_Adoption_Rate': np.random.beta(2, 2, n_samples),
            'Revenue_Growth': np.random.normal(10, 15, n_samples),
            'Market_Competition': np.random.randint(1, 6, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create realistic bounds
        df['Employee_Count'] = np.clip(df['Employee_Count'], 5, 500)
        df['team_size_grown'] = np.clip(df['team_size_grown'], 0, 10)
        df['churn_rate'] = np.clip(df['churn_rate'], 0.01, 0.8)
        df['burn_rate'] = np.clip(df['burn_rate'], 1, 50)
        df['customer_acquisition_cost'] = np.clip(df['customer_acquisition_cost'], 10, 500)
        df['monthly_recurring_revenue'] = np.clip(df['monthly_recurring_revenue'], 1000, 200000)
        df['Revenue_Growth'] = np.clip(df['Revenue_Growth'], -30, 100)
        
        # Create target variable
        success_score = (
            (df['customer_satisfaction_score'] > 7) * 15 +
            (df['churn_rate'] < 0.15) * 12 +
            (df['scalability_score'] > 70) * 10 +
            (df['Employee_Count'] > 20) * 8 +
            (df['team_size_grown'] > 1) * 6 +
            (df['burn_rate'] < 12) * 8 +
            (df['Revenue_Growth'] > 5) * 10 +
            (df['User_Adoption_Rate'] > 0.5) * 8 +
            (df['business_sustainability_score'] > 70) * 7 +
            (df['monthly_recurring_revenue'] > 30000) * 10 +
            np.random.normal(0, 8, n_samples)
        )
        
        y = (success_score > 50).astype(int)
        
        # Fit scaler and model
        X_scaled = self.scaler.fit_transform(df[self.feature_names])
        self.model.fit(X_scaled, y)
        self.model_loaded = True
    
    def predict(self, input_data):
        """Make prediction using the loaded model"""
        if not hasattr(self, 'model_loaded') or not self.model_loaded:
            return {
                'prediction': 'Error',
                'success_probability': 0.0,
                'error': 'Model not loaded'
            }
        
        try:
            # Convert input to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Ensure all required features are present
            for feature in self.feature_names:
                if feature not in input_df.columns:
                    input_df[feature] = 0  # Default value
            
            # Scale the input
            input_scaled = self.scaler.transform(input_df[self.feature_names])
            
            # Make prediction
            prediction = self.model.predict(input_scaled)[0]
            probability = self.model.predict_proba(input_scaled)[0][1]
            
            return {
                'prediction': 'Success' if prediction == 1 else 'Failure',
                'success_probability': probability
            }
            
        except Exception as e:
            return {
                'prediction': 'Error',
                'success_probability': 0.0,
                'error': str(e)
            }
    
    def analyze_features(self, input_data):
        analysis = []
        for feature, value in input_data.items():
            if feature in self.success_benchmarks:
                benchmark = self.success_benchmarks[feature]
                
                # Determine status based on feature type
                if feature in ['churn_rate', 'burn_rate', 'customer_acquisition_cost', 'Market_Competition']:
                    # Lower is better
                    if value <= benchmark * 0.7:
                        status = "Excellent"
                    elif value <= benchmark:
                        status = "Good"
                    elif value <= benchmark * 1.3:
                        status = "Needs Improvement"
                    else:
                        status = "Critical"
                else:
                    # Higher is better
                    if value >= benchmark * 1.2:
                        status = "Excellent"
                    elif value >= benchmark:
                        status = "Good"
                    elif value >= benchmark * 0.7:
                        status = "Needs Improvement"
                    else:
                        status = "Critical"
                
                # Feature importance mapping
                importance_map = {
                    'customer_satisfaction_score': 'High',
                    'churn_rate': 'High',
                    'scalability_score': 'High',
                    'monthly_recurring_revenue': 'High',
                    'burn_rate': 'Medium',
                    'User_Adoption_Rate': 'Medium',
                    'Revenue_Growth': 'High',
                    'Employee_Count': 'Medium',
                    'business_sustainability_score': 'Medium',
                    'team_size_grown': 'Low',
                    'Number_of_Co_founders': 'Low',
                    'Team_size_Senior_leadership': 'Medium',
                    'customer_acquisition_cost': 'Medium',
                    'Market_Competition': 'Low'
                }
                
                analysis.append({
                    'Feature': feature.replace('_', ' ').title(),
                    'Your Value': round(value, 2) if isinstance(value, float) else value,
                    'Success Benchmark': benchmark,
                    'Status': status,
                    'Impact': importance_map.get(feature, 'Low'),
                    'Gap': round(value - benchmark, 2) if isinstance(value, (int, float)) else 0
                })
        
        return pd.DataFrame(analysis)

@st.cache_data
def create_success_probability_gauge(probability):
    """Enhanced gauge chart for success probability"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "<b>Success Probability (%)</b>", 'font': {'size': 20}},
        delta = {
            'reference': 50, 
            'increasing': {'color': "green"}, 
            'decreasing': {'color': "red"},
            'font': {'size': 16}
        },
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue", 'thickness': 0.8},
            'steps': [
                {'range': [0, 25], 'color': "#ffcccc"},
                {'range': [25, 50], 'color': "#fff3cd"},
                {'range': [50, 75], 'color': "#d4edda"},
                {'range': [75, 100], 'color': "#28a745"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            },
            'borderwidth': 2,
            'bordercolor': "gray"
        }
    ))
    
    fig.update_layout(
        height=400,
        font={'color': "darkblue", 'family': "Arial"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_feature_comparison_chart(analysis_df):
    """Create histogram comparison of different attributes"""
    # Define weights and scores
    impact_weights = {'High': 3, 'Medium': 2, 'Low': 1}
    status_weights = {'Critical': 1, 'Needs Improvement': 2, 'Good': 3, 'Excellent': 4}
    
    # Prepare the data
    df = analysis_df.copy()
    df['Impact_Score'] = df['Impact'].map(impact_weights)
    df['Status_Score'] = df['Status'].map(status_weights)
    df['Priority_Score'] = df['Impact_Score'] * (5 - df['Status_Score'])
    
    # Select top features by priority score
    top_features = df.nlargest(10, 'Priority_Score')
    
    # Create figure with histogram subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Value Distribution',
            'Impact Distribution',
            'Status Distribution',
            'Priority Distribution'
        ),
        vertical_spacing=0.2,
        horizontal_spacing=0.1
    )
    
    # Color scheme
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
    
    # 1. Your Values Distribution
    fig.add_trace(go.Histogram(
        x=top_features['Your Value'],
        name='Your Values',
        marker_color=colors[0],
        opacity=0.8,
        nbinsx=8,
        hovertemplate='Value: %{x:.1f}<br>Count: %{y}<extra></extra>',
        showlegend=False
    ), row=1, col=1)
    
    # Add benchmark line
    avg_benchmark = top_features['Success Benchmark'].mean()
    fig.add_vline(
        x=avg_benchmark,
        line=dict(color=colors[1], width=2, dash='dash'),
        row=1, col=1,
        annotation_text=f"Avg Benchmark: {avg_benchmark:.1f}",
        annotation_position='top right'
    )
    
    # 2. Impact Distribution
    impact_counts = top_features['Impact'].value_counts().reindex(['Low', 'Medium', 'High']).fillna(0)
    fig.add_trace(go.Bar(
        x=impact_counts.index,
        y=impact_counts.values,
        name='Impact',
        marker_color=[colors[impact_weights[imp]-1] for imp in impact_counts.index],
        opacity=0.8,
        hovertemplate='Impact: %{x}<br>Count: %{y}<extra></extra>',
        showlegend=False
    ), row=1, col=2)
    
    # 3. Status Distribution
    status_counts = top_features['Status'].value_counts().reindex(['Critical', 'Needs Improvement', 'Good', 'Excellent']).fillna(0)
    fig.add_trace(go.Bar(
        x=status_counts.index,
        y=status_counts.values,
        name='Status',
        marker_color=[colors[i % len(colors)] for i in range(len(status_counts))],
        opacity=0.8,
        hovertemplate='Status: %{x}<br>Count: %{y}<extra></extra>',
        showlegend=False
    ), row=2, col=1)
    
    # 4. Priority Score Distribution
    fig.add_trace(go.Histogram(
        x=top_features['Priority_Score'],
        name='Priority',
        marker_color=colors[4],
        opacity=0.8,
        nbinsx=8,
        hovertemplate='Priority: %{x:.1f}<br>Count: %{y}<extra></extra>',
        showlegend=False
    ), row=2, col=2)
    
    # Add mean priority line
    mean_priority = top_features['Priority_Score'].mean()
    fig.add_vline(
        x=mean_priority,
        line=dict(color='#333', width=2, dash='dash'),
        row=2, col=2,
        annotation_text=f"Mean: {mean_priority:.1f}",
        annotation_position='top right'
    )
    
    # Add benchmark distribution as overlay in the first plot
    fig.add_trace(go.Histogram(
        x=top_features['Success Benchmark'],
        name='Benchmark',
        marker_color=colors[1],
        opacity=0.5,
        nbinsx=8,
        hovertemplate='Benchmark: %{x:.1f}<br>Count: %{y}<extra></extra>',
        showlegend=True
    ), row=1, col=1)
    
    # Update layout
    fig.update_layout(
        title='<b>Attribute Distribution Analysis</b>',
        height=700,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        template='plotly_white',
        margin=dict(l=50, r=50, b=80, t=80, pad=4),
        hoverlabel=dict(
            bgcolor='white',
            font_size=12,
            font_family='Arial'
        ),
        barmode='overlay'
    )
    
    # Update subplot titles and axes
    fig.update_xaxes(title_text='Your Value', row=1, col=1)
    fig.update_yaxes(title_text='Count', row=1, col=1)
    
    fig.update_xaxes(title_text='Impact Level', row=1, col=2)
    fig.update_yaxes(title_text='Count', row=1, col=2)
    
    fig.update_xaxes(title_text='Status', row=2, col=1)
    fig.update_yaxes(title_text='Count', row=2, col=1)
    
    fig.update_xaxes(title_text='Priority Score', row=2, col=2)
    fig.update_yaxes(title_text='Count', row=2, col=2)
    
    # Rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=-45)
    
    # Add gap between subplots and title
    fig.update_layout(margin=dict(t=100))
    
    return fig

def create_radar_chart(input_data):
    """Enhanced radar chart for key metrics"""
    key_metrics = [
        ('customer_satisfaction_score', 'Customer Satisfaction', 10),
        ('scalability_score', 'Scalability', 100),
        ('business_sustainability_score', 'Business Sustainability', 100),
        ('User_Adoption_Rate', 'User Adoption', 1),
        ('Revenue_Growth', 'Revenue Growth', 50)
    ]
    
    normalized_values = []
    metric_names = []
    
    for metric, display_name, max_val in key_metrics:
        if metric in input_data:
            # Normalize to 0-10 scale
            if metric == 'Revenue_Growth':
                # Handle negative values for revenue growth
                normalized_val = max(0, min(10, (input_data[metric] + 10) / 6))
            else:
                normalized_val = (input_data[metric] / max_val) * 10
            
            normalized_values.append(normalized_val)
            metric_names.append(display_name)
    
    # Close the radar chart
    normalized_values += normalized_values[:1]
    metric_names += metric_names[:1]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=normalized_values,
        theta=metric_names,
        fill='toself',
        name='Your Domain',
        line=dict(color='rgb(1,90,200)', width=3),
        fillcolor='rgba(1,90,200,0.25)'
    ))
    
    # Add benchmark line
    benchmark_values = [8, 8, 8, 6.5, 7] + [8]  # Normalized benchmark values
    fig.add_trace(go.Scatterpolar(
        r=benchmark_values,
        theta=metric_names,
        fill=None,
        name='Success Benchmark',
        line=dict(color='green', width=2, dash='dash')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickvals=[0, 2, 4, 6, 8, 10],
                ticktext=['0', '2', '4', '6', '8', '10']
            )),
        showlegend=True,
        title="<b>Project Performance Radar</b>",
        height=500,
        legend=dict(x=0.8, y=1)
    )
    
    return fig

def create_risk_assessment_donut(analysis_df):
    """Enhanced donut chart for risk assessment"""
    status_counts = analysis_df['Status'].value_counts()
    
    colors = {
        'Excellent': '#28a745',
        'Good': '#20c997', 
        'Needs Improvement': '#ffc107',
        'Critical': '#dc3545'
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=status_counts.index,
        values=status_counts.values,
        hole=0.6,
        marker=dict(
            colors=[colors.get(status, '#6c757d') for status in status_counts.index],
            line=dict(color='#FFFFFF', width=3)
        ),
        textinfo='label+percent',
        textposition='outside',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    # Calculate overall risk score
    risk_score = (
        status_counts.get('Excellent', 0) * 4 +
        status_counts.get('Good', 0) * 3 +
        status_counts.get('Needs Improvement', 0) * 2 +
        status_counts.get('Critical', 0) * 1
    ) / len(analysis_df)
    
    risk_text = f"Risk Score<br><b>{risk_score:.1f}/4.0</b>"
    
    fig.update_layout(
        title="<b>Feature Status Distribution</b>",
        annotations=[dict(text=risk_text, x=0.5, y=0.5, font_size=16, showarrow=False)],
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    return fig

def create_industry_comparison_chart(selected_industry, industry_success_rates):
    """Enhanced industry comparison chart"""
    industries = list(industry_success_rates.keys())
    success_rates = [industry_success_rates[ind] * 100 for ind in industries]
    colors = ['#007bff' if ind == selected_industry else '#e9ecef' for ind in industries]
    
    fig = go.Figure(data=[
        go.Bar(
            x=industries,
            y=success_rates,
            marker_color=colors,
            text=[f"{rate:.1f}%" for rate in success_rates],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Success Rate: %{y}%<extra></extra>'
        )
    ])
    
    # Add average line
    avg_success_rate = np.mean(success_rates)
    fig.add_hline(
        y=avg_success_rate, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Industry Average: {avg_success_rate:.1f}%",
        annotation_position="top left"
    )
    
    fig.update_layout(
        title=f'<b>Industry Success Rates - {selected_industry} Highlighted</b>',
        xaxis_title='<b>Industry</b>',
        yaxis_title='<b>Success Rate (%)</b>',
        height=450,
        template='plotly_white',
        xaxis={'tickangle': 45}
    )
    
    return fig

def create_financial_metrics_dashboard(input_data):
    """Enhanced financial metrics dashboard"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Monthly Recurring Revenue', 'Burn Rate', 'Customer Acquisition Cost', 'Financial Health Score'),
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}]]
    )
    
    # Monthly Revenue Gauge
    mrr = input_data.get('monthly_recurring_revenue', 0)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=mrr,
        title={'text': "MRR ($)", 'font': {'size': 14}},
        gauge={
            'axis': {'range': [None, 150000]},
            'bar': {'color': "green"},
            'steps': [
                {'range': [0, 30000], 'color': "lightgray"},
                {'range': [30000, 60000], 'color': "yellow"},
                {'range': [60000, 150000], 'color': "lightgreen"}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 100000}
        },
        number={'font': {'size': 14}}
    ), row=1, col=1)
    
    # Burn Rate Gauge
    burn_rate = input_data.get('burn_rate', 0)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=burn_rate,
        title={'text': "Burn Rate ($K)", 'font': {'size': 14}},
        gauge={
            'axis': {'range': [0, 30]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 10], 'color': "lightgreen"},
                {'range': [10, 20], 'color': "yellow"},
                {'range': [20, 30], 'color': "lightcoral"}
            ],
            'threshold': {'line': {'color': "darkred", 'width': 4}, 'thickness': 0.75, 'value': 25}
        },
        number={'font': {'size': 14}}
    ), row=1, col=2)
    
    # Customer Acquisition Cost Gauge
    cac = input_data.get('customer_acquisition_cost', 0)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=cac,
        title={'text': "CAC ($)", 'font': {'size': 14}},
        gauge={
            'axis': {'range': [0, 300]},
            'bar': {'color': "orange"},
            'steps': [
                {'range': [0, 75], 'color': "lightgreen"},
                {'range': [75, 150], 'color': "yellow"},
                {'range': [150, 300], 'color': "lightcoral"}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 200}
        },
        number={'font': {'size': 14}}
    ), row=2, col=1)
    
    # Financial Health Score
    revenue_score = min(100, (mrr / 50000) * 30) if mrr > 0 else 0
    burn_score = max(0, 40 - (burn_rate / 15) * 40) if burn_rate > 0 else 40
    cac_score = max(0, 30 - (cac / 150) * 30) if cac > 0 else 30
    
    financial_health = revenue_score + burn_score + cac_score
    
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=financial_health,
        title={'text': "Health Score", 'font': {'size': 14}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "blue"},
            'steps': [
                {'range': [0, 40], 'color': "lightcoral"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "lightgreen"}
            ],
            'threshold': {'line': {'color': "green", 'width': 4}, 'thickness': 0.75, 'value': 80}
        },
        number={'font': {'size': 14}}
    ), row=2, col=2)
    
    fig.update_layout(height=500, title_text="<b>Financial Metrics Dashboard</b>")
    
    return fig

def create_growth_trend_chart(input_data):
    """Create a projected growth trend chart"""
    months = list(range(1, 13))
    current_revenue = input_data.get('monthly_recurring_revenue', 40000)
    growth_rate = input_data.get('Revenue_Growth', 10) / 100
    
    # Project revenue growth
    projected_revenue = [current_revenue * (1 + growth_rate) ** (month/12) for month in months]
    
    # Create optimistic and pessimistic scenarios
    optimistic_revenue = [current_revenue * (1 + growth_rate * 1.5) ** (month/12) for month in months]
    pessimistic_revenue = [current_revenue * (1 + growth_rate * 0.5) ** (month/12) for month in months]
    
    fig = go.Figure()
    
    # Add pessimistic scenario
    fig.add_trace(go.Scatter(
        x=months, y=pessimistic_revenue,
        name='Pessimistic Scenario',
        line=dict(color='red', dash='dash'),
        fill=None
    ))
    
    # Add optimistic scenario
    fig.add_trace(go.Scatter(
        x=months, y=optimistic_revenue,
        name='Optimistic Scenario',
        line=dict(color='green', dash='dash'),
        fill='tonexty',
        fillcolor='rgba(0,255,0,0.1)'
    ))
    
    # Add projected scenario
    fig.add_trace(go.Scatter(
        x=months, y=projected_revenue,
        name='Expected Growth',
        line=dict(color='blue', width=3),
        fill=None
    ))
    
    fig.update_layout(
        title='<b>12-Month Revenue Growth Projection</b>',
        xaxis_title='<b>Months</b>',
        yaxis_title='<b>Monthly Revenue ($)</b>',
        height=400,
        template='plotly_white',
        hovermode='x'
    )
    
    return fig

def create_team_composition_chart(input_data):
    """Create a team composition pie chart"""
    employee_count = input_data.get('Employee_Count', 30)
    cofounders = input_data.get('Number_of_Co_founders', 2)
    senior_leadership = input_data.get('Team_size_Senior_leadership', 5)
    
    # Calculate team composition
    regular_employees = max(0, employee_count - cofounders - senior_leadership)
    
    labels = ['Regular Employees', 'Senior Leadership', 'Co-founders']
    values = [regular_employees, senior_leadership, cofounders]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)),
        textinfo='label+percent+value',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title='<b>Team Composition Analysis</b>',
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    return fig

def create_competitive_analysis_chart(input_data, selected_industry):
    """Create competitive positioning chart"""
    market_competition = input_data.get('Market_Competition', 3)
    customer_satisfaction = input_data.get('customer_satisfaction_score', 7)
    scalability = input_data.get('scalability_score', 70)
    
    # Sample competitor data based on industry
    competitors_data = {
        'Technology': [
            {'name': 'Competitor A', 'satisfaction': 8.2, 'scalability': 85, 'competition': 4},
            {'name': 'Competitor B', 'satisfaction': 7.8, 'scalability': 78, 'competition': 3},
            {'name': 'Competitor C', 'satisfaction': 7.5, 'scalability': 82, 'competition': 5}
        ],
        'Healthcare': [
            {'name': 'Competitor A', 'satisfaction': 8.5, 'scalability': 75, 'competition': 3},
            {'name': 'Competitor B', 'satisfaction': 7.9, 'scalability': 80, 'competition': 4},
            {'name': 'Competitor C', 'satisfaction': 8.1, 'scalability': 72, 'competition': 2}
        ],
        'E-Commerce': [
            {'name': 'Competitor A', 'satisfaction': 7.8, 'scalability': 88, 'competition': 5},
            {'name': 'Competitor B', 'satisfaction': 8.0, 'scalability': 75, 'competition': 4},
            {'name': 'Competitor C', 'satisfaction': 7.6, 'scalability': 82, 'competition': 3}
        ]
    }
    
    # Default competitors if industry not in sample data
    default_competitors = [
        {'name': 'Competitor A', 'satisfaction': 8.0, 'scalability': 80, 'competition': 4},
        {'name': 'Competitor B', 'satisfaction': 7.7, 'scalability': 75, 'competition': 3},
        {'name': 'Competitor C', 'satisfaction': 7.9, 'scalability': 78, 'competition': 4}
    ]
    
    competitors = competitors_data.get(selected_industry, default_competitors)
    
    fig = go.Figure()
    
    # Add competitors
    for comp in competitors:
        fig.add_trace(go.Scatter(
            x=[comp['satisfaction']],
            y=[comp['scalability']],
            mode='markers+text',
            marker=dict(
                size=comp['competition'] * 15,
                color='lightcoral',
                opacity=0.7,
                line=dict(width=2, color='red')
            ),
            text=[comp['name']],
            textposition="top center",
            name=comp['name'],
            hovertemplate=f"<b>{comp['name']}</b><br>Satisfaction: {comp['satisfaction']}<br>Scalability: {comp['scalability']}<br>Market Position: {comp['competition']}<extra></extra>"
        ))
    
    
    fig.add_trace(go.Scatter(
        x=[customer_satisfaction],
        y=[scalability],
        mode='markers+text',
        marker=dict(
            size=market_competition * 20,
            color='blue',
            opacity=0.8,
            line=dict(width=3, color='darkblue')
        ),
        text=['Your Domain'],
        textposition="top center",
        name='Your Project',
        hovertemplate=f"<b>Your Project</b><br>Satisfaction: {customer_satisfaction}<br>Scalability: {scalability}<br>Market Position: {market_competition}<extra></extra>"
    ))
    
    fig.update_layout(
        title=f'<b>Competitive Positioning - {selected_industry}</b>',
        xaxis_title='<b>Customer Satisfaction Score</b>',
        yaxis_title='<b>Scalability Score</b>',
        height=500,
        template='plotly_white',
        showlegend=False
    )
    
    # Add quadrant lines
    fig.add_hline(y=75, line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_vline(x=7.5, line_dash="dot", line_color="gray", opacity=0.5)
    
    # Add quadrant labels
    fig.add_annotation(x=9, y=90, text="<b>Stars</b><br>(High Satisfaction, High Scalability)", 
                      showarrow=False, bgcolor="lightgreen", opacity=0.8)
    fig.add_annotation(x=6, y=90, text="<b>Question Marks</b><br>(Low Satisfaction, High Scalability)", 
                      showarrow=False, bgcolor="yellow", opacity=0.8)
    fig.add_annotation(x=9, y=60, text="<b>Cash Cows</b><br>(High Satisfaction, Low Scalability)", 
                      showarrow=False, bgcolor="lightblue", opacity=0.8)
    fig.add_annotation(x=6, y=60, text="<b>Dogs</b><br>(Low Satisfaction, Low Scalability)", 
                      showarrow=False, bgcolor="lightcoral", opacity=0.8)
    
    return fig

def create_success_factors_chart(analysis_df):
    """Create a horizontal bar chart of success factors"""
    # Sort by impact and status
    impact_weights = {'High': 3, 'Medium': 2, 'Low': 1}
    status_weights = {'Critical': 1, 'Needs Improvement': 2, 'Good': 3, 'Excellent': 4}
    
    analysis_df['Impact_Score'] = analysis_df['Impact'].map(impact_weights)
    analysis_df['Status_Score'] = analysis_df['Status'].map(status_weights)
    analysis_df['Priority_Score'] = analysis_df['Impact_Score'] * (5 - analysis_df['Status_Score'])
    
    top_factors = analysis_df.nlargest(10, 'Priority_Score')
    
    colors = {
        'Critical': '#dc3545',
        'Needs Improvement': '#ffc107',
        'Good': '#20c997',
        'Excellent': '#28a745'
    }
    
    fig = go.Figure(go.Bar(
        x=top_factors['Priority_Score'],
        y=top_factors['Feature'],
        orientation='h',
        marker=dict(
            color=[colors[status] for status in top_factors['Status']],
            line=dict(color='rgb(8,48,107)', width=1.5)
        ),
        text=top_factors['Status'],
        textposition='inside',
        hovertemplate='<b>%{y}</b><br>Priority Score: %{x}<br>Status: %{text}<br>Impact: %{customdata}<extra></extra>',
        customdata=top_factors['Impact']
    ))
    
    fig.update_layout(
        title='<b>Success Factors Priority Matrix</b>',
        xaxis_title='<b>Priority Score (Higher = More Urgent)</b>',
        yaxis_title='<b>Features</b>',
        height=500,
        template='plotly_white',
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def create_kpi_trend_chart(input_data):
    """Create a simulated KPI trend chart"""
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    
    # Simulate historical data based on current values
    current_satisfaction = input_data.get('customer_satisfaction_score', 7.5)
    current_churn = input_data.get('churn_rate', 0.15)
    current_adoption = input_data.get('User_Adoption_Rate', 0.6)
    
    # Create trend data
    satisfaction_trend = [
        current_satisfaction - 1.2, current_satisfaction - 0.8, 
        current_satisfaction - 0.5, current_satisfaction - 0.2,
        current_satisfaction + 0.1, current_satisfaction
    ]
    
    churn_trend = [
        current_churn + 0.05, current_churn + 0.03,
        current_churn + 0.02, current_churn - 0.01,
        current_churn - 0.02, current_churn
    ]
    
    adoption_trend = [
        current_adoption - 0.15, current_adoption - 0.10,
        current_adoption - 0.05, current_adoption,
        current_adoption + 0.03, current_adoption + 0.05
    ]
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Customer Satisfaction Score',     'Churn Rate (%)',   'User Adoption Rate (%)'),
        vertical_spacing=0.08
    )
    
    # Customer Satisfaction
    fig.add_trace(go.Scatter(
        x=months, y=satisfaction_trend,
        mode='lines+markers',
        name='Customer Satisfaction',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ), row=1, col=1)
    
    # Churn Rate
    churn_percent = [rate * 100 for rate in churn_trend]
    fig.add_trace(go.Scatter(
        x=months, y=churn_percent,
        mode='lines+markers',
        name='Churn Rate',
        line=dict(color='red', width=3),
        marker=dict(size=8)
    ), row=2, col=1)
    
    # User Adoption
    adoption_percent = [rate * 100 for rate in adoption_trend]
    fig.add_trace(go.Scatter(
        x=months, y=adoption_percent,
        mode='lines+markers',
        name='User Adoption',
        line=dict(color='green', width=3),
        marker=dict(size=8)
    ), row=3, col=1)
    
    fig.update_layout(
        title='<b>Key Performance Indicators Trend</b>',
        height=600,
        template='plotly_white',
        showlegend=False
    )
    
    return fig

def generate_detailed_report(input_data, result, analysis_df, selected_industry):
    """Generate comprehensive PDF report data"""
    from datetime import datetime
    
    report_content = f"""
# Project Tracking ANALYSIS REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## EXECUTIVE SUMMARY
**Company Industry:** {selected_industry}
**Success Probability:** {result['success_probability']:.1%}
**Overall Prediction:** {result['prediction']}

## KEY FINDINGS
- Total Metrics Analyzed: {len(analysis_df)}
- Excellent Performance: {len(analysis_df[analysis_df['Status'] == 'Excellent'])} metrics
- Critical Areas: {len(analysis_df[analysis_df['Status'] == 'Critical'])} metrics
- Areas for Improvement: {len(analysis_df[analysis_df['Status'] == 'Needs Improvement'])} metrics

## DETAILED METRICS ANALYSIS
"""
    
    for _, row in analysis_df.iterrows():
        report_content += f"""
### {row['Feature']}
- Current Value: {row['Your Value']}
- Success Benchmark: {row['Success Benchmark']}
- Status: {row['Status']}
- Business Impact: {row['Impact']}
- Gap Analysis: {row['Gap']:.2f}
"""
    
    report_content += f"""
## FINANCIAL HEALTH SNAPSHOT
- Monthly Recurring Revenue: ${input_data.get('monthly_recurring_revenue', 0):,.2f}
- Monthly Burn Rate: ${input_data.get('burn_rate', 0) * 1000:,.2f}
- Customer Acquisition Cost: ${input_data.get('customer_acquisition_cost', 0):,.2f}
- Customer Satisfaction Score: {input_data.get('customer_satisfaction_score', 0):.1f}/10
- Revenue Growth Rate: {input_data.get('Revenue_Growth', 0):.1f}%

## TEAM COMPOSITION
- Total Employees: {input_data.get('Employee_Count', 0)}
- Co-founders: {input_data.get('Number_of_Co_founders', 0)}
- Senior Leadership: {input_data.get('Team_size_Senior_leadership', 0)}
- Team Growth: +{input_data.get('team_size_grown', 0)} recent hires

## STRATEGIC RECOMMENDATIONS

### High Priority Actions
"""
    
    critical_features = analysis_df[analysis_df['Status'] == 'Critical']
    for _, feature in critical_features.iterrows():
        report_content += f"- **{feature['Feature']}**: Requires immediate attention (Current: {feature['Your Value']}, Target: {feature['Success Benchmark']})\n"
    
    report_content += """
### Medium Priority Improvements
"""
    
    improvement_features = analysis_df[analysis_df['Status'] == 'Needs Improvement']
    for _, feature in improvement_features.iterrows():
        report_content += f"- **{feature['Feature']}**: Optimize for better performance (Gap: {feature['Gap']:.2f})\n"
    
    report_content += f"""
## COMPETITIVE POSITIONING
Your Project is positioned in the {selected_industry} industry with a market competition level of {input_data.get('Market_Competition', 'N/A')}.

## GROWTH PROJECTIONS
Based on current revenue growth rate of {input_data.get('Revenue_Growth', 0):.1f}%, projected 12-month outlook shows potential for significant expansion with proper execution of recommended improvements.

---
*This report is generated by the Project Tracking Predictor AI system and should be used in conjunction with professional business advice.*
"""
    
    return report_content

def main():
    # Initialize predictor
    if 'predictor' not in st.session_state:
        st.session_state.predictor = StartupSuccessPredictor()
    
    predictor = st.session_state.predictor
    
    # Header with gradient background
    st.markdown("""
    <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin-bottom: 2rem;'>
        <h1 style='color: white; text-align: center; margin: 0; font-size: 3rem;'>🚀 Project Tracking Predictor</h1>
        <p style='color: white; text-align: center; margin: 0.5rem 0 0 0; font-size: 1.2rem;'>
            AI-Powered Project Status Prediction with Advanced Machine Learning
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for inputs
    st.sidebar.markdown("## 📝 Enter Your Project Industry Details")
    
    # Industry selection with enhanced options
    industries = [
        'Technology', 'Healthcare', 'E-Commerce', 'Analytics', 'Marketing',
        'Cloud Computing', 'Mobile', 'Finance', 'Food & Beverages',
        'Education', 'Real Estate'
    ]
    
    selected_industry = st.sidebar.selectbox(
        "🏢 Industry of Project",
        industries,
        help="Select your Project primary industry sector"
    )
    
    # Map industry to numeric value for model
    industry_mapping = {industry: idx + 1 for idx, industry in enumerate(industries)}
    
    # Input form with enhanced validation
    with st.sidebar.form("prediction_form"):
        st.markdown("### 📊 Team & Business Metrics")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            employee_count = st.number_input("👥 Employee Count", min_value=1, max_value=1000, value=30, help="Total number of employees")
            team_size_grown = st.number_input("📈 Team Size Grown", min_value=0, max_value=20, value=2, help="How much has your team grown?")
            num_cofounders = st.number_input("🤝 Number of Co-founders", min_value=1, max_value=6, value=2, help="Total co-founders including yourself")
        
        with col2:
            senior_leadership = st.number_input("👔 Senior Leadership Team Size", min_value=1, max_value=30, value=5, help="C-level and VP positions")
            market_competition = st.selectbox("🏆 Market Competition Level", 
                                            options=[1, 2, 3, 4, 5], 
                                            index=2, 
                                            format_func=lambda x: f"{x} - {'Very Low' if x==1 else 'Low' if x==2 else 'Medium' if x==3 else 'High' if x==4 else 'Very High'}")
        
        st.markdown("### 🎯 Performance Metrics")
        customer_satisfaction = st.slider("😊 Customer Satisfaction Score", 1.0, 10.0, 7.5, 0.1, help="Average customer satisfaction rating")
        churn_rate = st.slider("📉 Monthly Churn Rate", 0.0, 0.5, 0.15, 0.01, help="Percentage of customers lost per month")
        scalability_score = st.slider("📊 Scalability Score", 0, 100, 75, help="How scalable is your business model?")
        user_adoption_rate = st.slider("👤 User Adoption Rate", 0.0, 1.0, 0.65, 0.01, help="Percentage of users actively using your product")
        revenue_growth = st.slider("💹 Monthly Revenue Growth (%)", -30.0, 100.0, 15.0, 0.5, help="Month-over-month revenue growth")
        business_sustainability = st.slider("🌱 Business Sustainability Score", 0, 100, 80, help="Long-term business viability score")
        
        st.markdown("### 💰 Financial Metrics")
        monthly_revenue = st.number_input("💵 Monthly Recurring Revenue ($)", min_value=0, max_value=1000000, value=50000, step=1000,
                                        help="Total monthly recurring revenue")
        burn_rate = st.number_input("🔥 Monthly Burn Rate ($)", min_value=0, max_value=100000, value=10000, step=500,
                                  help="Monthly cash burn rate")
        customer_acquisition_cost = st.number_input("💸 Customer Acquisition Cost ($)", min_value=0, max_value=2000, value=85, step=5,
                                                   help="Average cost to acquire one customer")
        
        submit_button = st.form_submit_button("🔮 Predict Success & Generate Analysis", use_container_width=True)
    
    # Add industry info card
    if selected_industry:
        industry_success_rate = predictor.industry_success_rates.get(selected_industry, 0.6)
        st.markdown(f"""
        <div class="industry-card">
            <h3>🏢 {selected_industry} Industry Analysis</h3>
            <p><b>Industry Success Rate:</b> {industry_success_rate:.1%}</p>
            <p>Specialized benchmarks and insights applied for your industry</p>
        </div>
        """, unsafe_allow_html=True)
    
    if submit_button:
        # Prepare input data
        input_data = {
            'Industry_of_Project': industry_mapping[selected_industry],
            'Employee_Count': employee_count,
            'team_size_grown': team_size_grown,
            'Number_of_Co_founders': num_cofounders,
            'Team_size_Senior_leadership': senior_leadership,
            'customer_satisfaction_score': customer_satisfaction,
            'scalability_score': scalability_score,
            'churn_rate': churn_rate,
            'burn_rate': burn_rate / 1000,  # Convert to thousands
            'customer_acquisition_cost': customer_acquisition_cost,
            'monthly_recurring_revenue': monthly_revenue,
            'business_sustainability_score': business_sustainability,
            'User_Adoption_Rate': user_adoption_rate,
            'Revenue_Growth': revenue_growth,
            'Market_Competition': market_competition
        }
        
        # Make prediction
        with st.spinner("🤖 Analyzing your Project data..."):
            result = predictor.predict(input_data)
            probability = result['success_probability']
            
            # Create analysis dataframe
            analysis_df = predictor.analyze_features(input_data)
        
        # Display results with enhanced styling
        st.markdown("## 🎯 Prediction Results")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            # Outcome card
            if result['prediction'] == 'Success':
                st.markdown(f"""
                <div class="success-card">
                    <h2>✅ HIGH SUCCESS POTENTIAL</h2>
                    <h1 style='font-size: 3rem; margin: 10px 0;'>{probability:.1%}</h1>
                    <p style='font-size: 1.1rem;'>Strong indicators for success</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="failure-card">
                    <h2>⚠️ NEEDS IMPROVEMENT</h2>
                    <h1 style='font-size: 3rem; margin: 10px 0;'>{probability:.1%}</h1>
                    <p style='font-size: 1.1rem;'>Focus on key areas</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Enhanced gauge chart
            gauge_fig = create_success_probability_gauge(probability)
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        with col3:
            # Risk assessment
            if probability > 0.75:
                risk_level = "LOW RISK"
                risk_desc = "Excellent prospects"
                risk_color = "success-card"
            elif probability > 0.6:
                risk_level = "MODERATE RISK"
                risk_desc = "Good potential"
                risk_color = "medium-risk-card"
            elif probability > 0.4:
                risk_level = "HIGH RISK"
                risk_desc = "Needs attention"
                risk_color = "medium-risk-card"
            else:
                risk_level = "VERY HIGH RISK"
                risk_desc = "Immediate action required"
                risk_color = "failure-card"
            
            st.markdown(f"""
            <div class="{risk_color}">
                <h3>🎯 Risk Assessment</h3>
                <h1 style='font-size: 1.5rem; margin: 10px 0;'>{risk_level}</h1>
                <p style='font-size: 1.1rem;'>{risk_desc}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Tabbed interface for charts
        st.markdown("## 📊 Comprehensive Analysis Dashboard")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈  Performance Analysis  ", 
            "💰  Financial Dashboard  ", 
            "🎯  Strategic Insights  ", 
            "🏆  Competitive Analysis  ", 
            "📋  Detailed Breakdown  "
        ])
        
        with tab1:
            st.markdown("### Performance Metrics Overview")
            
            col1, col2 = st.columns(2)
            with col1:
                comparison_fig = create_feature_comparison_chart(analysis_df)
                st.plotly_chart(comparison_fig, use_container_width=True)
            
            with col2:
                radar_fig = create_radar_chart(input_data)
                st.plotly_chart(radar_fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                kpi_trend_fig = create_kpi_trend_chart(input_data)
                st.plotly_chart(kpi_trend_fig, use_container_width=True)
            
            with col2:
                team_fig = create_team_composition_chart(input_data)
                st.plotly_chart(team_fig, use_container_width=True)
        
        with tab2:
            st.markdown("### Financial Health Analysis")
            
            financial_fig = create_financial_metrics_dashboard(input_data)
            st.plotly_chart(financial_fig, use_container_width=True)
            
            growth_fig = create_growth_trend_chart(input_data)
            st.plotly_chart(growth_fig, use_container_width=True)
        
        with tab3:
            st.markdown("### Strategic Business Insights")
            
            col1, col2 = st.columns(2)
            with col1:
                risk_fig = create_risk_assessment_donut(analysis_df)
                st.plotly_chart(risk_fig, use_container_width=True)
            
            with col2:
                success_factors_fig = create_success_factors_chart(analysis_df)
                st.plotly_chart(success_factors_fig, use_container_width=True)
            
            industry_fig = create_industry_comparison_chart(selected_industry, predictor.industry_success_rates)
            st.plotly_chart(industry_fig, use_container_width=True)
        
        with tab4:
            st.markdown("### Market Position & Competition")
            
            competitive_fig = create_competitive_analysis_chart(input_data, selected_industry)
            st.plotly_chart(competitive_fig, use_container_width=True)
            
            # Market insights
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Market Position", f"Level {market_competition}", 
                         delta="Competitive" if market_competition >= 3 else "Opportunity")
            with col2:
                industry_avg_satisfaction = 7.8
                satisfaction_delta = customer_satisfaction - industry_avg_satisfaction
                st.metric("vs Industry Avg Satisfaction", f"{customer_satisfaction:.1f}", 
                         delta=f"{satisfaction_delta:+.1f}")
            with col3:
                industry_avg_scalability = 75
                scalability_delta = scalability_score - industry_avg_scalability  
                st.metric("vs Industry Avg Scalability", f"{scalability_score}", 
                         delta=f"{scalability_delta:+.0f}")
        
        with tab5:
            st.markdown("### Detailed Feature Analysis")
            
            # Enhanced feature analysis table with color coding
            def color_status(val):
                colors = {
                    'Excellent': 'background-color: #d4edda; color: #155724; font-weight: bold;',
                    'Good': 'background-color: #d1ecf1; color: #0c5460; font-weight: bold;',
                    'Needs Improvement': 'background-color: #fff3cd; color: #856404; font-weight: bold;',
                    'Critical': 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
                }
                return colors.get(val, '')
            
            def color_impact(val):
                colors = {
                    'High': 'background-color: #ff6b6b; color: white; font-weight: bold;',
                    'Medium': 'background-color: #feca57; color: black; font-weight: bold;',
                    'Low': 'background-color: #48dbfb; color: black; font-weight: bold;'
                }
                return colors.get(val, '')
            
            styled_df = analysis_df.style\
                .applymap(color_status, subset=['Status'])\
                .applymap(color_impact, subset=['Impact'])\
                .format({'Your Value': '{:.2f}', 'Success Benchmark': '{:.2f}', 'Gap': '{:.2f}'})
            
            st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Action items and recommendations
        st.markdown("## 🎯 Actionable Recommendations")
        
        critical_features = analysis_df[analysis_df['Status'] == 'Critical']
        improvement_features = analysis_df[analysis_df['Status'] == 'Needs Improvement'] 
        excellent_features = analysis_df[analysis_df['Status'] == 'Excellent']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if len(critical_features) > 0:
                st.markdown("### 🚨 Immediate Action Required")
                for _, feature in critical_features.iterrows():
                    with st.expander(f"🔴 {feature['Feature']}", expanded=True):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Current", f"{feature['Your Value']}")
                            st.metric("Target", f"{feature['Success Benchmark']}")
                        with col_b:
                            gap = abs(feature['Gap'])
                            improvement_needed = (gap / feature['Success Benchmark']) * 100
                            st.metric("Gap", f"{feature['Gap']:+.2f}")
                            st.metric("Improvement Needed", f"{improvement_needed:.1f}%")
                        
                        # Specific recommendations
                        recommendations = {
                            'Customer Satisfaction Score': "💡 Focus on customer support, product quality, and user experience improvements",
                            'Churn Rate': "💡 Implement retention strategies, improve onboarding, and enhance customer success programs",
                            'Monthly Recurring Revenue': "💡 Optimize pricing strategy, expand customer base, and increase upselling",
                            'Scalability Score': "💡 Invest in infrastructure, automate processes, and standardize operations",
                            'Revenue Growth': "💡 Expand marketing efforts, enter new markets, and improve sales conversion",
                            'Burn Rate': "💡 Reduce unnecessary expenses, optimize operations, and extend runway"
                        }
                        
                        st.info(recommendations.get(feature['Feature'], "💡 This metric requires immediate attention and strategic focus"))
            else:
                st.success("✅ No critical issues identified!")
        
        with col2:
            if len(improvement_features) > 0:
                st.markdown("### ⚠️ Areas for Enhancement")
                for _, feature in improvement_features.iterrows():
                    with st.expander(f"🟡 {feature['Feature']}"):
                        st.metric("Current vs Target", 
                                f"{feature['Your Value']:.2f} → {feature['Success Benchmark']}", 
                                delta=f"{feature['Gap']:+.2f}")
                        st.caption(f"**Priority:** {feature['Impact']} Impact")
            else:
                st.info("ℹ️ All metrics are performing well or excellently!")
        
        with col3:
            if len(excellent_features) > 0:
                st.markdown("### ✅ Your Competitive Advantages")
                for _, feature in excellent_features.iterrows():
                    st.success(f"**{feature['Feature']}**: {feature['Your Value']} (Target: {feature['Success Benchmark']})")
                
            else:
                st.warning("Focus on building competitive advantages in key areas")
        
        # Executive summary
        st.markdown("## 📋 Executive Summary")
        
        with st.container():
            summary_col1, summary_col2 = st.columns([2, 1])
            
            with summary_col1:
                risk_level_text = "low" if probability > 0.7 else "moderate" if probability > 0.5 else "high"
                
                executive_summary = f"""
                **Project Tracking Predictor Summary**
                
                Your {selected_industry} Project shows a **{probability:.1%} probability of success** with **{risk_level_text} risk** profile.
                
                **Key Findings:**
                - **{len(excellent_features)} metrics** are performing excellently above industry benchmarks
                - **{len(critical_features)} critical areas** require immediate attention
                - **{len(improvement_features)} metrics** have room for strategic improvement
                
                **Strategic Priority:** Focus on {"critical metrics first" if len(critical_features) > 0 else "maintaining strengths while improving key areas"}
                
                **Industry Context:** Your Project  is positioned {"above" if probability > predictor.industry_success_rates.get(selected_industry, 0.6) else "below"} the {selected_industry} industry average success rate of {predictor.industry_success_rates.get(selected_industry, 0.6):.1%}.
                """
                
                st.markdown(executive_summary)
            
            with summary_col2:
                # Quick stats
                st.metric("Overall Score", f"{probability*100:.1f}/100", delta=f"Industry: {predictor.industry_success_rates.get(selected_industry, 0.6)*100:.1f}")
                st.metric("Critical Issues", len(critical_features), delta=f"{'⚠️' if len(critical_features) > 2 else '✅' if len(critical_features) == 0 else '⚡'}")
                st.metric("Strengths", len(excellent_features), delta="💪" if len(excellent_features) > 3 else "📈")
        
        # Download report option
        st.markdown("---")
        if st.button("📥 Generate Detailed Report", use_container_width=True):
            report_data = generate_detailed_report(input_data, result, analysis_df, selected_industry)
            
            # Create downloadable text file
            st.text_area("📄 Your Detailed Project Industry Details Report", 
                        value=report_data, 
                        height=400,
                        help="Copy this comprehensive report for your records or sharing with stakeholders")
            
            # Additional insights
            st.info("""
            **💡 How to Use This Report:**
            
            1. **Immediate Actions**: Focus on critical metrics first - these have the highest impact on success probability
            2. **Strategic Planning**: Use the competitive analysis to position yourself in the market
            3. **Financial Planning**: Leverage the financial projections for investor discussions
            4. **Team Development**: Use team composition insights for hiring and organizational planning
            5. **Performance Tracking**: Monitor the KPIs highlighted in the trend analysis
            
            **🎯 Next Steps:**
            - Share this analysis with your co-founders and key stakeholders
            - Create action plans for each critical metric
            - Set up regular monitoring for the key performance indicators
            - Consider seeking mentorship or consulting in areas marked as critical
            """)
        
        # Final recommendations and call-to-action
        st.markdown("""
        ---
        ## 🚀 Ready to Take Action?
        
        Your Project Industry Details analysis is complete! Use these insights to:
        - **Prioritize improvements** in critical areas
        - **Leverage your strengths** as competitive advantages  
        - **Make data-driven decisions** for future growth
        - **Communicate effectively** with investors and stakeholders
        
        Remember: Success in Project is about continuous improvement and strategic execution. 
        Regularly reassess your metrics and adjust your strategy accordingly.
        """)

# Run the application
if __name__ == "__main__":
    main()
