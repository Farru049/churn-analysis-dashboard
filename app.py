import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

def preprocess_data_for_vis(df):
    """Preprocess data for visualization"""
    # Create a copy of the dataframe
    df_vis = df.copy()
    
    # Drop non-numeric columns that we don't need for visualization
    columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
    df_vis = df_vis.drop(columns_to_drop, axis=1)
    
    # Convert categorical variables to numeric
    categorical_columns = ['Geography', 'Gender']
    label_encoders = {}
    
    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        df_vis[column] = label_encoders[column].fit_transform(df_vis[column])
    
    return df_vis, label_encoders

def create_distribution_plots(df):
    """Create distribution plots for numerical features"""
    numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
    
    fig = make_subplots(rows=2, cols=3, subplot_titles=numerical_cols)
    row, col = 1, 1
    
    for col_name in numerical_cols:
        fig.add_trace(
            go.Histogram(x=df[col_name], name=col_name, nbinsx=30, showlegend=False),
            row=row, col=col
        )
        
        col += 1
        if col > 3:
            row += 1
            col = 1
    
    fig.update_layout(height=700, title_text="Distribution of Numerical Features")
    return fig

def create_correlation_heatmap(df_processed):
    """Create correlation heatmap"""
    # Select only numeric columns
    numeric_df = df_processed.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmin=-1,
        zmax=1
    ))
    
    fig.update_layout(
        title='Feature Correlation Heatmap',
        width=700,
        height=700
    )
    
    return fig

def create_feature_importance_plot(feature_importance):
    """Create feature importance visualization"""
    fig = px.bar(feature_importance,
                 x='importance',
                 y='feature',
                 orientation='h',
                 title='Feature Importance',
                 labels={'importance': 'Importance Score', 'feature': 'Feature'})
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig

def create_confusion_matrix_plot(conf_matrix):
    """Create confusion matrix visualization"""
    labels = ['Not Churned', 'Churned']
    
    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=labels,
        y=labels,
        colorscale='RdBu',
        showscale=True
    ))
    
    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix[i])):
            fig.add_annotation(
                x=labels[j],
                y=labels[i],
                text=str(conf_matrix[i, j]),
                showarrow=False,
                font=dict(color='white' if conf_matrix[i, j] > conf_matrix.max()/2 else 'black')
            )
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        xaxis=dict(side='bottom'),
        width=500,
        height=500
    )
    
    return fig

def create_geographical_distribution(df):
    """Create geographical distribution visualization"""
    geo_counts = df['Geography'].value_counts()
    fig = px.pie(values=geo_counts.values, 
                 names=geo_counts.index,
                 title='Customer Distribution by Geography')
    return fig

def create_age_balance_scatter(df):
    """Create scatter plot of Age vs Balance colored by Churn"""
    fig = px.scatter(df, 
                    x='Age', 
                    y='Balance',
                    color='Exited',
                    title='Age vs Balance by Churn Status',
                    labels={'Exited': 'Churned'})
    return fig

def create_model(df):
    """Create and train the model with proper preprocessing"""
    # Preprocess data
    df_processed, label_encoders = preprocess_data_for_vis(df)
    
    X = df_processed.drop('Exited', axis=1)
    y = df_processed['Exited']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, min_samples_split=5)
    model.fit(X_train_scaled, y_train)
    
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, scaler, X_train_scaled, X_test_scaled, y_train, y_test, feature_importance, label_encoders

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return performance metrics"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    return metrics, conf_matrix

def main():
    st.set_page_config(layout="wide")
    
    st.title("📊 Customer Churn Analysis Dashboard")
    
    try:
        # Load data
        df = pd.read_csv('./Churn_Modelling.csv')
        
        # Preprocess data for visualization
        df_processed, label_encoders = preprocess_data_for_vis(df)
        
        # Create sidebar for data insights
        st.sidebar.header("📈 Dataset Overview")
        st.sidebar.write(f"Total Records: {len(df):,}")
        st.sidebar.write(f"Churn Rate: {(df['Exited'].mean()*100):.1f}%")
        
        # Create tabs for different sections
        tab1, tab2, tab3 = st.tabs(["📊 Data Visualization", "🎯 Model Performance", "🔍 Feature Analysis"])
        
        with tab1:
            st.header("Data Visualization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(create_geographical_distribution(df), use_container_width=True)
                st.plotly_chart(create_age_balance_scatter(df), use_container_width=True)
            
            with col2:
                st.plotly_chart(create_distribution_plots(df), use_container_width=True)
        
        # Train model and get results
        model, scaler, X_train, X_test, y_train, y_test, feature_importance, label_encoders = create_model(df)
        metrics, conf_matrix = evaluate_model(model, X_test, y_test)
        
        with tab2:
            st.header("Model Performance")
            
            metric_cols = st.columns(4)
            for i, (metric, value) in enumerate(metrics.items()):
                metric_cols[i].metric(
                    label=metric.capitalize(),
                    value=f"{value:.1%}"
                )
            
            st.plotly_chart(create_confusion_matrix_plot(conf_matrix), use_container_width=True)
        
        with tab3:
            st.header("Feature Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(create_correlation_heatmap(df_processed), use_container_width=True)
            
            with col2:
                st.plotly_chart(create_feature_importance_plot(feature_importance), use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in dashboard: {e}")
        st.error("Full error:", exc_info=True)

if __name__ == "__main__":
    main()