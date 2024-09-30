import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO, BytesIO
import chardet
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, roc_auc_score, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D, SimpleRNN, LSTM, GRU, InputLayer
import time
def detect_encoding(file):
    raw_data = file.read(10000)  
    result = chardet.detect(raw_data)
    file.seek(0)  
    return result['encoding']

# Function to create a deep learning model
def create_dl_model(model_type, input_dim, is_classification=True):
    model = Sequential()
    model.add(InputLayer(input_shape=(input_dim, 1)))
    model.add(Dense(32, activation='relu'))

    if model_type == 'FNN':
        model.add(Dense(1, activation='sigmoid' if is_classification else None))
    elif model_type == 'CNN':
        model.add(Conv1D(32, kernel_size=3, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(1, activation='sigmoid' if is_classification else None))
    elif model_type == 'RNN':
        model.add(SimpleRNN(32))
        model.add(Dense(1, activation='sigmoid' if is_classification else None))
    elif model_type == 'LSTM':
        model.add(LSTM(32))
        model.add(Dense(1, activation='sigmoid' if is_classification else None))
    elif model_type == 'GRU':
        model.add(GRU(32))
        model.add(Dense(1, activation='sigmoid' if is_classification else None))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy' if is_classification else 'mean_squared_error',
                  metrics=['accuracy'] if is_classification else ['mean_absolute_error'])
    return model

def clean_data(df, missing_thresh=0.5, fill_num_option='mean', fill_cat_option='unknown', drop_outliers=False):
    st.write("Initial Data Overview:")
    buffer = StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    df_cleaned = df.drop_duplicates()
    st.write(f"Removed duplicates. New shape: {df_cleaned.shape}")

    missing_percent = df_cleaned.isnull().mean()
    cols_to_drop = missing_percent[missing_percent > missing_thresh].index
    df_cleaned = df_cleaned.drop(columns=cols_to_drop)
    st.write(f"Dropped columns with more than {missing_thresh*100}% missing values. Remaining columns: {df_cleaned.columns.tolist()}")

    numeric_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns

    if fill_num_option in ['mean', 'median', 'mode']:
        if fill_num_option == 'mean':
            df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())
        elif fill_num_option == 'median':
            df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].median())
        elif fill_num_option == 'mode':
            df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mode().iloc[0])

    df_cleaned[categorical_cols] = df_cleaned[categorical_cols].fillna(fill_cat_option)
    st.write("Filled missing values based on the selected options.")

    if drop_outliers:
        for col in numeric_cols:
            q1 = df_cleaned[col].quantile(0.25)
            q3 = df_cleaned[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
        st.write(f"Removed outliers based on IQR. New shape: {df_cleaned.shape}")
    else:
        st.write("Outliers were not removed.")

    for col in categorical_cols:
        df_cleaned[col] = df_cleaned[col].str.lower()

    if 'date' in df_cleaned.columns:
        df_cleaned['date'] = pd.to_datetime(df_cleaned['date'], errors='coerce')

    st.write("Final Data Overview:")
    buffer = StringIO()
    df_cleaned.info(buf=buffer)
    st.text(buffer.getvalue())

    return df_cleaned

# Function to evaluate models
def evaluate_model(model, X_train, y_train, X_test, y_test, is_classification):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if is_classification:
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
        y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        return accuracy, roc_auc
    else:
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, mae, r2

def download_csv(data):
    csv = data.to_csv(index=False)
    b = BytesIO()
    b.write(csv.encode())
    b.seek(0)
    return b


# Function to evaluate a single model
def evaluate_single_model(model_name, model, X_train, y_train, X_test, y_test, is_classification):
    start_time = time.time()
    try:
        if model_name.startswith("Deep Learning"):
            early_stopping = EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)
            model.fit(X_train, y_train, epochs=5, batch_size=16, validation_split=0.2, callbacks=[early_stopping], verbose=0)
            if is_classification:
                y_pred_probs = model.predict(X_test)
                y_pred = (y_pred_probs > 0.5).astype(int).ravel()
                accuracy = accuracy_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred_probs)
                score = (accuracy, roc_auc)
            else:
                y_pred = model.predict(X_test).ravel()
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                score = (mse, mae, r2)
        else:
            score = evaluate_model(model, X_train, y_train, X_test, y_test, is_classification)

        st.write(f"{model_name} Evaluation Time: {time.time() - start_time:.2f} seconds")
        return model_name, score

    except MemoryError:
        st.error("Memory Error: This model requires more memory than is available.")
        return model_name, None
    except Exception as e:
        st.write(f"Error evaluating {model_name}: {e}")
        return model_name, None

# Function to evaluate all models
def evaluate_models(X_train, y_train, X_test, y_test, is_classification, params):
    X_train_ml, X_test_ml = X_train.reshape(X_train.shape[0], -1), X_test.reshape(X_test.shape[0], -1)

    if is_classification:
        ml_models = {
            "Logistic Regression": LogisticRegression(max_iter=100),
            "Random Forest": RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth']),
            "Decision Tree": DecisionTreeClassifier(max_depth=params['max_depth']),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=params['n_neighbors']),
            "Support Vector Machine": LinearSVC(),
            "XGBoost": XGBClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth']),
            "CatBoost": CatBoostClassifier(silent=True, iterations=params['n_estimators']),
            "LightGBM": LGBMClassifier(n_estimators=params['n_estimators'])
        }
    else:
        ml_models = {
            "Random Forest": RandomForestRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth']),
            "Decision Tree": DecisionTreeRegressor(max_depth=params['max_depth']),
            "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=params['n_neighbors']),
            "Support Vector Machine": SVR(),
            "XGBoost": XGBRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth']),
            "CatBoost": CatBoostRegressor(silent=True, iterations=params['n_estimators']),
            "LightGBM": LGBMRegressor(n_estimators=params['n_estimators'])
        }

    dl_models = {
        "Deep Learning CNN": create_dl_model('CNN', X_train.shape[1], is_classification),
        "Deep Learning RNN": create_dl_model('RNN', X_train.shape[1], is_classification),
        "Deep Learning LSTM": create_dl_model('LSTM', X_train.shape[1], is_classification),
        "Deep Learning GRU": create_dl_model('GRU', X_train.shape[1], is_classification),
    }

    best_ml_model_name = None
    best_ml_score = None
    best_dl_model_name = None
    best_dl_score = None

    # Evaluate ML models
    for model_name, model in ml_models.items():
        model_name, score = evaluate_single_model(model_name, model, X_train_ml, y_train, X_test_ml, y_test, is_classification)
        if score is not None:
            if is_classification:
                st.write(f"{model_name} Accuracy: {score[0]:.4f}, ROC AUC: {score[1]:.4f}")
                if best_ml_score is None or score[0] > best_ml_score[0]:
                    best_ml_score = score
                    best_ml_model_name = model_name
            else:
                mse, mae, r2 = score
                st.write(f"{model_name} MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
                if best_ml_score is None or mse < best_ml_score[0]:
                    best_ml_score = score
                    best_ml_model_name = model_name

    # Evaluate DL models
    for model_name, model in dl_models.items():
        model_name, score = evaluate_single_model(model_name, model, X_train, y_train, X_test, y_test, is_classification)
        if score is not None:
            if is_classification:
                st.write(f"{model_name} Accuracy: {score[0]:.4f}, ROC AUC: {score[1]:.4f}")
                if best_dl_score is None or score[0] > best_dl_score[0]:
                    best_dl_score = score
                    best_dl_model_name = model_name
            else:
                mse, mae, r2 = score
                st.write(f"{model_name} MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
                if best_dl_score is None or mse < best_dl_score[0]:
                    best_dl_score = score
                    best_dl_model_name = model_name

    return best_ml_model_name, best_ml_score, best_dl_model_name, best_dl_score

# Function to create a dashboard
def create_dashboard(df, chart_color):
    st.header('Advanced Dashboard')

    st.write(f"Total rows in dataset: {len(df)}")

    st.sidebar.header('Chart Axis Configuration')
    common_x_axis = st.sidebar.selectbox('Select column for x-axis:', df.columns.tolist())
    common_y_axis = st.sidebar.selectbox('Select column for y-axis:', df.columns.tolist())

    st.subheader('Interactive Bar Chart')
    if common_x_axis and common_y_axis:
        fig = px.bar(df, x=common_x_axis, y=common_y_axis, title=f"Bar Chart: {common_x_axis} vs {common_y_axis}", color_discrete_sequence=[chart_color])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader('Interactive Scatter Plot')
    scatter_color = st.selectbox('Select column for color coding (optional):', [None] + df.columns.tolist())
    if common_x_axis and common_y_axis:
        fig = px.scatter(df, x=common_x_axis, y=common_y_axis, color=scatter_color, title=f"Scatter Plot: {common_x_axis} vs {common_y_axis}", color_discrete_sequence=[chart_color])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader('Dynamic Line Chart')
    if common_x_axis and common_y_axis:
        fig = px.line(df, x=common_x_axis, y=common_y_axis, title=f"Line Chart: Trend over {common_x_axis}", color_discrete_sequence=[chart_color])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader('Advanced Pie Chart')
    pie_values = st.selectbox('Select column for pie chart values:', df.columns.tolist(), key='pie_values')
    pie_names = st.selectbox('Select column for pie chart names (optional):', [None] + df.columns.tolist(), key='pie_names')
    if pie_values:
        fig = px.pie(df, values=pie_values, names=pie_names, title=f"Pie Chart: Distribution of {pie_values}")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader('Donut Chart')
    donut_values = st.selectbox('Select column for donut chart values:', df.columns.tolist(), key='donut_values')
    donut_names = st.selectbox('Select column for donut chart names (optional):', [None] + df.columns.tolist(), key='donut_names')
    if donut_values:
        fig = px.pie(df, values=donut_values, names=donut_names, hole=0.3, title=f"Donut Chart: Distribution of {donut_values}")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader('Advanced Box Plot')
    if common_x_axis and common_y_axis:
        fig = px.box(df, x=common_x_axis, y=common_y_axis, title=f"Box Plot: {common_x_axis} vs {common_y_axis}", color_discrete_sequence=[chart_color])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader('Histogram')
    hist_col = st.selectbox('Select column for histogram:', df.columns.tolist(), key='hist_col')
    if hist_col:
        fig = px.histogram(df, x=hist_col, title=f"Histogram of {hist_col}", color_discrete_sequence=[chart_color])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader('Sunburst Chart (Hierarchical)')
    sunburst_path = st.multiselect('Select columns for sunburst path (hierarchical):', df.columns.tolist(), default=df.columns.tolist()[:2])
    sunburst_values = st.selectbox('Select values for the sunburst chart:', df.columns.tolist(), key='sunburst_values')

    # Check if the selected sunburst value column is numeric
    if sunburst_values:
        if pd.api.types.is_numeric_dtype(df[sunburst_values]):
            fig = px.sunburst(df, path=sunburst_path, values=sunburst_values, title=f"Sunburst Chart: {sunburst_values}", color_discrete_sequence=[chart_color])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please provide a numerical column for the sunburst chart values.")

    # 3D Scatter Plot
    st.subheader('3D Scatter Plot')
    z_axis = st.selectbox('Select column for z-axis:', df.columns.tolist(), key='z_axis')
    if common_x_axis and common_y_axis and z_axis:
        fig = px.scatter_3d(df, x=common_x_axis, y=common_y_axis, z=z_axis, title=f"3D Scatter Plot: {common_x_axis}, {common_y_axis}, {z_axis}", color_discrete_sequence=[chart_color])
        st.plotly_chart(fig, use_container_width=True)
    st.sidebar.header('Download Data')
    st.download_button(label='Download Filtered CSV', 
                       data=download_csv(df), 
                       file_name='filtered_data.csv', 
                       mime='text/csv')
def main():
    st.title("Automated CSV Cleaner & Visualizer | ML & DL Model Recommender App✨")

    # Initialize session state variables
    if 'df_cleaned' not in st.session_state:
        st.session_state.df_cleaned = None
    if 'feature_cols' not in st.session_state:
        st.session_state.feature_cols = []
    if 'target_col' not in st.session_state:
        st.session_state.target_col = None
    if 'best_ml_model_name' not in st.session_state:
        st.session_state.best_ml_model_name = None
    if 'best_ml_score' not in st.session_state:
        st.session_state.best_ml_score = None
    if 'best_dl_model_name' not in st.session_state:
        st.session_state.best_dl_model_name = None
    if 'best_dl_score' not in st.session_state:
        st.session_state.best_dl_score = None
    if 'chart_color' not in st.session_state:
        st.session_state.chart_color = '#1f77b4'

    uploaded_file = st.file_uploader("Upload a CSV, XLSX, or XLS file", type=['csv', 'xlsx', 'xls'])

    if uploaded_file is not None:
        try:
            # Read the uploaded file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.write("Raw Data:")
            st.write(df)

            # Sidebar for cleaning options
            st.sidebar.title('Cleaning Options')
            missing_thresh = st.sidebar.slider("Missing value threshold (%)", 0, 100, 50) / 100
            fill_num_option = st.sidebar.selectbox("Filling option for numeric columns", ['mean', 'median', 'mode'])
            fill_cat_option = st.sidebar.selectbox("Filling option for categorical columns", ['unknown', 'mode', 'none'])
            drop_outliers = st.sidebar.checkbox("Remove outliers", value=False)

            # Color picker for charts
            st.session_state.chart_color = st.sidebar.color_picker('Pick a color for all charts', st.session_state.chart_color)

            # Clean the data and store it in session state
            st.session_state.df_cleaned = clean_data(df, missing_thresh, fill_num_option, fill_cat_option, drop_outliers)

            # Display cleaned data
            st.write("Cleaned Data:")
            st.write(st.session_state.df_cleaned)

            # Assuming create_dashboard is defined elsewhere
            create_dashboard(st.session_state.df_cleaned, st.session_state.chart_color)

            # Model Recommendation Form
            with st.form("model_recommendation_form"):
                st.session_state.feature_cols = st.multiselect("Select feature columns", st.session_state.df_cleaned.columns.tolist())
                st.session_state.target_col = st.selectbox("Select target column", st.session_state.df_cleaned.columns.tolist())

                # Submit button for the form
                submit_button = st.form_submit_button("Recommend Model🤖")

                if submit_button:
                    if not st.session_state.feature_cols or not st.session_state.target_col:
                        st.warning("Please select both feature columns and the target column.")
                        return

                    # Prepare the data for model evaluation
                    X = st.session_state.df_cleaned[st.session_state.feature_cols]
                    y = st.session_state.df_cleaned[st.session_state.target_col].values

                    # Check for NaN values
                    if X.isnull().sum().sum() > 0 or pd.isnull(y).sum() > 0:
                        st.warning("There are missing values in the data. Please clean the data before proceeding.")
                        return

                    # Check the shape of X and y before proceeding
                    st.write(f"Feature matrix shape (X): {X.shape}")
                    st.write(f"Target vector shape (y): {y.shape}")

                    if X.shape[1] <= 2:
                        st.warning("Please select more than 2 features to evaluate the models.")
                        return

                    # Automatically determine task type
                    unique_classes = np.unique(y)
                    is_classification = len(unique_classes) < 20 and np.issubdtype(y.dtype, np.integer)

                    if is_classification:
                        le = LabelEncoder()
                        y = le.fit_transform(y)

                        if len(unique_classes) < 2:
                            st.warning("Not enough unique classes for classification. Please ensure the target variable has at least two distinct classes.")
                            return
                    else:
                        y = y.astype(float)

                    # Sample data for speed
                    sample_size = min(2000, X.shape[0])
                    if sample_size < 1:
                        st.warning("Sample size is too small for evaluation.")
                        return

                    # Sample data correctly and reset index
                    X_sampled = X.sample(n=sample_size, random_state=42).reset_index(drop=True)
                    y_sampled = y[X_sampled.index]  # Correct sampling logic

                    # Split the data
                    X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.2, random_state=42)
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                    # Reshape for deep learning models
                    X_train_dl = X_train.reshape(-1, X_train.shape[1], 1)
                    X_test_dl = X_test.reshape(-1, X_test.shape[1], 1)

                    st.write("Evaluating models...")

                    # Adjustable parameters
                    n_estimators = st.slider("Number of Estimators (for Tree-based models)", min_value=1, max_value=100, value=10)
                    max_depth = st.slider("Max Depth (for Tree-based models)", min_value=1, max_value=10, value=3)
                    n_neighbors = st.slider("Number of Neighbors (for KNN)", min_value=1, max_value=20, value=3)

                    params = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'n_neighbors': n_neighbors
                    }

                    # Evaluate models with try-except for error logging
                    try:
                        st.session_state.best_ml_model_name, st.session_state.best_ml_score, st.session_state.best_dl_model_name, st.session_state.best_dl_score = evaluate_models(
                            X_train_dl, y_train, X_test_dl, y_test, is_classification, params)
                    except Exception as e:
                        st.error(f"Error during model evaluation: {e}")
                        return

                    # Display results for ML models
                    if is_classification and st.session_state.best_ml_score:
                        st.success(f"The best ML model for classification is: {st.session_state.best_ml_model_name} with an Accuracy of {st.session_state.best_ml_score[0]:.4f} and ROC AUC of {st.session_state.best_ml_score[1]:.4f}", icon="✅")
                    elif st.session_state.best_ml_score:
                        st.success(f"The best ML model for regression is: {st.session_state.best_ml_model_name} with MSE: {st.session_state.best_ml_score[0]:.4f}, MAE: {st.session_state.best_ml_score[1]:.4f}, R²: {st.session_state.best_ml_score[2]:.4f}", icon="✅")

                    # Display results for DL models
                    if st.session_state.best_dl_model_name and st.session_state.best_dl_score:
                        if is_classification:
                            st.success(f"The best DL model for classification is: {st.session_state.best_dl_model_name} with an Accuracy of {st.session_state.best_dl_score[0]:.4f} and ROC AUC of {st.session_state.best_dl_score[1]:.4f}", icon="✅")
                        else:
                            st.success(f"The best DL model for regression is: {st.session_state.best_dl_model_name} with MSE: {st.session_state.best_dl_score[0]:.4f}, MAE: {st.session_state.best_dl_score[1]:.4f}, R²: {st.session_state.best_dl_score[2]:.4f}", icon="✅")
                    else:
                        st.warning("No DL models were evaluated successfully.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()