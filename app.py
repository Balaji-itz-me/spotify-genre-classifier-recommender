import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import gdown
import zipfile
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="Music Analysis Dashboard",
    page_icon="🎵",
    layout="wide"
)

# Constants
DRIVE_FILE_ID = "12aGkuSuDpVO_gLdcal-fyUJYLGqcPFyI"
AUDIO_FEATURES = ['danceability', 'energy', 'loudness', 'speechiness', 
                  'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

@st.cache_resource
def download_files():
    """Download and extract files - simplified"""
    try:
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "music_files.zip")
        
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        gdown.download(url, zip_path, quiet=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find data directory
        for root, dirs, files in os.walk(temp_dir):
            if 'df_merged.pkl' in files:
                return root
        
        return temp_dir
        
    except Exception as e:
        st.error(f"Download failed: {e}")
        return None

@st.cache_data
def load_data():
    """Load only essential data"""
    base_dir = download_files()
    if not base_dir:
        return None, None
    
    try:
        # Load main dataset
        df_path = os.path.join(base_dir, 'df_merged.pkl')
        df = pd.read_pickle(df_path) if os.path.exists(df_path) else None
        
        # Load feature matrix (optional)
        matrix_path = os.path.join(base_dir, 'feature_matrix.npy')
        features = np.load(matrix_path) if os.path.exists(matrix_path) else None
        
        return df, features
        
    except Exception as e:
        st.error(f"Data loading failed: {e}")
        return None, None

@st.cache_resource
def load_models():
    """Load only essential models"""
    base_dir = download_files()
    if not base_dir:
        return {}
    
    models = {}
    
    # Essential models only
    model_files = {
        'popularity': 'rf_popularity_cluster.pkl',
        'scaler': 'hierarchical_scaler.pkl',
        'super_genre': 'super_genre_model.pkl'
    }
    
    for name, filename in model_files.items():
        path = os.path.join(base_dir, filename)
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
            except:
                pass
    
    return models

def get_model_features(scaler_model, df_sample=None):
    """Get the exact feature names that the model expects"""
    try:
        # Try to get features from the scaler first
        if hasattr(scaler_model, 'feature_names_in_'):
            features = scaler_model.feature_names_in_.tolist()
            st.sidebar.info(f"Got {len(features)} features from scaler")
            return features
        
        # If no feature names in scaler, try from the dataset
        if df_sample is not None:
            # These are the most common features for music analysis
            # Including 'cluster' which your model clearly expects
            potential_features = [
                'danceability', 'energy', 'loudness', 'speechiness', 
                'acousticness', 'instrumentalness', 'liveness', 'valence', 
                'tempo', 'duration_ms', 'explicit', 'key', 'mode', 
                'time_signature', 'cluster'  # Make sure cluster is included
            ]
            
            # Return only features that exist in the dataset, but prioritize 'cluster'
            available_features = []
            
            # First, add cluster if it exists (it's clearly required)
            if 'cluster' in df_sample.columns:
                available_features.append('cluster')
            
            # Then add other features
            for feature in potential_features:
                if feature in df_sample.columns and feature not in available_features:
                    available_features.append(feature)
            
            # If we don't have enough features, add cluster anyway (with default value)
            if len(available_features) < 15 and 'cluster' not in available_features:
                available_features.append('cluster')
            
            return available_features[:15]  # Return exactly 15 features
        
        # Fallback: The exact 15 features your model expects (based on error message)
        return [
            'danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 
            'tempo', 'duration_ms', 'explicit', 'key', 'mode', 
            'time_signature', 'cluster'  # cluster MUST be included
        ]
        
    except Exception as e:
        st.warning(f"Could not determine model features: {e}")
        # Return exactly 15 features with cluster included
        return [
            'danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 
            'tempo', 'duration_ms', 'key', 'mode', 'time_signature', 
            'explicit', 'cluster'  # cluster is essential
        ]

def create_complete_feature_vector(user_inputs, required_features, df_sample=None):
    """Create a complete feature vector with defaults for missing features"""
    
    # Default values for common features
    defaults = {
        'duration_ms': 200000,
        'explicit': 0,
        'key': 5,
        'mode': 1,
        'time_signature': 4,
        'cluster': 0,  # This is crucial - your model expects this feature
        'popularity': 50,
    }
    
    # If we have a sample dataset, use median values as defaults
    if df_sample is not None:
        for feature in required_features:
            if feature in df_sample.columns and feature not in user_inputs:
                try:
                    defaults[feature] = float(df_sample[feature].median())
                except:
                    pass
    
    # Build complete feature vector - ONLY include required features IN THE CORRECT ORDER
    complete_features = {}
    
    for feature in required_features:
        if feature in user_inputs:
            complete_features[feature] = user_inputs[feature]
        elif feature in defaults:
            complete_features[feature] = defaults[feature]
        else:
            complete_features[feature] = 0.0  # Last resort default
    
    # Create DataFrame with the EXACT column names the model expects
    df = pd.DataFrame([complete_features])  # Note: wrap in list to create single row
    
    # Ensure columns are in the same order as required_features
    df = df[required_features]
    
    # Debug info
    st.sidebar.info(f"Features created: {len(df.columns)}")
    st.sidebar.info(f"Expected: {len(required_features)}")
    st.sidebar.info(f"Column names match: {list(df.columns) == required_features}")
    
    # Show which features we're using
    with st.sidebar.expander("Feature Details"):
        for feature in required_features:
            value = complete_features.get(feature, 'Missing')
            source = "User" if feature in user_inputs else "Default"
            st.write(f"{feature}: {value} ({source})")
    
    return df
def create_feature_inputs():
    """Create audio feature input widgets - simplified"""
    st.subheader("🎛️ Set Audio Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
        energy = st.slider("Energy", 0.0, 1.0, 0.5)
        speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1)
    
    with col2:
        acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.1)
        liveness = st.slider("Liveness", 0.0, 1.0, 0.2)
    
    with col3:
        valence = st.slider("Valence", 0.0, 1.0, 0.5)
        tempo = st.slider("Tempo", 50.0, 200.0, 120.0)
        loudness = st.slider("Loudness", -30.0, 0.0, -10.0)
    
    # Return just the user inputs
    return {
        'danceability': danceability,
        'energy': energy, 
        'loudness': loudness,
        'speechiness': speechiness,
        'acousticness': acousticness,
        'instrumentalness': instrumentalness,
        'liveness': liveness,
        'valence': valence,
        'tempo': tempo
    }

def debug_model_features(models, df_sample=None):
    """Debug what features the model actually expects"""
    st.subheader("🔍 Model Feature Debugging")
    
    # Check scaler features
    if 'scaler' in models:
        scaler = models['scaler']
        st.write("**Scaler Information:**")
        
        if hasattr(scaler, 'feature_names_in_'):
            st.write(f"✅ Scaler has feature_names_in_: {len(scaler.feature_names_in_)} features")
            st.write("Features expected by scaler:")
            for i, feature in enumerate(scaler.feature_names_in_, 1):
                st.write(f"  {i}. {feature}")
            return scaler.feature_names_in_.tolist()
        else:
            st.write("❌ Scaler doesn't have feature_names_in_")
    
    # Check dataset features
    if df_sample is not None:
        st.write("**Dataset Features:**")
        st.write(f"Dataset has {len(df_sample.columns)} features")
        numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
        st.write(f"Numeric columns: {len(numeric_cols)}")
        
        # Show first few numeric columns
        for col in numeric_cols[:20]:  # Show first 20
            st.write(f"  - {col}")
    
    return None

def get_exact_model_features(models, df_sample=None):
    """Get the EXACT features the scaler expects (including target variables for scaling)"""
    
    scaler_features = None
    if 'scaler' in models and hasattr(models['scaler'], 'feature_names_in_'):
        scaler_features = models['scaler'].feature_names_in_.tolist()
        st.info(f"Scaler was trained with {len(scaler_features)} features")
        st.write(f"Scaler features: {scaler_features}")
        return scaler_features  # Return ALL features the scaler needs
    
    # Fallback: Include popularity because the scaler expects it
    return [
        'danceability', 'energy', 'loudness', 'speechiness', 
        'acousticness', 'instrumentalness', 'liveness', 'valence', 
        'tempo', 'duration_ms', 'key', 'mode', 'time_signature', 
        'explicit', 'cluster', 'popularity'  # Include popularity for scaler
    ]

def create_feature_dataframe(user_inputs, required_features, df_sample=None):
    """Create feature DataFrame with EXACT column names and order (including targets for scaler)"""
    
    # Default values for missing features (including targets for scaler compatibility)
    defaults = {
        'duration_ms': 200000,
        'explicit': 0,
        'key': 5,
        'mode': 1,
        'time_signature': 4,
        'cluster': 0,
        'popularity': 50,  # Include default popularity for scaler (will be removed before model)
    }
    
    # Get median values from dataset if available
    if df_sample is not None:
        for feature in required_features:
            if feature in df_sample.columns and feature not in user_inputs:
                try:
                    median_val = df_sample[feature].median()
                    if pd.notna(median_val):
                        defaults[feature] = float(median_val)
                except:
                    pass
    
    # Create feature dictionary in EXACT order (including all features scaler expects)
    feature_dict = {}
    for feature_name in required_features:
        if feature_name in user_inputs:
            feature_dict[feature_name] = user_inputs[feature_name]
        elif feature_name in defaults:
            feature_dict[feature_name] = defaults[feature_name]
        else:
            feature_dict[feature_name] = 0.0
    
    # Create DataFrame - CRITICAL: column names must match scaler exactly
    df = pd.DataFrame([feature_dict])
    
    # Ensure column order matches exactly what scaler expects
    df = df[required_features]
    
    return df

def safe_predict_popularity():
    """Safe popularity prediction with proper debugging"""
    st.header("🎯 Popularity Prediction (Debug Mode)")
    
    if 'models' not in st.session_state or 'popularity' not in st.session_state.models:
        st.error("Popularity model not available")
        return
    
    # Debug section
    with st.expander("🔍 Debug Model Features", expanded=True):
        models = st.session_state.models
        df_sample = st.session_state.df if 'df' in st.session_state else None
        
        expected_features = debug_model_features(models, df_sample)
        
        if expected_features:
            st.success(f"✅ Found exact features: {len(expected_features)}")
            scaler_features = expected_features
        else:
            st.warning("⚠️ Using inferred features")
            scaler_features = get_exact_model_features(models, df_sample)
        
        st.write(f"**Scaler expects these {len(scaler_features)} features:**")
        for i, feat in enumerate(scaler_features, 1):
            st.write(f"{i}. {feat}")
    
    # Get user inputs
    user_inputs = create_feature_inputs()
    
    if st.button("🎯 Predict Popularity (Debug)", type="primary"):
        try:
            models = st.session_state.models
            df_sample = st.session_state.df if 'df' in st.session_state else None
            
            # Get ALL features that scaler expects (including targets)
            scaler_features = get_exact_model_features(models, df_sample)
            
            st.info(f"Creating DataFrame with {len(scaler_features)} features for scaler")
            
            # Create feature DataFrame with ALL features scaler needs
            features_df = create_feature_dataframe(user_inputs, scaler_features, df_sample)
            
            st.write("**Created DataFrame for Scaler:**")
            st.write(f"Shape: {features_df.shape}")
            st.write(f"Columns ({len(features_df.columns)}): {list(features_df.columns)}")
            st.dataframe(features_df)
            
            # Scale ALL features (including target)
            scaled_features = models['scaler'].transform(features_df)
            st.write(f"Scaled features shape: {scaled_features.shape}")
            
            # Now REMOVE the target variable columns from scaled features for model prediction
            target_indices = []
            for i, feat in enumerate(scaler_features):
                if feat in ['popularity']:  # Add other targets if needed
                    target_indices.append(i)
            
            # Remove target columns from scaled features
            if target_indices:
                st.info(f"Removing target columns at indices: {target_indices}")
                model_features = np.delete(scaled_features, target_indices, axis=1)
                st.write(f"Model features shape after removing targets: {model_features.shape}")
            else:
                model_features = scaled_features
                st.warning("No target columns found to remove")
            
            # CRITICAL: Verify model features has exactly 15 columns
            if model_features.shape[1] != 15:
                st.error(f"❌ Model features has {model_features.shape[1]} columns, model expects 15")
                return
            
            # Now predict with the correct number of features
            prediction = models['popularity'].predict(model_features)[0]
            
            st.success(f"🎵 Predicted Popularity: **{prediction:.1f}/100**")
            
            # Interpretation
            if prediction >= 70:
                st.info("🔥 High popularity potential!")
            elif prediction >= 50:
                st.info("📈 Moderate popularity")
            else:
                st.info("📊 Niche appeal")
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.exception(e)

# Add this to your main function to replace popularity_module
def popularity_module():
    """Enhanced popularity module with debugging"""
    safe_predict_popularity()
def genre_module():
    """Genre classification module with proper cluster handling"""
    st.header("🎼 Genre Classification")
    
    if 'models' not in st.session_state or 'super_genre' not in st.session_state.models:
        st.error("Genre model not available")
        return
    
    # Get user inputs for audio features
    user_inputs = create_feature_inputs()
    
    # Add cluster selection for user
    st.subheader("🎯 Additional Settings")
    cluster = st.selectbox(
        "Music Cluster (style/pattern group):",
        options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        index=0,
        help="Different music clusters represent different musical patterns and styles"
    )
    
    # Add cluster to user inputs
    user_inputs['cluster'] = cluster
    
    if st.button("🎼 Classify Genre", type="primary"):
        try:
            models = st.session_state.models
            df_sample = st.session_state.df if 'df' in st.session_state else None
            
            # Get the exact features the scaler expects
            if 'scaler' in models and hasattr(models['scaler'], 'feature_names_in_'):
                required_features = models['scaler'].feature_names_in_.tolist()
                st.info(f"Model expects {len(required_features)} features from scaler")
            else:
                # Fallback - include cluster which is essential
                required_features = [
                    'danceability', 'energy', 'loudness', 'speechiness', 
                    'acousticness', 'instrumentalness', 'liveness', 'valence', 
                    'tempo', 'duration_ms', 'key', 'mode', 'time_signature', 
                    'explicit', 'cluster'
                ]
                st.info(f"Using fallback features: {len(required_features)} features")
            
            # Ensure cluster is in required features (it's essential for your model)
            if 'cluster' not in required_features:
                required_features.append('cluster')
                st.info("Added 'cluster' to required features")
            
            # Create complete feature vector
            features_df = create_feature_dataframe(user_inputs, required_features, df_sample)
            
            # Debug: Show what we're sending to the model
            with st.expander("🔍 Feature Details"):
                st.write(f"Created DataFrame shape: {features_df.shape}")
                st.write("Features being used:")
                for col in features_df.columns:
                    source = "User Input" if col in user_inputs else "Default"
                    st.write(f"- {col}: {features_df[col].iloc[0]} ({source})")
            
            # Scale features first
            scaled_features = models['scaler'].transform(features_df)
            st.info(f"Scaled features shape: {scaled_features.shape}")
            
            # If scaler includes target variables, remove them for model prediction
            # For genre classification, we might need to remove 'super_genre' or 'popularity' if they were in training
            target_columns = ['popularity', 'super_genre']  # Common target columns
            target_indices = []
            
            for i, feat in enumerate(required_features):
                if feat in target_columns:
                    target_indices.append(i)
            
            # Remove target columns if they exist
            if target_indices:
                model_features = np.delete(scaled_features, target_indices, axis=1)
                st.info(f"Removed {len(target_indices)} target columns for model prediction")
                st.info(f"Final model features shape: {model_features.shape}")
            else:
                model_features = scaled_features
                st.info("No target columns found to remove")
            
            # Predict genre
            genre_prediction = models['super_genre'].predict(model_features)[0]
            
            # Display prediction
            col1, col2, col3 = st.columns(3)
            
            with col2:  # Center the result
                st.success(f"🎵 **Predicted Genre: {genre_prediction.upper()}**")
            
            # Show prediction confidence/probability if available
            try:
                probabilities = models['super_genre'].predict_proba(model_features)[0]
                
                if hasattr(models['super_genre'], 'classes_'):
                    classes = models['super_genre'].classes_
                    
                    # Create probability dataframe
                    prob_df = pd.DataFrame({
                        'Genre': classes,
                        'Probability': probabilities
                    }).sort_values('Probability', ascending=False)
                    
                    st.subheader("🎯 Genre Probabilities")
                    
                    # Show top 5 predictions
                    top_5 = prob_df.head(5)
                    
                    for idx, row in top_5.iterrows():
                        confidence = row['Probability']
                        genre = row['Genre']
                        
                        if confidence > 0.1:  # Only show if probability > 10%
                            st.progress(confidence, text=f"{genre}: {confidence:.1%}")
                    
                    # Show confidence for the top prediction
                    max_confidence = probabilities.max()
                    if max_confidence >= 0.7:
                        st.success(f"🎯 High confidence: {max_confidence:.1%}")
                    elif max_confidence >= 0.4:
                        st.info(f"📊 Moderate confidence: {max_confidence:.1%}")
                    else:
                        st.warning(f"🤔 Low confidence: {max_confidence:.1%} - The track might be a genre blend")
                
            except Exception as prob_error:
                st.info("Confidence scores not available")
            
            # Create a simple feature visualization
            try:
                audio_features_only = ['danceability', 'energy', 'speechiness', 
                                     'acousticness', 'instrumentalness', 'liveness', 'valence']
                
                feature_values = []
                feature_names = []
                
                for feature in audio_features_only:
                    if feature in user_inputs:
                        feature_values.append(user_inputs[feature])
                        feature_names.append(feature.capitalize())
                
                if feature_values and len(feature_values) >= 3:
                    import plotly.graph_objects as go
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=feature_values,
                        theta=feature_names,
                        fill='toself',
                        name=f'{genre_prediction} Profile',
                        line_color='rgb(32, 201, 151)'
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )),
                        showlegend=True,
                        title=f"Audio Profile for {genre_prediction} Classification"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            except Exception as viz_error:
                st.info("Feature visualization not available")
                
        except Exception as e:
            st.error(f"❌ Genre classification failed: {str(e)}")
            
            # Better error handling
            if "feature names should match" in str(e).lower():
                st.error("🔧 **Feature Mismatch Error**")
                st.info("The genre model was trained with different features. Make sure 'cluster' is included.")
                
                # Show what features the model expects vs what we provided
                if 'scaler' in st.session_state.models:
                    scaler = st.session_state.models['scaler']
                    if hasattr(scaler, 'feature_names_in_'):
                        st.write("**Model expects these features:**")
                        expected_features = scaler.feature_names_in_
                        for i, feat in enumerate(expected_features):
                            st.write(f"{i+1}. {feat}")
                        
                        # Show what we tried to provide
                        provided_features = list(user_inputs.keys()) + ['cluster']
                        st.write("**We provided these features:**")
                        for i, feat in enumerate(provided_features):
                            st.write(f"{i+1}. {feat}")
                            
                        # Show missing features
                        missing = set(expected_features) - set(provided_features)
                        if missing:
                            st.write(f"**Missing features:** {list(missing)}")
                        
                        extra = set(provided_features) - set(expected_features)
                        if extra:
                            st.write(f"**Extra features:** {list(extra)}")
            
            elif "input contains nan" in str(e).lower():
                st.error("📊 **Data Error**: Some features contain invalid values")
            else:
                with st.expander("🔍 Technical Details"):
                    st.code(str(e))

def recommendation_module():
    """Simple recommendation module"""
    st.header("🎵 Music Recommendations")
    
    if 'df' not in st.session_state or st.session_state.df is None:
        st.error("Dataset not available")
        return
    
    df = st.session_state.df
    
    # Simple search
    search = st.text_input("Search for a track or artist:")
    
    if search:
        # Filter tracks
        mask = (df['track_name'].str.contains(search, case=False, na=False) |
               df['main_artist'].str.contains(search, case=False, na=False))
        results = df[mask].head(10)
        
        if len(results) > 0:
            st.subheader("Search Results:")
            
            for idx, row in results.iterrows():
                col1, col2, col3 = st.columns([3, 2, 1])
                
                with col1:
                    st.write(f"**{row.get('track_name', 'Unknown')}**")
                with col2:
                    st.write(row.get('main_artist', 'Unknown'))
                with col3:
                    if st.button(f"Get Recs", key=f"rec_{idx}"):
                        # Simple genre-based recommendations
                        genre = row.get('super_genre', '')
                        same_genre = df[df['super_genre'] == genre]
                        recs = same_genre.sample(min(5, len(same_genre)))
                        
                        st.subheader(f"Similar {genre} tracks:")
                        for _, rec in recs.iterrows():
                            st.write(f"• {rec.get('track_name', 'Unknown')} - {rec.get('main_artist', 'Unknown')}")
        else:
            st.info("No tracks found")

def dataset_overview():
    """Simple dataset overview"""
    st.header("📊 Dataset Overview")
    
    if 'df' not in st.session_state or st.session_state.df is None:
        st.error("Dataset not available")
        return
    
    df = st.session_state.df
    
    # Basic stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Tracks", f"{len(df):,}")
    with col2:
        st.metric("Artists", f"{df['main_artist'].nunique():,}")
    with col3:
        st.metric("Genres", f"{df['super_genre'].nunique()}")
    
    # Simple genre chart
    if st.checkbox("Show Genre Distribution"):
        genre_counts = df['super_genre'].value_counts().head(10)
        fig = px.bar(x=genre_counts.values, y=genre_counts.index, 
                     orientation='h', title="Top Genres")
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application - simplified"""
    st.title("🎵 Music Analysis Dashboard")
    
    # Initialize session state
    if 'initialized' not in st.session_state:
        with st.spinner("Loading..."):
            df, features = load_data()
            models = load_models()
            
            st.session_state.df = df
            st.session_state.features = features
            st.session_state.models = models
            st.session_state.initialized = True
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    mode = st.sidebar.selectbox(
        "Choose Function:",
        ["Dataset Overview", "Popularity Prediction", "Genre Classification", "Recommendations"]
    )
    
    # Status
    st.sidebar.markdown("---")
    if st.session_state.df is not None:
        st.sidebar.success(f"✅ {len(st.session_state.df):,} tracks loaded")
    else:
        st.sidebar.error("❌ Data not loaded")
    
    if st.session_state.models:
        st.sidebar.success(f"✅ {len(st.session_state.models)} models loaded")
    else:
        st.sidebar.error("❌ Models not loaded")
    
    # Show selected module
    if mode == "Dataset Overview":
        dataset_overview()
    elif mode == "Popularity Prediction":
        popularity_module()
    elif mode == "Genre Classification":
        genre_module()
    elif mode == "Recommendations":
        recommendation_module()

if __name__ == "__main__":

    main()





