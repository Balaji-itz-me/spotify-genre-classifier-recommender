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
        # Try to get features from the scaler
        if hasattr(scaler_model, 'feature_names_in_'):
            return scaler_model.feature_names_in_.tolist()
        
        # If scaler doesn't have feature names, try to infer from dataset
        if df_sample is not None:
            # Common features that are typically used for music prediction
            potential_features = [
                'danceability', 'energy', 'loudness', 'speechiness', 
                'acousticness', 'instrumentalness', 'liveness', 'valence', 
                'tempo', 'duration_ms', 'explicit', 'key', 'mode', 
                'time_signature', 'cluster'
            ]
            
            # Return only features that exist in the dataset
            available_features = [f for f in potential_features if f in df_sample.columns]
            return available_features[:15]  # Limit to 15 features as expected by model
        
        # Fallback: Most common 15 features for music analysis
        return [
            'danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 
            'tempo', 'duration_ms', 'explicit', 'key', 'mode', 
            'time_signature', 'cluster'
        ]
        
    except Exception as e:
        st.warning(f"Could not determine model features: {e}")
        # Return exactly 15 most important features
        return [
            'danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 
            'tempo', 'duration_ms', 'key', 'mode', 'time_signature', 
            'explicit', 'cluster'
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
        'cluster': 0,
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
    
    # Build complete feature vector - ONLY include required features
    complete_features = {}
    
    for feature in required_features:
        if feature in user_inputs:
            complete_features[feature] = [user_inputs[feature]]
        elif feature in defaults:
            complete_features[feature] = [defaults[feature]]
        else:
            complete_features[feature] = [0.0]  # Last resort default
    
    # Create DataFrame and ensure we have exactly the right number of features
    df = pd.DataFrame(complete_features)
    
    # Debug info
    st.sidebar.info(f"Features created: {len(df.columns)}")
    st.sidebar.info(f"Expected: {len(required_features)}")
    
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

def popularity_module():
    """Popularity prediction module"""
    st.header("🎯 Popularity Prediction")
    
    if 'models' not in st.session_state or 'popularity' not in st.session_state.models:
        st.error("Popularity model not available")
        return
    
    user_inputs = create_feature_inputs()
    
    if st.button("Predict Popularity", type="primary"):
        try:
            models = st.session_state.models
            df_sample = st.session_state.df if 'df' in st.session_state else None
            
            # Get required features for the model
            required_features = get_model_features(models['scaler'], df_sample)
            
            # Ensure we have exactly 15 features (as expected by the model)
            if len(required_features) != 15:
                st.warning(f"Adjusting features: got {len(required_features)}, need 15")
                required_features = required_features[:15]  # Take first 15
            
            # Create complete feature vector with defaults
            features_df = create_complete_feature_vector(user_inputs, required_features, df_sample)
            
            # Verify feature count before prediction
            if features_df.shape[1] != 15:
                st.error(f"Feature mismatch: created {features_df.shape[1]} features, model expects 15")
                return
            
            # Scale and predict
            scaled = models['scaler'].transform(features_df)
            
            # Verify scaled features
            if scaled.shape[1] != 15:
                st.error(f"Scaled feature mismatch: got {scaled.shape[1]}, expected 15")
                return
                
            prediction = models['popularity'].predict(scaled)[0]
            
            st.success(f"🎵 Predicted Popularity: **{prediction:.1f}/100**")
            
            if prediction >= 70:
                st.info("🔥 High popularity potential!")
            elif prediction >= 50:
                st.info("📈 Moderate popularity")
            else:
                st.info("📊 Niche appeal")
                
            # Show which features were used
            with st.expander("🔍 Features Used"):
                st.write(f"Using exactly {len(required_features)} features:")
                for i, feature in enumerate(required_features, 1):
                    st.write(f"{i}. {feature}")
                
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            with st.expander("Error Details"):
                st.code(str(e))
                st.write("Debug info:")
                try:
                    st.write(f"User inputs: {len(user_inputs)} features")
                    st.write(f"Required features: {len(required_features) if 'required_features' in locals() else 'Unknown'}")
                    st.write(f"Feature DF shape: {features_df.shape if 'features_df' in locals() else 'Not created'}")
                except:
                    pass

def genre_module():
    """Genre classification module"""
    st.header("🎼 Genre Classification")
    
    if 'models' not in st.session_state or 'super_genre' not in st.session_state.models:
        st.error("Genre model not available")
        return
    
    user_inputs = create_feature_inputs()
    
    if st.button("Classify Genre", type="primary"):
        try:
            models = st.session_state.models
            df_sample = st.session_state.df if 'df' in st.session_state else None
            
            # Get required features for the model
            required_features = get_model_features(models['scaler'], df_sample)
            
            # Ensure exactly 15 features
            if len(required_features) != 15:
                required_features = required_features[:15]
            
            # Create complete feature vector with defaults
            features_df = create_complete_feature_vector(user_inputs, required_features, df_sample)
            
            # Verify feature count
            if features_df.shape[1] != 15:
                st.error(f"Feature count mismatch: {features_df.shape[1]} != 15")
                return
            
            # Scale and predict
            scaled = models['scaler'].transform(features_df)
            genre = models['super_genre'].predict(scaled)[0]
            
            st.success(f"🎵 Predicted Genre: **{genre.upper()}**")
            
            # Show confidence if available
            try:
                proba = models['super_genre'].predict_proba(scaled)[0]
                confidence = proba.max()
                st.info(f"Confidence: {confidence:.1%}")
            except:
                pass
                
        except Exception as e:
            st.error(f"Classification failed: {e}")
            with st.expander("Error Details"):
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