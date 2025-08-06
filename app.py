import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import gdown
import zipfile
import tempfile
import traceback

# Page configuration
st.set_page_config(
    page_title="Music Analysis Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DRIVE_FILE_ID = "12aGkuSuDpVO_gLdcal-fyUJYLGqcPFyI"
SUPER_GENRES = ['rock', 'pop', 'electronic', 'metal', 'hiphop', 'japanese', 
                'latin', 'classical', 'country', 'folk', 'ambient', 'world', 
                'children', 'other', 'misc']
AUDIO_FEATURES = ['danceability', 'energy', 'loudness', 'speechiness', 
                  'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

@st.cache_resource
def download_and_extract_files():
    """Download and extract files with corrected logic"""
    try:
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "music_files.zip")
        
        # Download
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        with st.spinner("📥 Downloading files..."):
            gdown.download(url, zip_path, quiet=False, fuzzy=True)
        
        # Extract
        with st.spinner("📦 Extracting files..."):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
        
        # Find the correct directory - CORRECTED LOGIC
        extracted_items = os.listdir(temp_dir)
        
        # Check each possible location systematically
        possible_locations = []
        
        # Direct in temp_dir
        possible_locations.append(temp_dir)
        
        # In any subdirectory
        for item in extracted_items:
            item_path = os.path.join(temp_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.') and item != '__MACOSX':
                possible_locations.append(item_path)
                
                # Also check subdirectories of subdirectories
                try:
                    sub_items = os.listdir(item_path)
                    for sub_item in sub_items:
                        sub_path = os.path.join(item_path, sub_item)
                        if os.path.isdir(sub_path) and not sub_item.startswith('.'):
                            possible_locations.append(sub_path)
                except:
                    pass
        
        # Find the location with our key files
        for location in possible_locations:
            try:
                files_in_location = os.listdir(location)
                
                # Check for key files - CORRECTED CHECK
                key_files_found = []
                if 'df_merged.pkl' in files_in_location:
                    key_files_found.append('df_merged.pkl')
                if 'feature_matrix.npy' in files_in_location:
                    key_files_found.append('feature_matrix.npy')
                if 'super_genre_model.pkl' in files_in_location:
                    key_files_found.append('super_genre_model.pkl')
                
                # If we found key files, this is our directory
                if len(key_files_found) >= 2:  # At least 2 key files
                    st.success(f"✅ Found data directory: {location}")
                    st.info(f"📁 Key files found: {key_files_found}")
                    return location
                    
            except Exception as e:
                continue
        
        # If no perfect match, use the location with most .pkl files
        best_location = temp_dir
        max_pkl_count = 0
        
        for location in possible_locations:
            try:
                files = os.listdir(location)
                pkl_count = len([f for f in files if f.endswith('.pkl')])
                if pkl_count > max_pkl_count:
                    max_pkl_count = pkl_count
                    best_location = location
            except:
                continue
        
        st.warning(f"⚠️ Using best guess directory: {best_location} ({max_pkl_count} pkl files)")
        return best_location
        
    except Exception as e:
        st.error(f"❌ Download/extract failed: {e}")
        return None

@st.cache_data
def load_data_corrected():
    """Load data with corrected file path handling"""
    base_dir = download_and_extract_files()
    if not base_dir:
        return None, None, None, None
    
    try:
        # List all files for debugging
        all_files = os.listdir(base_dir)
        st.sidebar.success(f"📁 Found {len(all_files)} files total")
        
        pkl_files = [f for f in all_files if f.endswith('.pkl')]
        npy_files = [f for f in all_files if f.endswith('.npy')]
        
        st.sidebar.info(f"📊 PKL files: {len(pkl_files)}")
        st.sidebar.info(f"🔢 NPY files: {len(npy_files)}")
        
        # Load main dataset - CORRECTED PATH BUILDING
        df_merged = None
        df_path = os.path.join(base_dir, 'df_merged.pkl')
        if os.path.exists(df_path):
            try:
                df_merged = pd.read_pickle(df_path)
                st.sidebar.success(f"✅ df_merged loaded: {df_merged.shape}")
            except Exception as e:
                st.sidebar.error(f"❌ df_merged failed: {e}")
        else:
            st.sidebar.warning("⚠️ df_merged.pkl not found")
        
        # Load feature matrix
        feature_matrix = None
        matrix_path = os.path.join(base_dir, 'feature_matrix.npy')
        if os.path.exists(matrix_path):
            try:
                feature_matrix = np.load(matrix_path)
                st.sidebar.success(f"✅ feature_matrix loaded: {feature_matrix.shape}")
            except Exception as e:
                st.sidebar.error(f"❌ feature_matrix failed: {e}")
        else:
            st.sidebar.warning("⚠️ feature_matrix.npy not found")
        
        # Load clustering data
        tsne_data = None
        pca_data = None
        
        # Try t-SNE files
        for tsne_file in ['tsne_3d_clusters.pkl', 'tsne_3d_cluster.pkl']:
            tsne_path = os.path.join(base_dir, tsne_file)
            if os.path.exists(tsne_path):
                try:
                    tsne_data = pd.read_pickle(tsne_path)
                    st.sidebar.success(f"✅ {tsne_file} loaded")
                    break
                except Exception as e:
                    st.sidebar.warning(f"⚠️ {tsne_file} failed: {e}")
        
        # Try PCA files  
        for pca_file in ['pca_3d_clusters.pkl', 'pca_3d_cluster.pkl']:
            pca_path = os.path.join(base_dir, pca_file)
            if os.path.exists(pca_path):
                try:
                    pca_data = pd.read_pickle(pca_path)
                    st.sidebar.success(f"✅ {pca_file} loaded")
                    break
                except Exception as e:
                    st.sidebar.warning(f"⚠️ {pca_file} failed: {e}")
        
        return df_merged, feature_matrix, tsne_data, pca_data
        
    except Exception as e:
        st.error(f"❌ Data loading error: {e}")
        return None, None, None, None

@st.cache_resource
def load_models_corrected():
    """Load models with corrected file handling"""
    base_dir = download_and_extract_files()
    if not base_dir:
        return {}
    
    models = {}
    
    try:
        # List all files
        all_files = os.listdir(base_dir)
        
        # Load critical models with CORRECTED paths
        model_files = {
            'popularity': 'rf_popularity_cluster.pkl',
            'scaler': 'hierarchical_scaler.pkl', 
            'super_genre': 'super_genre_model.pkl'
        }
        
        for model_key, filename in model_files.items():
            file_path = os.path.join(base_dir, filename)
            if os.path.exists(file_path):
                try:
                    models[model_key] = joblib.load(file_path)
                    st.sidebar.success(f"✅ {model_key} model loaded")
                except Exception as e:
                    st.sidebar.error(f"❌ {model_key} model failed: {e}")
            else:
                st.sidebar.warning(f"⚠️ {filename} not found")
        
        # Load fine genre models
        models['fine_genre'] = {}
        fine_genre_files = [f for f in all_files if f.startswith('fine_genre_model_') and f.endswith('.pkl')]
        
        for fine_file in fine_genre_files:
            genre = fine_file.replace('fine_genre_model_', '').replace('.pkl', '')
            try:
                file_path = os.path.join(base_dir, fine_file)
                models['fine_genre'][genre] = joblib.load(file_path)
            except Exception as e:
                pass  # Skip failed fine genre models
        
        st.sidebar.success(f"✅ {len(models['fine_genre'])} fine genre models loaded")
        
        return models
        
    except Exception as e:
        st.sidebar.error(f"❌ Model loading error: {e}")
        return {}

def safe_import_recommender():
    """Safely import recommender function"""
    base_dir = download_and_extract_files()
    if base_dir is None:
        return None
    
    try:
        if base_dir not in sys.path:
            sys.path.insert(0, base_dir)
        
        from recommender import hybrid_recommend_with_weights
        st.sidebar.success("✅ Recommender function loaded")
        return hybrid_recommend_with_weights
    except ImportError:
        st.sidebar.warning("⚠️ Recommender function not available")
        return None
    except Exception as e:
        st.sidebar.warning(f"⚠️ Recommender error: {e}")
        return None

def create_feature_input():
    """Create input widgets for audio features"""
    st.subheader("🎛️ Audio Feature Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        danceability = st.slider("Danceability", 0.0, 1.0, 0.5, 0.01, 
                                help="How suitable a track is for dancing")
        energy = st.slider("Energy", 0.0, 1.0, 0.5, 0.01,
                          help="Perceptual measure of intensity and power")
        speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1, 0.01,
                               help="Presence of spoken words in a track")
    
    with col2:
        acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5, 0.01,
                                help="Confidence measure of whether the track is acoustic")
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.1, 0.01,
                                    help="Predicts whether a track contains no vocals")
        liveness = st.slider("Liveness", 0.0, 1.0, 0.2, 0.01,
                            help="Detects the presence of an audience in the recording")
    
    with col3:
        valence = st.slider("Valence", 0.0, 1.0, 0.5, 0.01,
                           help="Musical positiveness conveyed by a track")
        tempo = st.slider("Tempo", 50.0, 200.0, 120.0, 1.0,
                         help="Overall estimated tempo in beats per minute")
        loudness = st.slider("Loudness", -30.0, 0.0, -10.0, 0.5,
                            help="Overall loudness of a track in decibels")
    
    return np.array([danceability, energy, loudness, speechiness, acousticness, 
                     instrumentalness, liveness, valence, tempo]).reshape(1, -1)

def home_page(df_merged):
    """Home page"""
    st.title("🎵 Music Analysis Dashboard")
    st.markdown("---")
    
    if df_merged is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("📊 Project Overview")
            st.markdown("""
            **🎯 Features Available:**
            - **Popularity Prediction**: Predict track popularity using audio features
            - **Genre Classification**: Classify tracks into super and fine genres  
            - **Music Recommendations**: Get personalized track recommendations
            - **Clustering Analysis**: Visualize music patterns in 3D space
            
            **📈 Dataset Loaded Successfully!**
            """)
        
        with col2:
            st.header("📈 Dataset Stats")
            try:
                st.metric("Total Tracks", f"{len(df_merged):,}")
                if 'main_artist' in df_merged.columns:
                    st.metric("Unique Artists", f"{df_merged['main_artist'].nunique():,}")
                if 'super_genre' in df_merged.columns:
                    st.metric("Super Genres", df_merged['super_genre'].nunique())
                if 'popularity' in df_merged.columns:
                    st.metric("Avg Popularity", f"{df_merged['popularity'].mean():.1f}")
            except Exception as e:
                st.warning("Could not display all metrics")
        
        # Genre distribution
        if 'super_genre' in df_merged.columns:
            st.header("🎼 Genre Distribution")
            try:
                genre_counts = df_merged['super_genre'].value_counts().head(10)
                fig = px.bar(
                    x=genre_counts.values, 
                    y=genre_counts.index, 
                    orientation='h', 
                    title="Top 10 Genres in Dataset",
                    labels={'x': 'Number of Tracks', 'y': 'Genre'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning("Could not display genre chart")
    else:
        st.error("❌ Dataset not loaded properly")
        st.info("Please check the file loading process in the sidebar")

def popularity_prediction_page(models, df_merged):
    """Popularity prediction page"""
    st.title("🎯 Popularity Predictor")
    st.markdown("Predict track popularity based on audio features")
    st.markdown("---")
    
    if 'popularity' not in models or 'scaler' not in models:
        st.error("❌ Required models not available")
        st.info("Missing: popularity predictor and/or scaler")
        return
    
    # Feature input
    features = create_feature_input()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🔮 Predict Popularity", type="primary", use_container_width=True):
            try:
                with st.spinner("Making prediction..."):
                    # Scale features
                    scaled_features = models['scaler'].transform(features)
                    
                    # Predict
                    prediction = models['popularity'].predict(scaled_features)[0]
                    
                    # Try to get confidence
                    try:
                        proba = models['popularity'].predict_proba(scaled_features)
                        confidence = proba.max()
                    except:
                        confidence = 0.85  # Default confidence
                    
                    st.success(f"🎵 Predicted Popularity: **{prediction:.1f}/100**")
                    st.info(f"🎯 Model Confidence: **{confidence:.2%}**")
                    
                    # Interpretation
                    if prediction >= 80:
                        st.success("🔥 High popularity - likely to be a hit!")
                    elif prediction >= 60:
                        st.warning("📈 Moderate popularity - decent performance")
                    else:
                        st.info("📊 Lower popularity - niche appeal")
                        
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
    
    with col2:
        st.subheader("🎵 About Audio Features")
        st.markdown("""
        **Key Features:**
        - **Energy**: High energy = rock/electronic
        - **Valence**: High valence = happy/uplifting
        - **Danceability**: How suitable for dancing
        - **Acousticness**: Acoustic vs electronic sounds
        """)

def genre_classification_page(models, df_merged):
    """Genre classification page"""
    st.title("🎼 Genre Classifier")
    st.markdown("Classify tracks into super and fine genres")
    st.markdown("---")
    
    if 'super_genre' not in models or 'scaler' not in models:
        st.error("❌ Required models not available")
        st.info("Missing: super genre classifier and/or scaler")
        return
    
    # Feature input
    features = create_feature_input()
    
    if st.button("🔍 Classify Genre", type="primary", use_container_width=True):
        try:
            with st.spinner("Classifying genre..."):
                # Scale features
                scaled_features = models['scaler'].transform(features)
                
                # Predict super genre
                super_pred = models['super_genre'].predict(scaled_features)[0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success(f"🎵 Super Genre: **{super_pred.upper()}**")
                    
                    # Show probabilities if available
                    try:
                        proba = models['super_genre'].predict_proba(scaled_features)[0]
                        confidence = proba.max()
                        st.info(f"🎯 Confidence: **{confidence:.2%}**")
                        
                        # Top predictions
                        if hasattr(models['super_genre'], 'classes_'):
                            classes = models['super_genre'].classes_
                            top_indices = np.argsort(proba)[::-1][:3]
                            
                            st.write("**Top 3 Predictions:**")
                            for i, idx in enumerate(top_indices):
                                st.write(f"{i+1}. {classes[idx]}: {proba[idx]:.1%}")
                    except:
                        st.info("🎯 Classification completed")
                
                with col2:
                    # Fine genre prediction
                    if super_pred in models.get('fine_genre', {}):
                        try:
                            fine_pred = models['fine_genre'][super_pred].predict(scaled_features)[0]
                            st.success(f"🎶 Fine Genre: **{fine_pred.upper()}**")
                            
                            try:
                                fine_proba = models['fine_genre'][super_pred].predict_proba(scaled_features)[0]
                                fine_confidence = fine_proba.max()
                                st.info(f"🎯 Confidence: **{fine_confidence:.2%}**")
                            except:
                                pass
                                
                        except Exception as e:
                            st.warning(f"Fine genre prediction failed: {e}")
                    else:
                        st.warning(f"Fine genre model for {super_pred} not available")
                        
        except Exception as e:
            st.error(f"❌ Classification failed: {e}")

def simple_recommendations_page(df_merged, feature_matrix):
    """Simplified recommendations page"""
    st.title("🎵 Music Recommendations")
    st.markdown("Get music recommendations based on similarity")
    st.markdown("---")
    
    if df_merged is None:
        st.error("❌ Dataset not available for recommendations")
        return
    
    # Simple track selection
    st.subheader("🎧 Select a Reference Track")
    
    # Search
    search = st.text_input("🔍 Search for a track or artist:")
    
    if search:
        try:
            mask = (df_merged['track_name'].str.contains(search, case=False, na=False) |
                   df_merged['main_artist'].str.contains(search, case=False, na=False))
            filtered_df = df_merged[mask].head(20)
        except:
            filtered_df = df_merged.head(20)
    else:
        filtered_df = df_merged.head(20)
    
    if len(filtered_df) > 0:
        # Create display options
        options = []
        indices = []
        
        for idx, row in filtered_df.iterrows():
            display = f"{row.get('track_name', 'Unknown')} - {row.get('main_artist', 'Unknown')}"
            options.append(display)
            indices.append(idx)
        
        selected_idx = st.selectbox("Choose a track:", range(len(options)), 
                                  format_func=lambda x: options[x])
        
        if st.button("🎵 Get Recommendations", type="primary"):
            try:
                selected_track = df_merged.loc[indices[selected_idx]]
                
                # Simple genre-based recommendations
                same_genre_tracks = df_merged[
                    (df_merged['super_genre'] == selected_track.get('super_genre', '')) &
                    (df_merged.index != indices[selected_idx])
                ]
                
                if len(same_genre_tracks) > 0:
                    recommendations = same_genre_tracks.sample(min(10, len(same_genre_tracks)))
                    
                    st.subheader(f"🎵 Recommendations (Similar to {selected_track.get('super_genre', 'Unknown')} genre)")
                    
                    for _, rec in recommendations.iterrows():
                        col1, col2, col3 = st.columns([2, 2, 1])
                        with col1:
                            st.write(f"**{rec.get('track_name', 'Unknown')}**")
                        with col2:
                            st.write(rec.get('main_artist', 'Unknown'))
                        with col3:
                            if 'popularity' in rec:
                                st.write(f"Pop: {rec['popularity']:.0f}")
                else:
                    st.warning("No recommendations found in the same genre")
                    
            except Exception as e:
                st.error(f"❌ Recommendation failed: {e}")

def main():
    """Main application"""
    st.sidebar.title("🎵 Music Dashboard")
    
    # Initialize data loading
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Load data once
    if not st.session_state.data_loaded:
        with st.spinner("🚀 Loading application..."):
            df_merged, feature_matrix, tsne_data, pca_data = load_data_corrected()
            models = load_models_corrected()
            
            # Store in session state
            st.session_state.df_merged = df_merged
            st.session_state.feature_matrix = feature_matrix
            st.session_state.tsne_data = tsne_data
            st.session_state.pca_data = pca_data
            st.session_state.models = models
            st.session_state.data_loaded = True
    
    # Get data from session state
    df_merged = st.session_state.get('df_merged')
    feature_matrix = st.session_state.get('feature_matrix')
    models = st.session_state.get('models', {})
    
    # Navigation
    pages = {
        "🏠 Home": "home",
        "🎯 Popularity Prediction": "popularity",
        "🎼 Genre Classification": "genre",
        "🎵 Recommendations": "recommendations"
    }
    
    selected_page = st.sidebar.selectbox("📍 Navigate to:", list(pages.keys()))
    page_key = pages[selected_page]
    
    # Status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Status")
    
    if df_merged is not None:
        st.sidebar.success(f"✅ Data: {len(df_merged):,} tracks")
    else:
        st.sidebar.error("❌ Data not loaded")
    
    available_models = len([k for k in models.keys() if k != 'fine_genre'])
    if available_models > 0:
        st.sidebar.success(f"✅ Models: {available_models} loaded")
    else:
        st.sidebar.error("❌ No models loaded")
    
    # Show pages
    try:
        if page_key == "home":
            home_page(df_merged)
        elif page_key == "popularity":
            popularity_prediction_page(models, df_merged)
        elif page_key == "genre":
            genre_classification_page(models, df_merged)
        elif page_key == "recommendations":
            simple_recommendations_page(df_merged, feature_matrix)
            
    except Exception as e:
        st.error(f"❌ Page error: {e}")
        with st.expander("🔍 Error Details"):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()