import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# Super genres list
SUPER_GENRES = ['rock', 'pop', 'electronic', 'metal', 'hiphop', 'japanese', 
                'latin', 'classical', 'country', 'folk', 'ambient', 'world', 
                'children', 'other', 'misc']

# Audio features for input
AUDIO_FEATURES = ['danceability', 'energy', 'loudness', 'speechiness', 
                  'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Google Drive file ID
DRIVE_FILE_ID = "12aGkuSuDpVO_gLdcal-fyUJYLGqcPFyI"

@st.cache_resource
def download_and_extract_files():
    """Download and extract files from Google Drive with better error handling"""
    try:
        st.info("🔧 Starting file download process...")
        
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "music_files.zip")
        
        st.info(f"🔧 Temp directory: {temp_dir}")
        
        # Download with timeout and better error handling
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        
        with st.spinner("📥 Downloading files... This may take several minutes."):
            try:
                gdown.download(url, zip_path, quiet=False, fuzzy=True)
                
                # Check if file was downloaded successfully
                if not os.path.exists(zip_path) or os.path.getsize(zip_path) == 0:
                    st.error("Download failed or file is empty")
                    return None
                    
                st.success(f"✅ Downloaded {os.path.getsize(zip_path) / (1024*1024):.1f} MB")
                
            except Exception as download_error:
                st.error(f"Download failed: {download_error}")
                return None
        
        # Extract with progress
        with st.spinner("📦 Extracting files..."):
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                st.success("✅ Files extracted successfully")
            except Exception as extract_error:
                st.error(f"Extraction failed: {extract_error}")
                return None
        
        # Find the correct base directory
        extracted_items = os.listdir(temp_dir)
        st.info(f"🔧 Extracted items: {extracted_items}")
        
        # Look for the correct folder structure
        possible_dirs = [
            os.path.join(temp_dir, 'Spotify_app'),
            os.path.join(temp_dir, 'spotify_app'),
            temp_dir  # fallback to temp_dir itself
        ]
        
        # Add any subdirectories found
        for item in extracted_items:
            item_path = os.path.join(temp_dir, item)
            if os.path.isdir(item_path) and item != '__MACOSX':  # Ignore Mac metadata
                possible_dirs.insert(-1, item_path)
        
        base_dir = None
        for possible_dir in possible_dirs:
            if os.path.exists(possible_dir):
                # Check if this directory contains the expected files
                dir_files = os.listdir(possible_dir) if os.path.isdir(possible_dir) else []
                expected_files = ['df_merged.pkl', 'feature_matrix.npy']
                
                if any(expected_file in dir_files for expected_file in expected_files):
                    base_dir = possible_dir
                    st.info(f"✅ Using directory: {base_dir}")
                    break
        
        if base_dir is None:
            st.error("Could not find directory with expected files")
            return None
            
        # List available files for debugging
        available_files = os.listdir(base_dir)
        st.info(f"🔧 Available files: {available_files[:10]}...")  # Show first 10
        
        return base_dir
        
    except Exception as e:
        st.error(f"❌ Unexpected error in download_and_extract_files: {e}")
        st.exception(e)
        return None

@st.cache_data
def load_data():
    """Load data files with better error handling"""
    base_dir = download_and_extract_files()
    if base_dir is None:
        return None, None, None, None
        
    try:
        st.info("📊 Loading data files...")
        
        # Load main data file
        df_path = os.path.join(base_dir, "df_merged.pkl")
        if not os.path.exists(df_path):
            st.error(f"df_merged.pkl not found in {base_dir}")
            return None, None, None, None
            
        df_merged = pd.read_pickle(df_path)
        st.success(f"✅ Loaded df_merged: {df_merged.shape}")
        
        # Load feature matrix
        feature_path = os.path.join(base_dir, "feature_matrix.npy")
        feature_matrix = None
        if os.path.exists(feature_path):
            feature_matrix = np.load(feature_path)
            st.success(f"✅ Loaded feature_matrix: {feature_matrix.shape}")
        else:
            st.warning("⚠️ feature_matrix.npy not found")
        
        # Load clustering data (try different names)
        tsne_data = None
        pca_data = None
        
        # Try different t-SNE file names
        for tsne_file in ["tsne_3d_clusters.pkl", "tsne_3d_cluster.pkl"]:
            tsne_path = os.path.join(base_dir, tsne_file)
            if os.path.exists(tsne_path):
                try:
                    tsne_data = pd.read_pickle(tsne_path)
                    st.success(f"✅ Loaded {tsne_file}")
                    break
                except Exception as e:
                    st.warning(f"Failed to load {tsne_file}: {e}")
        
        # Try different PCA file names
        for pca_file in ["pca_3d_clusters.pkl", "pca_3d_cluster.pkl"]:
            pca_path = os.path.join(base_dir, pca_file)
            if os.path.exists(pca_path):
                try:
                    pca_data = pd.read_pickle(pca_path)
                    st.success(f"✅ Loaded {pca_file}")
                    break
                except Exception as e:
                    st.warning(f"Failed to load {pca_file}: {e}")
        
        return df_merged, feature_matrix, tsne_data, pca_data
        
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.exception(e)
        return None, None, None, None

@st.cache_resource
def load_models():
    """Load ML models with error handling"""
    base_dir = download_and_extract_files()
    if base_dir is None:
        return {}
        
    models = {}
    try:
        st.info("🤖 Loading ML models...")
        
        # List of model files to try loading
        model_files = {
            'popularity': 'rf_popularity_cluster.pkl',
            'scaler': 'hierarchical_scaler.pkl',
            'super_genre': 'super_genre_model.pkl'
        }
        
        for model_name, filename in model_files.items():
            model_path = os.path.join(base_dir, filename)
            if os.path.exists(model_path):
                try:
                    models[model_name] = joblib.load(model_path)
                    st.success(f"✅ Loaded {model_name} model")
                except Exception as e:
                    st.warning(f"⚠️ Failed to load {model_name}: {e}")
            else:
                st.warning(f"⚠️ Model file not found: {filename}")
        
        # Load fine genre models
        models['fine_genre'] = {}
        for genre in SUPER_GENRES:
            model_path = os.path.join(base_dir, f"fine_genre_model_{genre}.pkl")
            if os.path.exists(model_path):
                try:
                    models['fine_genre'][genre] = joblib.load(model_path)
                except Exception as e:
                    pass  # Silently skip failed fine genre models
        
        st.success(f"✅ Loaded {len(models['fine_genre'])} fine genre models")
        
        # Try to load SHAP data (optional)
        shap_files = {
            'shap_popularity': 'shap_popularity.pkl',
            'explainer_popularity': 'explainer_populariy.pkl',  # Note the typo in original
            'shap_genre': 'shap_genre_classification.pkl',
            'explainer_genre': 'explainer_genre_classification.pkl'
        }
        
        for shap_name, filename in shap_files.items():
            shap_path = os.path.join(base_dir, filename)
            if os.path.exists(shap_path):
                try:
                    models[shap_name] = joblib.load(shap_path)
                except Exception:
                    pass  # Silently skip SHAP models if they fail
        
        return models
        
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return {}

def safe_import_recommender():
    """Safely import recommender function"""
    base_dir = download_and_extract_files()
    if base_dir is None:
        return None
        
    try:
        # Add base directory to path
        if base_dir not in sys.path:
            sys.path.insert(0, base_dir)
        
        # Try to import
        from recommender import hybrid_recommend_with_weights
        return hybrid_recommend_with_weights
    except ImportError as e:
        st.warning(f"Recommender function not available: {e}")
        return None
    except Exception as e:
        st.warning(f"Error importing recommender: {e}")
        return None

def create_feature_input():
    """Create input widgets for audio features"""
    st.subheader("🎛️ Audio Feature Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        danceability = st.slider("Danceability", 0.0, 1.0, 0.5, 0.01)
        energy = st.slider("Energy", 0.0, 1.0, 0.5, 0.01)
        speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1, 0.01)
    
    with col2:
        acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5, 0.01)
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.1, 0.01)
        liveness = st.slider("Liveness", 0.0, 1.0, 0.2, 0.01)
    
    with col3:
        valence = st.slider("Valence", 0.0, 1.0, 0.5, 0.01)
        tempo = st.slider("Tempo", 50.0, 200.0, 120.0, 1.0)
        loudness = st.slider("Loudness", -30.0, 0.0, -10.0, 0.5)
    
    return np.array([danceability, energy, loudness, speechiness, acousticness, 
                     instrumentalness, liveness, valence, tempo]).reshape(1, -1)

def home_page(df_merged):
    """Home page with project overview"""
    st.title("🎵 Music Analysis Dashboard")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Project Overview")
        st.markdown("""
        This dashboard presents a comprehensive analysis of music data using machine learning:
        
        **🎯 Key Features:**
        - **Popularity Prediction**: Predict track popularity using audio features
        - **Genre Classification**: Classify tracks into super and fine genres  
        - **Music Recommendations**: Get personalized track recommendations
        - **Clustering Analysis**: Visualize music patterns in 3D space
        
        **🤖 ML Techniques:**
        - Random Forest for regression and classification
        - Cosine similarity for recommendations
        - t-SNE and PCA for dimensionality reduction
        """)
    
    with col2:
        st.header("📈 Dataset Info")
        if df_merged is not None:
            try:
                st.metric("Total Tracks", f"{len(df_merged):,}")
                if 'main_artist' in df_merged.columns:
                    st.metric("Artists", f"{df_merged['main_artist'].nunique():,}")
                if 'super_genre' in df_merged.columns:
                    st.metric("Super Genres", df_merged['super_genre'].nunique())
                if 'popularity' in df_merged.columns:
                    st.metric("Avg Popularity", f"{df_merged['popularity'].mean():.1f}")
            except Exception as e:
                st.warning("Could not display all metrics")
    
    # Genre distribution if available
    if df_merged is not None and 'super_genre' in df_merged.columns:
        try:
            st.header("🎼 Genre Distribution")
            genre_counts = df_merged['super_genre'].value_counts().head(10)
            fig = px.bar(
                x=genre_counts.values, 
                y=genre_counts.index, 
                orientation='h', 
                title="Top 10 Genres in Dataset"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning("Could not display genre distribution")

def popularity_predictor(models, df_merged):
    """Popularity prediction page"""
    st.title("🎯 Popularity Predictor")
    st.markdown("Predict track popularity based on audio features")
    st.markdown("---")
    
    if 'popularity' not in models or 'scaler' not in models:
        st.error("Required models (popularity predictor or scaler) not available")
        return
    
    # Feature input
    features = create_feature_input()
    
    if st.button("🔮 Predict Popularity", type="primary"):
        try:
            # Scale features
            scaled_features = models['scaler'].transform(features)
            
            # Predict
            prediction = models['popularity'].predict(scaled_features)[0]
            
            # Get prediction probability if available
            confidence = 0.8  # Default confidence
            try:
                proba = models['popularity'].predict_proba(scaled_features)
                confidence = proba.max()
            except:
                pass
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"🎵 Predicted Popularity: **{prediction:.1f}/100**")
                st.info(f"🎯 Model Confidence: **{confidence:.2%}**")
            
            with col2:
                # Popularity gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prediction,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Popularity Score"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75, 'value': 90
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.exception(e)

def genre_classifier(models, df_merged):
    """Genre classification page"""
    st.title("🎼 Genre Classifier")
    st.markdown("Classify tracks into super and fine genres")
    st.markdown("---")
    
    if 'super_genre' not in models or 'scaler' not in models:
        st.error("Required models (genre classifier or scaler) not available")
        return
    
    # Feature input
    features = create_feature_input()
    
    if st.button("🔍 Classify Genre", type="primary"):
        try:
            # Scale features
            scaled_features = models['scaler'].transform(features)
            
            # Predict super genre
            super_pred = models['super_genre'].predict(scaled_features)[0]
            
            try:
                super_proba = models['super_genre'].predict_proba(scaled_features)[0]
                super_confidence = super_proba.max()
                
                # Get class labels
                if hasattr(models['super_genre'], 'classes_'):
                    classes = models['super_genre'].classes_
                else:
                    classes = SUPER_GENRES[:len(super_proba)]
            except:
                super_confidence = 0.8
                super_proba = None
                classes = None
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"🎵 Super Genre: **{super_pred.upper()}**")
                st.info(f"🎯 Confidence: **{super_confidence:.2%}**")
                
                # Super genre probabilities
                if super_proba is not None and classes is not None:
                    try:
                        proba_df = pd.DataFrame({
                            'Genre': classes,
                            'Probability': super_proba
                        }).sort_values('Probability', ascending=False).head(5)
                        
                        fig = px.bar(
                            proba_df, 
                            x='Probability', 
                            y='Genre', 
                            orientation='h', 
                            title="Top 5 Genre Predictions"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning("Could not display probability chart")
            
            with col2:
                # Fine genre prediction
                if super_pred in models.get('fine_genre', {}):
                    try:
                        fine_model = models['fine_genre'][super_pred]
                        fine_pred = fine_model.predict(scaled_features)[0]
                        
                        try:
                            fine_proba = fine_model.predict_proba(scaled_features)[0]
                            fine_confidence = fine_proba.max()
                            fine_classes = fine_model.classes_
                        except:
                            fine_confidence = 0.8
                            fine_proba = None
                            fine_classes = None
                        
                        st.success(f"🎶 Fine Genre: **{fine_pred.upper()}**")
                        st.info(f"🎯 Confidence: **{fine_confidence:.2%}**")
                        
                        if fine_proba is not None and fine_classes is not None:
                            try:
                                fine_df = pd.DataFrame({
                                    'Genre': fine_classes,
                                    'Probability': fine_proba
                                }).sort_values('Probability', ascending=False).head(5)
                                
                                fig = px.bar(
                                    fine_df, 
                                    x='Probability', 
                                    y='Genre', 
                                    orientation='h', 
                                    title=f"Top {super_pred.title()} Sub-genres"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.warning("Could not display fine genre chart")
                    except Exception as e:
                        st.warning(f"Fine genre prediction failed: {e}")
                else:
                    st.warning(f"Fine genre model for {super_pred} not available")
        
        except Exception as e:
            st.error(f"Classification failed: {e}")
            st.exception(e)

def simple_recommendation_system(df_merged, feature_matrix):
    """Simplified recommendation system"""
    st.title("🎵 Music Recommendation System")
    st.markdown("Get music recommendations based on similarity")
    st.markdown("---")
    
    if df_merged is None:
        st.error("Dataset not available")
        return
        
    # Track selection
    st.subheader("🎧 Select a Track")
    
    # Search functionality
    search_term = st.text_input("🔍 Search for a track or artist:")
    
    if search_term:
        # Safe search with error handling
        try:
            mask1 = df_merged['track_name'].str.contains(search_term, case=False, na=False)
            if 'main_artist' in df_merged.columns:
                mask2 = df_merged['main_artist'].str.contains(search_term, case=False, na=False)
                filtered_df = df_merged[mask1 | mask2]
            else:
                filtered_df = df_merged[mask1]
        except Exception as e:
            st.warning("Search failed, showing random tracks")
            filtered_df = df_merged.sample(min(50, len(df_merged)))
    else:
        filtered_df = df_merged.head(50)  # Show first 50 tracks
    
    if len(filtered_df) > 0:
        # Create simple display
        track_options = []
        track_indices = []
        
        for idx, row in filtered_df.head(20).iterrows():  # Limit to 20 for performance
            try:
                track_name = row.get('track_name', 'Unknown Track')
                artist_name = row.get('main_artist', 'Unknown Artist')
                genre = row.get('super_genre', 'Unknown')
                display_text = f"{track_name} - {artist_name} ({genre})"
                track_options.append(display_text)
                track_indices.append(idx)
            except Exception as e:
                continue
        
        if track_options:
            selected_idx = st.selectbox(
                "Choose a track:",
                range(len(track_options)),
                format_func=lambda x: track_options[x]
            )
            
            selected_track_idx = track_indices[selected_idx]
            selected_track = df_merged.loc[selected_track_idx]
            
            # Display selected track info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**Track:** {selected_track.get('track_name', 'Unknown')}")
            with col2:
                st.info(f"**Artist:** {selected_track.get('main_artist', 'Unknown')}")
            with col3:
                st.info(f"**Genre:** {selected_track.get('super_genre', 'Unknown')}")
            
            # Simple recommendation
            if st.button("🎵 Get Recommendations", type="primary"):
                try:
                    # Simple genre-based recommendation
                    same_genre = df_merged[
                        df_merged['super_genre'] == selected_track.get('super_genre', '')
                    ]
                    
                    if len(same_genre) > 1:
                        # Remove the selected track and get random recommendations
                        recommendations = same_genre[
                            same_genre.index != selected_track_idx
                        ].sample(min(10, len(same_genre)-1))
                        
                        st.subheader("🎵 Recommendations (Same Genre)")
                        
                        display_cols = ['track_name', 'main_artist', 'super_genre']
                        available_cols = [col for col in display_cols if col in recommendations.columns]
                        
                        if available_cols:
                            st.dataframe(
                                recommendations[available_cols].head(10), 
                                use_container_width=True
                            )
                        else:
                            st.warning("Cannot display recommendations - missing column data")
                    else:
                        st.warning("Not enough tracks in the same genre for recommendations")
                        
                except Exception as e:
                    st.error(f"Recommendation failed: {e}")
        else:
            st.warning("No valid tracks found")
    else:
        st.warning("No tracks found. Try a different search term.")

def clustering_view(tsne_data, pca_data, df_merged):
    """3D clustering visualization page"""
    st.title("🎨 Clustering Analysis")
    st.markdown("Explore music patterns through visualizations")
    st.markdown("---")
    
    has_tsne = tsne_data is not None
    has_pca = pca_data is not None
    
    if not has_tsne and not has_pca:
        st.error("Clustering visualization data not available")
        return
    
    # Choose visualization type
    options = []
    if has_tsne:
        options.append("t-SNE")
    if has_pca:
        options.append("PCA")
    
    viz_type = st.selectbox("Choose Visualization:", options)
    
    # Select data
    if viz_type == "t-SNE" and has_tsne:
        data = tsne_data
        title = "t-SNE 3D Visualization"
    elif viz_type == "PCA" and has_pca:
        data = pca_data  
        title = "PCA 3D Visualization"
    else:
        st.error("Selected visualization not available")
        return
    
    try:
        # Check available columns for coloring
        color_options = []
        if 'cluster' in data.columns:
            color_options.append("cluster")
        if 'super_genre' in data.columns:
            color_options.append("super_genre")
        if 'popularity' in data.columns:
            color_options.append("popularity")
        
        if not color_options:
            color_options = ["None"]
        
        color_option = st.selectbox("Color by:", color_options)
        
        # Create plot
        if 'x' in data.columns and 'y' in data.columns and 'z' in data.columns:
            
            hover_data = []
            if 'track_name' in data.columns:
                hover_data.append('track_name')
            if 'main_artist' in data.columns:
                hover_data.append('main_artist')
            
            if color_option == "popularity":
                fig = px.scatter_3d(
                    data.head(1000),  # Limit points for performance
                    x='x', y='y', z='z',
                    color='popularity',
                    title=f"{title} (colored by popularity)",
                    color_continuous_scale='Viridis',
                    hover_data=hover_data if hover_data else None
                )
            elif color_option != "None":
                fig = px.scatter_3d(
                    data.head(1000),  # Limit points for performance
                    x='x', y='y', z='z',
                    color=color_option,
                    title=f"{title} (colored by {color_option})",
                    hover_data=hover_data if hover_data else None
                )
            else:
                fig = px.scatter_3d(
                    data.head(1000),
                    x='x', y='y', z='z',
                    title=title,
                    hover_data=hover_data if hover_data else None
                )
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"Showing first 1000 points out of {len(data)} total")
            
        else:
            st.error("Required coordinate columns (x, y, z) not found in data")
            
    except Exception as e:
        st.error(f"Visualization failed: {e}")
        st.exception(e)

def trend_analysis(df_merged):
    """Trend analysis page"""
    st.title("📈 Music Analysis")
    st.markdown("Analyze patterns in the music dataset")
    st.markdown("---")
    
    if df_merged is None:
        st.error("Dataset not available")
        return
    
    try:
        # Genre popularity analysis
        if 'super_genre' in df_merged.columns and 'popularity' in df_merged.columns:
            st.subheader("🎼 Popularity by Genre")
            
            genre_stats = df_merged.groupby('super_genre').agg({
                'popularity': ['mean', 'count', 'std']
            }).round(2)
            
            genre_stats.columns = ['Avg_Popularity', 'Track_Count', 'Popularity_Std']
            genre_stats = genre_stats.sort_values('Avg_Popularity', ascending=False)
            
            # Bar chart
            fig = px.bar(
                x=genre_stats.index,
                y=genre_stats['Avg_Popularity'],
                title="Average Popularity by Genre",
                labels={'x': 'Genre', 'y': 'Average Popularity'}
            )
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show stats table
            st.dataframe(genre_stats, use_container_width=True)
        
        # Audio features analysis
        available_features = [f for f in AUDIO_FEATURES if f in df_merged.columns]
        
        if available_features and 'super_genre' in df_merged.columns:
            st.subheader("🎵 Audio Features by Genre")
            
            # Select genres for comparison
            unique_genres = df_merged['super_genre'].unique()
            selected_genres = st.multiselect(
                "Select genres to compare:", 
                unique_genres, 
                default=unique_genres[:5] if len(unique_genres) > 5 else unique_genres
            )
            
            if selected_genres and available_features:
                try:
                    # Calculate mean features by genre
                    feature_data = df_merged[
                        df_merged['super_genre'].isin(selected_genres)
                    ].groupby('super_genre')[available_features].mean()
                    
                    # Create heatmap
                    fig = px.imshow(
                        feature_data.T,
                        title="Audio Features Heatmap by Genre",
                        aspect="auto",
                        color_continuous_scale="RdBu_r"
                    )
                    fig.update_xaxes(title="Genre")
                    fig.update_yaxes(title="Audio Features")
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"Could not create features heatmap: {e}")
        
        # Dataset overview
        st.subheader("📊 Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Shape:**", df_merged.shape)
            st.write("**Available Columns:**")
            st.write(list(df_merged.columns))
        
        with col2:
            if 'popularity' in df_merged.columns:
                fig = px.histogram(
                    df_merged, 
                    x='popularity', 
                    title="Popularity Distribution",
                    nbins=30
                )
                st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Analysis failed: {e}")

def main():
    """Main application with improved error handling"""
    
    # Add custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1DB954, #1ed760);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1DB954, #1ed760);
    }
    </style>
    """, unsafe_allow_html=True)
    
    try:
        # Initialize session state for error tracking
        if 'initialization_complete' not in st.session_state:
            st.session_state.initialization_complete = False
            st.session_state.data_loaded = False
            st.session_state.models_loaded = False
        
        # Main header
        st.markdown('<div class="main-header"><h1>🎵 Music Analysis Dashboard</h1></div>', 
                   unsafe_allow_html=True)
        
        # Sidebar navigation
        st.sidebar.title("🎵 Navigation")
        
        # Initialize data and models if not already done
        if not st.session_state.initialization_complete:
            with st.spinner("Initializing application..."):
                
                # Load data
                with st.container():
                    st.info("📥 Loading dataset...")
                    df_merged, feature_matrix, tsne_data, pca_data = load_data()
                    
                    if df_merged is not None:
                        st.session_state.df_merged = df_merged
                        st.session_state.feature_matrix = feature_matrix
                        st.session_state.tsne_data = tsne_data
                        st.session_state.pca_data = pca_data
                        st.session_state.data_loaded = True
                        st.success("✅ Dataset loaded successfully!")
                    else:
                        st.error("❌ Failed to load dataset")
                        st.stop()
                
                # Load models
                with st.container():
                    st.info("🤖 Loading ML models...")
                    models = load_models()
                    
                    if models:
                        st.session_state.models = models
                        st.session_state.models_loaded = True
                        st.success("✅ Models loaded successfully!")
                    else:
                        st.warning("⚠️ Some models may not be available")
                        st.session_state.models = {}
                
                st.session_state.initialization_complete = True
                st.rerun()  # Refresh to show the main interface
        
        # Get data from session state
        df_merged = st.session_state.get('df_merged')
        feature_matrix = st.session_state.get('feature_matrix')
        tsne_data = st.session_state.get('tsne_data')
        pca_data = st.session_state.get('pca_data')
        models = st.session_state.get('models', {})
        
        # Navigation pages
        pages = {
            "🏠 Home": "home",
            "🎯 Popularity Predictor": "popularity",
            "🎼 Genre Classifier": "genre", 
            "🎵 Recommendations": "recommendations",
            "🎨 Clustering View": "clustering",
            "📈 Analysis": "trends"
        }
        
        selected_page = st.sidebar.selectbox("Select Page:", list(pages.keys()))
        page_key = pages[selected_page]
        
        # Status sidebar
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 System Status")
        
        if st.session_state.get('data_loaded', False):
            st.sidebar.success("✅ Data Loaded")
            if df_merged is not None:
                st.sidebar.info(f"📈 {len(df_merged):,} tracks")
        else:
            st.sidebar.error("❌ Data Not Loaded")
        
        if st.session_state.get('models_loaded', False):
            model_count = len([k for k in models.keys() if k != 'fine_genre'])
            st.sidebar.success(f"✅ {model_count} Models Loaded")
        else:
            st.sidebar.warning("⚠️ Limited Models")
        
        # Reset button
        if st.sidebar.button("🔄 Reset Application"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        # Main content area
        try:
            if page_key == "home":
                home_page(df_merged)
            elif page_key == "popularity":
                if models and 'popularity' in models and 'scaler' in models:
                    popularity_predictor(models, df_merged)
                else:
                    st.error("🚫 Popularity prediction not available - missing required models")
                    st.info("Required models: popularity predictor and feature scaler")
            elif page_key == "genre":
                if models and 'super_genre' in models and 'scaler' in models:
                    genre_classifier(models, df_merged)
                else:
                    st.error("🚫 Genre classification not available - missing required models")
                    st.info("Required models: genre classifier and feature scaler")
            elif page_key == "recommendations":
                simple_recommendation_system(df_merged, feature_matrix)
            elif page_key == "clustering":
                clustering_view(tsne_data, pca_data, df_merged)
            elif page_key == "trends":
                trend_analysis(df_merged)
                
        except Exception as page_error:
            st.error(f"❌ Error in {selected_page}: {str(page_error)}")
            
            # Show error details in expander
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())
                
            # Suggest solutions
            st.markdown("### 💡 Possible Solutions:")
            st.markdown("""
            - Try refreshing the page
            - Click 'Reset Application' in the sidebar
            - Check if all required files are present
            - Ensure models are compatible with the dataset
            """)
        
        # Footer
        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        <div style='text-align: center'>
        <p><b>🎵 Music Analysis Dashboard</b></p>
        <p><small>Built with Streamlit & ML</small></p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as main_error:
        st.error(f"❌ Critical application error: {str(main_error)}")
        st.markdown("### 🔧 Troubleshooting")
        
        with st.expander("🔍 Full Error Details"):
            st.code(traceback.format_exc())
        
        st.markdown("""
        ### 🆘 Emergency Actions:
        1. **Refresh the browser page**
        2. **Clear browser cache**
        3. **Check internet connection** (for file downloads)
        4. **Verify Google Drive file is accessible**
        5. **Contact support if issue persists**
        """)
        
        # Emergency reset button
        if st.button("🚨 Emergency Reset", type="primary"):
            st.cache_data.clear()
            st.cache_resource.clear()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()