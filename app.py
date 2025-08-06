import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
from sklearn.preprocessing import StandardScaler
import sys
import os
import gdown
import zipfile
import tempfile

# Import recommendation function
try:
    from recommender import hybrid_recommend_with_weights
except ImportError:
    # We'll handle this after downloading files
    pass

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

# Google Drive file ID (extracted from your link)
DRIVE_FILE_ID = "12aGkuSuDpVO_gLdcal-fyUJYLGqcPFyI"

@st.cache_resource
def download_and_extract_files():
    """Download and extract files from Google Drive"""
    try:
        st.info("🔧 Debug: Starting file download process...")
        
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "music_files.zip")
        
        st.info(f"🔧 Debug: Temp directory created at: {temp_dir}")
        
        # Download the zip file from Google Drive
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        
        with st.spinner("📥 Downloading files from Google Drive... This may take a few minutes."):
            st.info(f"🔧 Debug: Downloading from URL: {url}")
            gdown.download(url, zip_path, quiet=False)
            st.info(f"🔧 Debug: Download completed. File size: {os.path.getsize(zip_path)} bytes")
        
        # Extract the zip file
        with st.spinner("📦 Extracting files..."):
            st.info("🔧 Debug: Starting file extraction...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                st.info(f"🔧 Debug: Found {len(file_list)} files in zip")
                zip_ref.extractall(temp_dir)
            st.info("🔧 Debug: Extraction completed")
        
        # Find the extracted folder (it might be in a subfolder)
        extracted_items = os.listdir(temp_dir)
        st.info(f"🔧 Debug: Items in temp dir: {extracted_items}")
        
        # Look for Spotify_app folder specifically, or any subfolder
        if 'Spotify_app' in extracted_items:
            base_dir = os.path.join(temp_dir, 'Spotify_app')
            st.info(f"🔧 Debug: Found Spotify_app folder, using as base: {base_dir}")
        else:
            extracted_folders = [f for f in extracted_items if os.path.isdir(os.path.join(temp_dir, f))]
            if extracted_folders:
                # If there's a subfolder, use that as the base directory
                base_dir = os.path.join(temp_dir, extracted_folders[0])
                st.info(f"🔧 Debug: Using first subfolder as base: {base_dir}")
            else:
                # Otherwise use temp_dir directly
                base_dir = temp_dir
                st.info(f"🔧 Debug: Using temp_dir as base: {base_dir}")
        
        # List files in base directory
        base_files = os.listdir(base_dir)
        st.info(f"🔧 Debug: Files in base directory: {base_files[:15]}")  # Show first 15 files
        
        st.success("✅ Files downloaded and extracted successfully!")
        return base_dir
        
    except Exception as e:
        st.error(f"❌ Error downloading files: {e}")
        st.error("Please check if the Google Drive link is accessible and the file exists.")
        st.exception(e)  # Show full traceback
        return None

@st.cache_data
def load_data():
    """Load all necessary data files"""
    base_dir = download_and_extract_files()
    if base_dir is None:
        return None, None, None, None
        
    try:
        st.info(f"🔧 Debug: Looking for data files in: {base_dir}")
        
        # Load main data files
        df_merged = pd.read_pickle(os.path.join(base_dir, "df_merged.pkl"))
        st.info("✅ df_merged.pkl loaded successfully")
        
        feature_matrix = np.load(os.path.join(base_dir, "feature_matrix.npy"))
        st.info("✅ feature_matrix.npy loaded successfully")
        
        # Try both file names for t-SNE data (with and without 's')
        tsne_files = ["tsne_3d_clusters.pkl", "tsne_3d_cluster.pkl"]
        tsne_data = None
        for tsne_file in tsne_files:
            try:
                tsne_data = pd.read_pickle(os.path.join(base_dir, tsne_file))
                st.info(f"✅ {tsne_file} loaded successfully")
                break
            except FileNotFoundError:
                continue
        
        if tsne_data is None:
            st.warning("⚠️ t-SNE data file not found (tried both tsne_3d_cluster.pkl and tsne_3d_clusters.pkl)")
        
        # Try both file names for PCA data (with and without 's')
        pca_files = ["pca_3d_clusters.pkl", "pca_3d_cluster.pkl"]
        pca_data = None
        for pca_file in pca_files:
            try:
                pca_data = pd.read_pickle(os.path.join(base_dir, pca_file))
                st.info(f"✅ {pca_file} loaded successfully")
                break
            except FileNotFoundError:
                continue
        
        if pca_data is None:
            st.warning("⚠️ PCA data file not found (tried both pca_3d_cluster.pkl and pca_3d_clusters.pkl)")
        
        return df_merged, feature_matrix, tsne_data, pca_data
    except Exception as e:
        st.error(f"Error loading data files: {e}")
        # List what files are actually available
        try:
            available_files = os.listdir(base_dir)
            st.info(f"Available files: {available_files}")
        except:
            pass
        return None, None, None, None

@st.cache_resource
def load_models():
    """Load all ML models and scalers"""
    base_dir = download_and_extract_files()
    if base_dir is None:
        return {}
        
    models = {}
    try:
        # Load main models
        models['popularity'] = joblib.load(os.path.join(base_dir, "rf_popularity_cluster.pkl"))
        models['scaler'] = joblib.load(os.path.join(base_dir, "hierarchical_scaler.pkl"))
        models['super_genre'] = joblib.load(os.path.join(base_dir, "super_genre_model.pkl"))
        
        # Load fine genre models
        models['fine_genre'] = {}
        for genre in SUPER_GENRES:
            try:
                model_path = os.path.join(base_dir, f"fine_genre_model_{genre}.pkl")
                models['fine_genre'][genre] = joblib.load(model_path)
            except Exception as e:
                st.warning(f"Fine genre model for {genre} not found: {e}")
        
        # Load SHAP data
        try:
            models['shap_popularity'] = joblib.load(os.path.join(base_dir, "shap_popularity.pkl"))
            models['explainer_popularity'] = joblib.load(os.path.join(base_dir, "explainer_populariy.pkl"))
            models['shap_genre'] = joblib.load(os.path.join(base_dir, "shap_genre_classification.pkl"))
            models['explainer_genre'] = joblib.load(os.path.join(base_dir, "explainer_genre_classification.pkl"))
        except Exception as e:
            st.warning(f"SHAP data not found: {e}")
        
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}

@st.cache_resource
def load_recommender_function():
    """Load the recommendation function from recommender.py"""
    base_dir = download_and_extract_files()
    if base_dir is None:
        return None
        
    try:
        # Add the base directory to Python path
        if base_dir not in sys.path:
            sys.path.append(base_dir)
        
        # Import the function
        from recommender import hybrid_recommend_with_weights
        return hybrid_recommend_with_weights
    except Exception as e:
        st.error(f"Error loading recommender function: {e}")
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
    
    # Project overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Project Overview")
        st.markdown("""
        This dashboard presents a comprehensive analysis of music data using machine learning techniques:
        
        **🎯 Objectives:**
        - **Popularity Prediction**: Predict track popularity based on audio features
        - **Genre Classification**: Classify tracks into super and fine genres
        - **Music Recommendation**: Hybrid recommendation system using multiple factors
        - **Clustering Analysis**: Discover patterns in music through unsupervised learning
        
        **🤖 ML Tasks:**
        - Random Forest for popularity prediction
        - Multi-class classification for genre prediction
        - Cosine similarity + clustering for recommendations
        - t-SNE and PCA for visualization
        """)
    
    with col2:
        st.header("📈 Dataset Statistics")
        if df_merged is not None:
            st.metric("Total Tracks", len(df_merged))
            st.metric("Artists", df_merged['main_artist'].nunique())
            st.metric("Super Genres", df_merged['super_genre'].nunique())
            st.metric("Avg Popularity", f"{df_merged['popularity'].mean():.1f}")
    
    # Genre distribution
    if df_merged is not None:
        st.header("🎼 Genre Distribution")
        genre_counts = df_merged['super_genre'].value_counts()
        fig = px.bar(x=genre_counts.values, y=genre_counts.index, 
                     orientation='h', title="Super Genre Distribution")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def popularity_predictor(models, df_merged):
    """Popularity prediction page"""
    st.title("🎯 Popularity Predictor")
    st.markdown("Predict track popularity based on audio features")
    st.markdown("---")
    
    # Feature input
    features = create_feature_input()
    
    if st.button("🔮 Predict Popularity", type="primary"):
        try:
            # Scale features
            scaled_features = models['scaler'].transform(features)
            
            # Predict
            prediction = models['popularity'].predict(scaled_features)[0]
            confidence = models['popularity'].predict_proba(scaled_features).max()
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"🎵 Predicted Popularity: **{prediction:.1f}/100**")
                st.info(f"🎯 Confidence: **{confidence:.2%}**")
            
            with col2:
                # Popularity gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prediction,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Popularity Score"},
                    gauge = {'axis': {'range': [None, 100]},
                             'bar': {'color': "darkblue"},
                             'steps': [{'range': [0, 50], 'color': "lightgray"},
                                       {'range': [50, 80], 'color': "yellow"},
                                       {'range': [80, 100], 'color': "green"}],
                             'threshold': {'line': {'color': "red", 'width': 4},
                                           'thickness': 0.75, 'value': 90}}))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # SHAP explanation
            if 'shap_popularity' in models:
                st.subheader("🔍 Feature Importance (SHAP)")
                try:
                    # Create SHAP explanation plot
                    shap_values = models['explainer_popularity'].shap_values(scaled_features)
                    st.info("SHAP values show how each feature contributes to the prediction")
                    
                    # Feature importance bar chart
                    feature_importance = dict(zip(AUDIO_FEATURES, shap_values[0]))
                    importance_df = pd.DataFrame(list(feature_importance.items()), 
                                               columns=['Feature', 'SHAP Value'])
                    importance_df = importance_df.reindex(importance_df['SHAP Value'].abs().sort_values(ascending=False).index)
                    
                    fig = px.bar(importance_df, x='SHAP Value', y='Feature', 
                                orientation='h', title="Feature Impact on Prediction",
                                color='SHAP Value', color_continuous_scale='RdBu')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning("Could not generate SHAP explanation")
        
        except Exception as e:
            st.error(f"Prediction error: {e}")

def genre_classifier(models, df_merged):
    """Genre classification page"""
    st.title("🎼 Genre Classifier")
    st.markdown("Classify tracks into super and fine genres")
    st.markdown("---")
    
    # Feature input
    features = create_feature_input()
    
    if st.button("🔍 Classify Genre", type="primary"):
        try:
            # Scale features
            scaled_features = models['scaler'].transform(features)
            
            # Predict super genre
            super_pred = models['super_genre'].predict(scaled_features)[0]
            super_proba = models['super_genre'].predict_proba(scaled_features)[0]
            super_confidence = super_proba.max()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"🎵 Super Genre: **{super_pred.upper()}**")
                st.info(f"🎯 Confidence: **{super_confidence:.2%}**")
                
                # Super genre probabilities
                super_proba_df = pd.DataFrame({
                    'Genre': SUPER_GENRES,
                    'Probability': super_proba
                }).sort_values('Probability', ascending=False).head(5)
                
                fig = px.bar(super_proba_df, x='Probability', y='Genre', 
                            orientation='h', title="Top 5 Super Genre Predictions")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Fine genre prediction
                if super_pred in models['fine_genre']:
                    fine_pred = models['fine_genre'][super_pred].predict(scaled_features)[0]
                    fine_proba = models['fine_genre'][super_pred].predict_proba(scaled_features)[0]
                    fine_confidence = fine_proba.max()
                    
                    st.success(f"🎶 Fine Genre: **{fine_pred.upper()}**")
                    st.info(f"🎯 Confidence: **{fine_confidence:.2%}**")
                    
                    # Fine genre probabilities (top 5)
                    fine_classes = models['fine_genre'][super_pred].classes_
                    fine_proba_df = pd.DataFrame({
                        'Genre': fine_classes,
                        'Probability': fine_proba
                    }).sort_values('Probability', ascending=False).head(5)
                    
                    fig = px.bar(fine_proba_df, x='Probability', y='Genre', 
                                orientation='h', title=f"Top 5 {super_pred.title()} Sub-genres")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Fine genre model for {super_pred} not available")
        
        except Exception as e:
            st.error(f"Classification error: {e}")

def recommendation_system(df_merged, feature_matrix):
    """Music recommendation page"""
    st.title("🎵 Music Recommendation System")
    st.markdown("Get personalized music recommendations")
    st.markdown("---")
    
    # Load recommendation function
    recommend_func = load_recommender_function()
    
    if df_merged is not None and feature_matrix is not None and recommend_func is not None:
        # Track selection
        st.subheader("🎧 Select a Track")
        
        # Search functionality
        search_term = st.text_input("🔍 Search for a track or artist:")
        
        if search_term:
            filtered_df = df_merged[
                df_merged['track_name'].str.contains(search_term, case=False, na=False) |
                df_merged['main_artist'].str.contains(search_term, case=False, na=False)
            ]
        else:
            filtered_df = df_merged.head(100)  # Show first 100 tracks by default
        
        if len(filtered_df) > 0:
            # Create display format for dropdown
            track_options = []
            track_indices = []
            
            for idx, row in filtered_df.iterrows():
                display_text = f"{row['track_name']} - {row['main_artist']} ({row['super_genre']})"
                track_options.append(display_text)
                track_indices.append(idx)
            
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
                st.info(f"**Track:** {selected_track['track_name']}")
            with col2:
                st.info(f"**Artist:** {selected_track['main_artist']}")
            with col3:
                st.info(f"**Genre:** {selected_track['super_genre']}")
            
            # Recommendation parameters
            st.subheader("⚙️ Recommendation Settings")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                top_n = st.slider("Number of recommendations", 5, 50, 20)
                weight_similarity = st.slider("Similarity Weight", 0.0, 1.0, 0.4, 0.05)
            
            with col2:
                weight_cluster = st.slider("Cluster Weight", 0.0, 1.0, 0.2, 0.05)
                weight_genre = st.slider("Genre Weight", 0.0, 1.0, 0.3, 0.05)
            
            with col3:
                weight_artist = st.slider("Artist Weight", 0.0, 1.0, 0.05, 0.01)
                weight_popularity = st.slider("Popularity Weight", 0.0, 1.0, 0.25, 0.05)
            
            # Generate recommendations
            if st.button("🎵 Get Recommendations", type="primary"):
                try:
                    recommendations = recommend_func(
                        input_index=selected_track_idx,
                        feature_matrix=feature_matrix,
                        df_merged=df_merged,
                        top_n=top_n,
                        weight_similarity=weight_similarity,
                        weight_cluster=weight_cluster,
                        weight_genre=weight_genre,
                        weight_artist=weight_artist,
                        weight_popularity=weight_popularity
                    )
                    
                    st.subheader(f"🎵 Top {top_n} Recommendations")
                    
                    # Display recommendations table
                    recommendations_display = recommendations.copy()
                    recommendations_display['final_score'] = recommendations_display['final_score'].round(3)
                    recommendations_display.columns = ['Track Name', 'Artist', 'Genre', 'Score']
                    
                    st.dataframe(recommendations_display, use_container_width=True, height=600)
                    
                    # Recommendations by genre
                    genre_dist = recommendations['super_genre'].value_counts()
                    if len(genre_dist) > 1:
                        fig = px.pie(values=genre_dist.values, names=genre_dist.index, 
                                    title="Recommended Tracks by Genre")
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Recommendation error: {e}")
        else:
            st.warning("No tracks found. Try a different search term.")
    else:
        if recommend_func is None:
            st.error("Recommendation function not loaded properly")
        else:
            st.error("Data not loaded properly")

def clustering_view(tsne_data, pca_data, df_merged):
    """3D clustering visualization page"""
    st.title("🎨 Clustering Analysis")
    st.markdown("Explore music patterns through 3D visualizations")
    st.markdown("---")
    
    if tsne_data is not None and pca_data is not None:
        # Choose visualization type
        viz_type = st.selectbox("Choose Visualization:", ["t-SNE", "PCA"])
        
        if viz_type == "t-SNE":
            data = tsne_data
            title = "t-SNE 3D Clustering"
        else:
            data = pca_data
            title = "PCA 3D Clustering"
        
        # Color options
        color_option = st.selectbox("Color by:", ["cluster", "super_genre", "popularity"])
        
        # Create 3D plot
        if color_option == "popularity":
            fig = px.scatter_3d(
                data, x='x', y='y', z='z',
                color='popularity',
                title=title,
                color_continuous_scale='Viridis',
                hover_data=['track_name', 'main_artist'] if 'track_name' in data.columns else None
            )
        else:
            fig = px.scatter_3d(
                data, x='x', y='y', z='z',
                color=color_option,
                title=title,
                hover_data=['track_name', 'main_artist'] if 'track_name' in data.columns else None
            )
        
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster statistics
        if df_merged is not None:
            st.subheader("📊 Cluster Statistics")
            
            if 'cluster' in df_merged.columns:
                cluster_stats = df_merged.groupby('cluster').agg({
                    'popularity': 'mean',
                    'track_name': 'count',
                    'super_genre': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
                }).round(2)
                cluster_stats.columns = ['Avg Popularity', 'Track Count', 'Dominant Genre']
                st.dataframe(cluster_stats, use_container_width=True)
    else:
        st.error("Clustering data not available")

def trend_analysis(df_merged):
    """Trend analysis page"""
    st.title("📈 Trend Analysis")
    st.markdown("Analyze music trends over time")
    st.markdown("---")
    
    if df_merged is not None:
        # Check if we have year data (you might need to extract from album_name or track_id)
        st.info("Note: Trend analysis requires temporal data. If your dataset includes release dates, we can show popularity trends over time.")
        
        # Genre popularity comparison
        st.subheader("🎼 Genre Popularity Comparison")
        genre_popularity = df_merged.groupby('super_genre')['popularity'].agg(['mean', 'count']).round(2)
        genre_popularity.columns = ['Average Popularity', 'Track Count']
        genre_popularity = genre_popularity.sort_values('Average Popularity', ascending=False)
        
        fig = px.bar(
            x=genre_popularity.index,
            y=genre_popularity['Average Popularity'],
            title="Average Popularity by Super Genre"
        )
        fig.update_xaxis(title="Super Genre")
        fig.update_yaxis(title="Average Popularity")
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(genre_popularity, use_container_width=True)
        
        # Audio feature analysis by genre
        st.subheader("🎵 Audio Features by Genre")
        selected_genres = st.multiselect("Select genres to compare:", SUPER_GENRES, default=SUPER_GENRES[:5])
        
        if selected_genres:
            feature_comparison = df_merged[df_merged['super_genre'].isin(selected_genres)].groupby('super_genre')[AUDIO_FEATURES].mean()
            
            fig = px.heatmap(
                feature_comparison.T,
                title="Audio Features Heatmap by Genre",
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Data not loaded")

def main():
    """Main application"""
    
    st.title("🎵 Music Analysis Dashboard")
    
    try:
        # Add debug information
        st.info("🔧 Debug: Starting app initialization...")
        
        # Load data with error handling
        try:
            st.info("📥 Loading data files...")
            df_merged, feature_matrix, tsne_data, pca_data = load_data()
            if df_merged is not None:
                st.success(f"✅ Data loaded successfully! Dataset shape: {df_merged.shape}")
            else:
                st.error("❌ Failed to load data")
                return
        except Exception as e:
            st.error(f"❌ Data loading error: {str(e)}")
            st.stop()
        
        # Load models with error handling
        try:
            st.info("🤖 Loading ML models...")
            models = load_models()
            if models:
                st.success(f"✅ Models loaded successfully! Found {len(models)} model components")
            else:
                st.warning("⚠️ No models loaded")
        except Exception as e:
            st.error(f"❌ Model loading error: {str(e)}")
            models = {}
        
        # Sidebar navigation
        st.sidebar.title("🎵 Navigation")
        pages = {
            "🏠 Home": "home",
            "🎯 Popularity Predictor": "popularity",
            "🎼 Genre Classifier": "genre",
            "🎵 Recommendations": "recommendations",
            "🎨 Clustering View": "clustering",
            "📈 Trend Analysis": "trends"
        }
        
        selected_page = st.sidebar.selectbox("Select Page:", list(pages.keys()))
        page_key = pages[selected_page]
        
        # Display selected page with error handling
        try:
            if page_key == "home":
                home_page(df_merged)
            elif page_key == "popularity":
                if models and 'popularity' in models:
                    popularity_predictor(models, df_merged)
                else:
                    st.error("Popularity prediction models not available")
            elif page_key == "genre":
                if models and 'super_genre' in models:
                    genre_classifier(models, df_merged)
                else:
                    st.error("Genre classification models not available")
            elif page_key == "recommendations":
                recommendation_system(df_merged, feature_matrix)
            elif page_key == "clustering":
                clustering_view(tsne_data, pca_data, df_merged)
            elif page_key == "trends":
                trend_analysis(df_merged)
        except Exception as e:
            st.error(f"❌ Error in {selected_page}: {str(e)}")
            st.exception(e)  # This will show the full traceback
        
        # Footer
        st.sidebar.markdown("---")
        st.sidebar.markdown("**🎵 Music Analysis Dashboard**")
        st.sidebar.markdown("Built with Streamlit")
        
    except Exception as e:
        st.error(f"❌ Critical error in main(): {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()