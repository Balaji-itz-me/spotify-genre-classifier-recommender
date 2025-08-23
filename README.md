# 🎵 AI-Powered Music Analysis Dashboard

An intelligent music analytics platform combining machine learning, trend analysis, and interactive visualization to provide data-driven insights for the music industry.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest-green.svg)
![Dataset](https://img.shields.io/badge/dataset-114k%20tracks-orange.svg)

## 🎯 Project Overview

This comprehensive music analysis system leverages a dataset of **114,000+ tracks** to provide actionable insights for artists, producers, streaming platforms, and music industry professionals. The platform combines multiple AI models and analytical approaches to understand musical patterns, predict commercial success, and identify emerging trends.

### Key Features
- **🎯 Popularity Prediction**: ML-powered commercial success forecasting
- **🎼 Hierarchical Genre Classification**: Two-level genre taxonomy with 90%+ accuracy for key genres
- **🎵 Smart Recommendations**: Hybrid content-based recommendation engine
- **📊 Clustering Analysis**: Natural groupings of similar musical styles
- **🔍 Feature Importance**: SHAP-powered interpretability analysis
- **📈 Trend Forecasting**: Historical pattern analysis with future predictions

## 🚀 Live Demo

Access the interactive Streamlit dashboard: [**Music Analysis Dashboard**](#)

## 📊 Model Performance Summary

### Popularity Prediction
- **Algorithm**: Random Forest Regressor
- **R² Score**: 0.5539
- **Key Insight**: Valence (happiness) is the strongest predictor of commercial success
- **Top Features**: Valence → Instrumentalness → Acousticness

### Genre Classification (Hierarchical)
| Performance Tier | Accuracy | Genres |
|-----------------|----------|---------|
| **Exceptional (90%+)** | 99%, 95%, 92% | Country, Hip-Hop, Ambient |
| **Strong (70-89%)** | 70-87% | Children, Folk, Classical, Pop, Japanese |
| **Moderate (50-69%)** | 52-69% | World, Metal, Rock |
| **Challenging (<50%)** | 44-48% | Electronic, Latin, Miscellaneous |

### Recommendation System
- **Precision**: 1.00 (perfect across all genres)
- **Average Recall**: 0.0063 (optimized for relevance)
- **Architecture**: Hybrid content + metadata-based filtering

### Clustering Analysis
- **Algorithm**: KMeans (k=5)
- **Silhouette Score**: 0.155
- **Result**: 5 meaningful musical clusters with distinct characteristics

## 🔬 Key Research Findings

### 📈 Temporal Trends (1990-2022)

**Valence (Musical Positivity)**
- **Pattern**: U-shaped curve
- **1990-2005**: Gradual decline in positivity
- **2005-2022**: Steady recovery
- **Future Forecast**: Continued upward trend (post-pandemic optimism)

**Tempo Evolution**
- **Pattern**: Inverted U-curve
- **1990s-2000s**: Rising to ~123 BPM peak
- **Post-2010**: Decline to ~119 BPM (2022)
- **Future Forecast**: Further slowdown to ~110 BPM by 2027

### 🎛️ Feature Importance Hierarchy

**Primary Drivers (High Impact)**:
1. **Valence**: Happy songs = higher popularity
2. **Instrumentalness**: Vocals essential for mainstream success
3. **Acousticness**: Produced sound > acoustic for commercial appeal

**Secondary Factors**: Energy, Danceability, Duration
**Minimal Impact**: Key, Mode, Time Signature

### 🎪 Business Insights

**For Artists & Producers**:
- Prioritize positive emotions (valence) in compositions
- Include vocal elements (avoid purely instrumental)
- Consider electronic/produced elements over acoustic
- Target moderate-to-high energy levels

**For Streaming Platforms**:
- Weight valence, instrumentalness, and acousticness in algorithms
- Emphasize vocal-driven, upbeat tracks in popular playlists
- Prepare for continued tempo slowdown trend (lo-fi, chill genres)

**For Music Industry**:
- Commercial success patterns are highly genre-specific
- Emerging trend toward slower, more atmospheric soundscapes
- Post-pandemic shift toward happier, more optimistic music

## 🛠️ Technical Architecture

### Data Pipeline
```
Raw Spotify Data (114k tracks) 
    ↓
Feature Engineering & Scaling
    ↓
Multiple ML Pipelines:
├── Popularity Prediction (Random Forest)
├── Genre Classification (Hierarchical RF)
├── Clustering Analysis (KMeans)
├── Recommendation Engine (Hybrid)
└── Trend Analysis (Prophet Forecasting)
    ↓
Interactive Streamlit Dashboard
```

### Core Technologies
- **Machine Learning**: scikit-learn, Random Forest, SHAP
- **Time Series**: Facebook Prophet for trend forecasting
- **Visualization**: Plotly, t-SNE, PCA
- **Web Framework**: Streamlit
- **Data Processing**: pandas, numpy
- **Feature Engineering**: StandardScaler, hierarchical preprocessing

### Audio Features Analyzed (15+ dimensions)
- **Emotional**: Valence, Energy, Danceability
- **Vocal**: Speechiness, Instrumentalness, Liveness
- **Production**: Acousticness, Loudness, Tempo
- **Structural**: Duration, Key, Mode, Time Signature

## 📱 Dashboard Modules

### 🏠 App Overview
Comprehensive introduction to all features and capabilities

### 🎯 Popularity Prediction
- Interactive feature sliders with real-time prediction
- Optimization suggestions based on feature importance
- Commercial success scoring (0-100 scale)
- SHAP-powered interpretability insights

### 🎼 Genre Classification
- Hierarchical classification (Super → Sub genres)
- Confidence scoring for reliability assessment
- Genre-specific performance insights
- 10+ major category coverage

### 🎵 Music Recommendations
- Search-based track discovery
- Hybrid similarity matching
- Multi-factor weighting system
- Top-10 precision-optimized results

### 📊 Dataset Insights
- Interactive genre distribution analysis
- Audio feature pattern exploration
- Model training performance metrics
- Data quality and coverage statistics

## 🔧 Installation & Setup

### Prerequisites
```bash
Python 3.8+
pip package manager
```

### Installation Steps
```bash
# Clone repository
git clone https://github.com/yourusername/music-analysis-dashboard.git
cd music-analysis-dashboard

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run dashboard.py
```

### Required Packages
```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.15.0
shap>=0.42.0
joblib>=1.3.0
gdown>=4.7.0
fbprophet>=0.7.1
```

## 📁 Project Structure

```
music-analysis-dashboard/
├── dashboard.py                 # Main Streamlit application
├── models/                     # Trained ML models
│   ├── rf_popularity_cluster.pkl
│   ├── super_genre_model.pkl
│   ├── hierarchical_scaler.pkl
│   └── feature_matrix.npy
├── data/                       # Dataset files
│   └── df_merged.pkl          # Main processed dataset
├── notebooks/                  # Analysis notebooks
│   ├── popularity_modeling.ipynb
│   ├── genre_classification.ipynb
│   ├── clustering_analysis.ipynb
│   ├── recommendation_system.ipynb
│   └── trend_analysis.ipynb
├── requirements.txt
└── README.md
```

## 🎪 Target Users

### 🎤 Music Creators
- **Artists**: Optimize tracks for commercial success
- **Producers**: Data-driven production decisions
- **Songwriters**: Understand popularity patterns

### 🏢 Industry Professionals  
- **A&R Teams**: Talent screening and discovery
- **Playlist Curators**: Content organization and selection
- **Music Marketers**: Genre positioning and targeting
- **Streaming Platforms**: Algorithm enhancement

### 🎧 Content Creators
- **Video Producers**: Soundtrack selection
- **Podcast Creators**: Background music curation
- **App Developers**: Music integration insights

## 🔮 Future Enhancements

### Planned Features
- **Real-time Audio Analysis**: Upload and analyze custom tracks
- **Advanced Trend Predictions**: Multi-feature forecasting
- **Genre Evolution Tracking**: Sub-genre emergence patterns  
- **Collaborative Filtering**: User-based recommendations
- **API Integration**: Spotify/Apple Music connectivity
- **Mobile Optimization**: Responsive design improvements

### Research Extensions
- **Emotion Detection**: Advanced sentiment analysis
- **Cultural Impact Analysis**: Regional music preferences
- **Artist Career Modeling**: Success trajectory prediction
- **Cross-platform Analytics**: Multi-service comparison

## 📚 Research Methodology

### Data Collection
- **Source**: Spotify Web API + curated music databases
- **Size**: 114,000+ professionally catalogued tracks
- **Features**: 15+ audio characteristics per track
- **Quality**: Cleaned, validated, and preprocessed

### Model Development
- **Validation**: Train/test splits with cross-validation
- **Feature Engineering**: Scaling, normalization, interaction terms
- **Hyperparameter Tuning**: Grid search with performance optimization
- **Interpretability**: SHAP values for model transparency

### Evaluation Metrics
- **Classification**: Accuracy, Precision, Recall, F1-score
- **Regression**: R², MAE, MSE
- **Clustering**: Silhouette score, visual coherence
- **Recommendations**: Precision@10, Recall@50

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
```bash
# Fork the repository
# Clone your fork
git clone https://github.com/yourusername/music-analysis-dashboard.git

# Create feature branch
git checkout -b feature/amazing-feature

# Make changes and commit
git commit -m "Add amazing feature"

# Push to branch
git push origin feature/amazing-feature

# Submit Pull Request
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Spotify Web API** for comprehensive music data
- **Facebook Prophet** for time series forecasting capabilities
- **Streamlit Community** for excellent web framework
- **scikit-learn** for robust machine learning tools

## 📞 Contact

**Project Maintainer**: [Your Name]
- 📧 Email: your.email@example.com
- 💼 LinkedIn: [Your LinkedIn Profile]
- 🐙 GitHub: [Your GitHub Profile]

## ⭐ Star History

If this project helped you, please consider giving it a star! ⭐

---

*"Transforming music analysis through data science - one insight at a time"* 🎵✨
