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
import gc
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Music Analysis Dashboard - Diagnostic Mode",
    page_icon="🔧",
    layout="wide"
)

# Constants
DRIVE_FILE_ID = "12aGkuSuDpVO_gLdcal-fyUJYLGqcPFyI"
SUPER_GENRES = ['rock', 'pop', 'electronic', 'metal', 'hiphop', 'japanese', 
                'latin', 'classical', 'country', 'folk', 'ambient', 'world', 
                'children', 'other', 'misc']
AUDIO_FEATURES = ['danceability', 'energy', 'loudness', 'speechiness', 
                  'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

def get_file_size_mb(filepath):
    """Get file size in MB"""
    try:
        return os.path.getsize(filepath) / (1024 * 1024)
    except:
        return 0

def memory_usage():
    """Get current memory usage info"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)  # MB
    except:
        return "Unknown"

@st.cache_resource
def download_files_diagnostic():
    """Download files with detailed diagnostics"""
    
    st.header("🔧 Diagnostic Mode - File Download")
    
    try:
        # Memory check before download
        st.info(f"💾 Memory usage before download: {memory_usage()} MB")
        
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "music_files.zip")
        
        st.info(f"📂 Temp directory: {temp_dir}")
        
        # Download with progress
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("📥 Starting download...")
        progress_bar.progress(10)
        
        try:
            gdown.download(url, zip_path, quiet=False, fuzzy=True)
            progress_bar.progress(50)
            status_text.text("✅ Download completed")
        except Exception as e:
            st.error(f"❌ Download failed: {e}")
            return None
        
        # Check file size
        file_size = get_file_size_mb(zip_path)
        st.success(f"📦 Downloaded file: {file_size:.1f} MB")
        
        if file_size < 100:  # Expect ~476MB
            st.error("⚠️ Downloaded file seems too small - may be corrupted")
            return None
        
        # Extract files
        status_text.text("📦 Extracting files...")
        progress_bar.progress(70)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                st.info(f"📋 Found {len(file_list)} files in archive")
                zip_ref.extractall(temp_dir)
            progress_bar.progress(90)
            status_text.text("✅ Extraction completed")
        except Exception as e:
            st.error(f"❌ Extraction failed: {e}")
            return None
        
        # Find base directory
        extracted_items = os.listdir(temp_dir)
        st.info(f"📁 Extracted items: {extracted_items}")
        
        # Look for the right directory
        base_candidates = [
            os.path.join(temp_dir, 'Spotify_app'),
            os.path.join(temp_dir, 'spotify_app'),
        ]
        
        # Add any subdirectories
        for item in extracted_items:
            item_path = os.path.join(temp_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                base_candidates.append(item_path)
        
        base_candidates.append(temp_dir)  # fallback
        
        base_dir = None
        for candidate in base_candidates:
            if os.path.exists(candidate):
                files_in_candidate = os.listdir(candidate)
                if 'df_merged.pkl' in files_in_candidate:
                    base_dir = candidate
                    st.success(f"✅ Found data directory: {candidate}")
                    break
        
        if base_dir is None:
            st.error("❌ Could not find directory with expected files")
            # Show what we have
            for candidate in base_candidates[:3]:  # Show first 3
                if os.path.exists(candidate):
                    files = os.listdir(candidate)
                    st.info(f"📁 {candidate}: {files[:10]}")  # First 10 files
            return None
        
        progress_bar.progress(100)
        status_text.text("🎉 Setup completed successfully!")
        
        # Final memory check
        st.info(f"💾 Memory usage after download: {memory_usage()} MB")
        
        return base_dir
        
    except Exception as e:
        st.error(f"❌ Critical error in download: {e}")
        st.code(traceback.format_exc())
        return None

def diagnose_files(base_dir):
    """Diagnose all files in the directory"""
    
    st.header("🔍 File Diagnostics")
    
    if not base_dir or not os.path.exists(base_dir):
        st.error("Base directory not available")
        return {}
    
    files_info = {}
    
    try:
        all_files = os.listdir(base_dir)
        st.info(f"📁 Total files found: {len(all_files)}")
        
        # Categorize files
        pkl_files = [f for f in all_files if f.endswith('.pkl')]
        npy_files = [f for f in all_files if f.endswith('.npy')]  
        other_files = [f for f in all_files if not f.endswith('.pkl') and not f.endswith('.npy')]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📊 PKL Files")
            for pkl_file in pkl_files[:10]:  # Show first 10
                file_path = os.path.join(base_dir, pkl_file)
                size_mb = get_file_size_mb(file_path)
                files_info[pkl_file] = {'path': file_path, 'size_mb': size_mb, 'type': 'pkl'}
                st.write(f"• {pkl_file} ({size_mb:.1f} MB)")
        
        with col2:
            st.subheader("🔢 NPY Files")
            for npy_file in npy_files:
                file_path = os.path.join(base_dir, npy_file)
                size_mb = get_file_size_mb(file_path)
                files_info[npy_file] = {'path': file_path, 'size_mb': size_mb, 'type': 'npy'}
                st.write(f"• {npy_file} ({size_mb:.1f} MB)")
        
        with col3:
            st.subheader("📄 Other Files")
            for other_file in other_files[:10]:  # Show first 10
                file_path = os.path.join(base_dir, other_file)
                size_mb = get_file_size_mb(file_path)
                files_info[other_file] = {'path': file_path, 'size_mb': size_mb, 'type': 'other'}
                st.write(f"• {other_file} ({size_mb:.1f} MB)")
        
        return files_info
        
    except Exception as e:
        st.error(f"Error diagnosing files: {e}")
        st.code(traceback.format_exc())
        return {}

def test_data_loading(files_info):
    """Test loading data files one by one"""
    
    st.header("📊 Data Loading Test")
    
    results = {}
    
    # Test df_merged.pkl
    if 'df_merged.pkl' in files_info:
        st.subheader("🔍 Testing df_merged.pkl")
        try:
            file_path = files_info['df_merged.pkl']['path']
            size_mb = files_info['df_merged.pkl']['size_mb']
            
            st.info(f"💾 Memory before loading: {memory_usage()} MB")
            st.info(f"📦 File size: {size_mb:.1f} MB")
            
            with st.spinner("Loading df_merged.pkl..."):
                df = pd.read_pickle(file_path)
                
            st.success(f"✅ Loaded successfully!")
            st.info(f"📊 Shape: {df.shape}")
            st.info(f"🏷️ Columns: {list(df.columns)[:10]}...")  # First 10 columns
            st.info(f"💾 Memory after loading: {memory_usage()} MB")
            
            results['df_merged'] = {'status': 'success', 'shape': df.shape, 'data': df}
            
            # Clean up memory
            del df
            gc.collect()
            
        except Exception as e:
            st.error(f"❌ Failed to load df_merged.pkl: {e}")
            results['df_merged'] = {'status': 'failed', 'error': str(e)}
    else:
        st.warning("⚠️ df_merged.pkl not found")
        results['df_merged'] = {'status': 'missing'}
    
    # Test feature_matrix.npy
    if 'feature_matrix.npy' in files_info:
        st.subheader("🔍 Testing feature_matrix.npy")
        try:
            file_path = files_info['feature_matrix.npy']['path']
            size_mb = files_info['feature_matrix.npy']['size_mb']
            
            st.info(f"💾 Memory before loading: {memory_usage()} MB")
            st.info(f"📦 File size: {size_mb:.1f} MB")
            
            with st.spinner("Loading feature_matrix.npy..."):
                matrix = np.load(file_path)
                
            st.success(f"✅ Loaded successfully!")
            st.info(f"📊 Shape: {matrix.shape}")
            st.info(f"📈 Data type: {matrix.dtype}")
            st.info(f"💾 Memory after loading: {memory_usage()} MB")
            
            results['feature_matrix'] = {'status': 'success', 'shape': matrix.shape}
            
            # Clean up memory
            del matrix
            gc.collect()
            
        except Exception as e:
            st.error(f"❌ Failed to load feature_matrix.npy: {e}")
            results['feature_matrix'] = {'status': 'failed', 'error': str(e)}
    else:
        st.warning("⚠️ feature_matrix.npy not found")
        results['feature_matrix'] = {'status': 'missing'}
    
    return results

def test_model_loading(files_info):
    """Test loading models one by one with detailed diagnostics"""
    
    st.header("🤖 Model Loading Test")
    
    # Critical models to test
    critical_models = {
        'rf_popularity_cluster.pkl': 'Popularity Predictor',
        'hierarchical_scaler.pkl': 'Feature Scaler',
        'super_genre_model.pkl': 'Super Genre Classifier'
    }
    
    results = {}
    
    for model_file, model_name in critical_models.items():
        st.subheader(f"🔍 Testing {model_name}")
        
        if model_file in files_info:
            try:
                file_path = files_info[model_file]['path']
                size_mb = files_info[model_file]['size_mb']
                
                st.info(f"📦 File: {model_file} ({size_mb:.1f} MB)")
                st.info(f"💾 Memory before loading: {memory_usage()} MB")
                
                # Try to load with timeout simulation
                with st.spinner(f"Loading {model_name}..."):
                    try:
                        model = joblib.load(file_path)
                        st.success(f"✅ {model_name} loaded successfully!")
                        st.info(f"🔧 Model type: {type(model)}")
                        st.info(f"💾 Memory after loading: {memory_usage()} MB")
                        
                        results[model_file] = {'status': 'success', 'type': str(type(model))}
                        
                        # Clean up immediately
                        del model
                        gc.collect()
                        
                    except Exception as model_error:
                        st.error(f"❌ Model loading failed: {model_error}")
                        results[model_file] = {'status': 'failed', 'error': str(model_error)}
                        
            except Exception as e:
                st.error(f"❌ Error accessing {model_file}: {e}")
                results[model_file] = {'status': 'error', 'error': str(e)}
        else:
            st.warning(f"⚠️ {model_file} not found")
            results[model_file] = {'status': 'missing'}
        
        # Memory cleanup after each model
        gc.collect()
        st.info(f"🧹 Memory after cleanup: {memory_usage()} MB")
        st.markdown("---")
    
    return results

def test_fine_genre_models(files_info):
    """Test fine genre models"""
    
    st.header("🎼 Fine Genre Models Test")
    
    fine_genre_files = [f for f in files_info.keys() if f.startswith('fine_genre_model_')]
    
    st.info(f"Found {len(fine_genre_files)} fine genre model files")
    
    success_count = 0
    failed_count = 0
    
    for i, model_file in enumerate(fine_genre_files[:5]):  # Test only first 5
        genre = model_file.replace('fine_genre_model_', '').replace('.pkl', '')
        
        try:
            file_path = files_info[model_file]['path']
            size_mb = files_info[model_file]['size_mb']
            
            with st.spinner(f"Testing {genre} model ({i+1}/{min(5, len(fine_genre_files))})..."):
                model = joblib.load(file_path)
                st.success(f"✅ {genre} model OK ({size_mb:.1f} MB)")
                success_count += 1
                del model
                gc.collect()
                
        except Exception as e:
            st.error(f"❌ {genre} model failed: {str(e)[:100]}")
            failed_count += 1
    
    st.info(f"📊 Fine Genre Models: {success_count} success, {failed_count} failed")

def main():
    """Main diagnostic function"""
    
    st.title("🔧 Music Dashboard - Diagnostic Mode")
    st.markdown("This diagnostic version will help identify what's causing the crashes.")
    
    try:
        # Step 1: Download and extract files
        base_dir = download_files_diagnostic()
        
        if base_dir is None:
            st.error("❌ Cannot proceed - file download/extraction failed")
            st.stop()
        
        # Step 2: Diagnose files
        files_info = diagnose_files(base_dir)
        
        if not files_info:
            st.error("❌ Cannot proceed - file diagnosis failed")
            st.stop()
        
        # Step 3: Test data loading
        st.markdown("---")
        data_results = test_data_loading(files_info)
        
        # Step 4: Test critical model loading  
        st.markdown("---")
        model_results = test_model_loading(files_info)
        
        # Step 5: Test some fine genre models
        st.markdown("---")
        test_fine_genre_models(files_info)
        
        # Summary
        st.header("📋 Diagnostic Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Data Status")
            for data_name, result in data_results.items():
                if result['status'] == 'success':
                    st.success(f"✅ {data_name}: OK")
                elif result['status'] == 'failed':
                    st.error(f"❌ {data_name}: FAILED")
                else:
                    st.warning(f"⚠️ {data_name}: MISSING")
        
        with col2:
            st.subheader("🤖 Model Status")
            for model_file, result in model_results.items():
                model_name = model_file.replace('.pkl', '')
                if result['status'] == 'success':
                    st.success(f"✅ {model_name}: OK")
                elif result['status'] == 'failed':
                    st.error(f"❌ {model_name}: FAILED")
                else:
                    st.warning(f"⚠️ {model_name}: MISSING")
        
        # Final memory usage
        st.info(f"💾 Final memory usage: {memory_usage()} MB")
        
        # Recommendations based on results
        st.header("💡 Recommendations")
        
        failed_items = []
        for name, result in {**data_results, **model_results}.items():
            if result['status'] == 'failed':
                failed_items.append(name)
        
        if failed_items:
            st.warning(f"⚠️ Failed to load: {', '.join(failed_items)}")
            st.markdown("""
            **Possible solutions:**
            1. **Memory Issue**: Try running on a machine with more RAM
            2. **File Corruption**: Re-download the Google Drive file
            3. **Version Compatibility**: Update scikit-learn, pandas, numpy
            4. **Streamlit Cloud Limits**: Consider local deployment
            """)
        else:
            st.success("🎉 All critical components loaded successfully!")
            st.info("The original app should work. The issue might be in the complex UI rendering.")
    
    except Exception as e:
        st.error(f"❌ Diagnostic failed: {e}")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()