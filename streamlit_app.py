#!/usr/bin/env python3
"""
Streamlit Frontend for Building Coordinate Extractor
Interactive web interface for extracting building coordinates with OSM metadata
"""

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import zipfile
import io
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import time

# Import our modular components
try:
    from app.utils import vprint, sanitize_filename, get_quadkeys_for_area, escape_xml_text
    from app.cache_manager import CacheManager
    from app.microsoft_data import (
        load_australia_dataset_links, 
        find_available_quadkeys, 
        download_building_files, 
        process_building_file,
        cleanup_temp_files
    )
    from app.output_generators import (
        save_buildings_to_csv,
        save_buildings_to_kml, 
        generate_community_summary
    )
    from app.osm_integration import get_osm_metadata_cached, reverse_geocode_cached
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.error("Please ensure all module files are in the the sub folder app")
    st.stop()

# Configure Streamlit
st.set_page_config(
    page_title="Building Coordinate Extractor",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global cache manager
if 'cache_manager' not in st.session_state:
    st.session_state.cache_manager = CacheManager("cache")

def load_csv_data(csv_file) -> pd.DataFrame:
    """Load and validate CSV data"""
    try:
        df = pd.read_csv(csv_file)
        
        required_mapping = {}
        
        # Map community name column
        if 'Community Name' in df.columns:
            required_mapping['community_name'] = 'Community Name'
        elif 'town_name' in df.columns:
            required_mapping['community_name'] = 'town_name'
        elif 'name' in df.columns:
            required_mapping['community_name'] = 'name'
        else:
            raise ValueError("Could not find community name column. Expected 'Community Name', 'town_name', or 'name'")
        
        # Map latitude column
        if 'Latitude' in df.columns:
            required_mapping['latitude'] = 'Latitude'
        elif 'latitude' in df.columns:
            required_mapping['latitude'] = 'latitude'
        elif 'lat' in df.columns:
            required_mapping['latitude'] = 'lat'
        else:
            raise ValueError("Could not find latitude column. Expected 'Latitude', 'latitude', or 'lat'")
        
        # Map longitude column
        if 'Longitude' in df.columns:
            required_mapping['longitude'] = 'Longitude'
        elif 'longitude' in df.columns:
            required_mapping['longitude'] = 'longitude'
        elif 'lon' in df.columns:
            required_mapping['longitude'] = 'lon'
        else:
            raise ValueError("Could not find longitude column. Expected 'Longitude', 'longitude', or 'lon'")
        
        # Create standardized dataframe
        standardized_df = pd.DataFrame()
        for standard_name, original_name in required_mapping.items():
            standardized_df[standard_name] = df[original_name]
        
        # Add additional columns if they exist
        additional_cols = ['State', 'LGA', 'ABS Remoteness', 'AGIL CODE']
        for col in additional_cols:
            if col in df.columns:
                standardized_df[col] = df[col]
        
        return standardized_df
        
    except Exception as e:
        raise ValueError(f"Error processing CSV file: {e}")

def process_single_community(community_name: str, lat: float, lon: float, distance: float, 
                           australia_links: pd.DataFrame, progress_callback=None) -> Dict[str, Any]:
    """Process a single community with progress updates"""
    
    if progress_callback:
        progress_callback(f"Processing {community_name}...", 0.1)
    
    # Check cache first
    cached_buildings = st.session_state.cache_manager.get_building_area(lat, lon, distance)
    if cached_buildings is not None:
        if progress_callback:
            progress_callback(f"Loaded {community_name} from cache", 1.0)
        
        return {
            'community_name': community_name,
            'status': 'success',
            'buildings': cached_buildings,
            'from_cache': True
        }
    
    try:
        if progress_callback:
            progress_callback(f"Getting QuadKeys for {community_name}...", 0.2)
        
        # Step 1: Get QuadKeys
        target_quadkeys = get_quadkeys_for_area(lat, lon, distance, zoom=9)
        
        if progress_callback:
            progress_callback(f"Finding available data files...", 0.3)
        
        # Step 2: Find available data
        available_quadkeys = find_available_quadkeys(target_quadkeys, australia_links)
        if not available_quadkeys:
            return {
                'community_name': community_name,
                'status': 'no_data',
                'buildings': []
            }
        
        if progress_callback:
            progress_callback(f"Downloading {len(available_quadkeys)} data files...", 0.4)
        
        # Step 3: Download data
        download_dir = "temp_buildings"
        downloaded_files = download_building_files(available_quadkeys, australia_links, download_dir)
        
        if not downloaded_files:
            return {
                'community_name': community_name,
                'status': 'download_failed',
                'buildings': []
            }
        
        if progress_callback:
            progress_callback(f"Processing buildings with OSM metadata...", 0.6)
        
        # Step 4: Process buildings with progress updates
        all_buildings = []
        total_files = len(downloaded_files)
        
        for i, filepath in enumerate(downloaded_files):
            buildings = process_building_file(filepath, lat, lon, distance, st.session_state.cache_manager)
            all_buildings.extend(buildings)
            
            if progress_callback:
                file_progress = 0.6 + (0.3 * (i + 1) / total_files)
                progress_callback(f"Processed file {i+1}/{total_files}, found {len(all_buildings)} buildings so far...", file_progress)
        
        if progress_callback:
            progress_callback(f"Finalizing {community_name}...", 0.95)
        
        # Cache the results
        if all_buildings:
            st.session_state.cache_manager.save_building_area(lat, lon, distance, all_buildings)
        
        if progress_callback:
            progress_callback(f"Completed {community_name}: {len(all_buildings)} buildings found", 1.0)
        
        return {
            'community_name': community_name,
            'status': 'success',
            'buildings': all_buildings,
            'from_cache': False
        }
        
    except Exception as e:
        return {
            'community_name': community_name,
            'status': 'error',
            'error': str(e),
            'buildings': []
        }

def create_building_map(buildings: List[Dict[str, Any]], community_name: str, center_lat: float, center_lon: float) -> folium.Map:
    """Create a folium map with building markers"""
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles='OpenStreetMap'
    )
    
    # Add center marker
    folium.Marker(
        [center_lat, center_lon],
        popup=f"{community_name} Center",
        tooltip=f"{community_name} Search Center",
        icon=folium.Icon(color='purple', icon='star')
    ).add_to(m)
    
    # Color scheme for building categories
    color_map = {
        'Residential': 'green',
        'Community': 'red', 
        'Commercial': 'blue'
    }
    
    icon_map = {
        'Residential': 'home',
        'Community': 'university',
        'Commercial': 'shopping-cart'
    }
    
    # Add building markers
    for building in buildings:
        category = building.get('building_category', 'Residential')
        color = color_map.get(category, 'gray')
        icon = icon_map.get(category, 'circle')
        
        # Create popup content
        name = building.get('name', 'Unnamed Building')
        if not name or name.lower() in ['', 'nan', 'none']:
            name = f"Building #{building.get('building_id', 'Unknown')}"
        
        popup_content = f"""
        <b>{escape_xml_text(name)}</b><br/>
        Category: {category}<br/>
        Distance: {building['distance_km']:.2f}km<br/>
        Area: {building['area_sqm']:.1f}sqm
        """
        
        if building.get('amenity'):
            popup_content += f"<br/>Amenity: {building['amenity']}"
        if building.get('address'):
            popup_content += f"<br/>Address: {building['address'][:50]}..."
        
        folium.Marker(
            [building['latitude'], building['longitude']],
            popup=folium.Popup(popup_content, max_width=250),
            tooltip=f"{name} ({category})",
            icon=folium.Icon(color=color, icon=icon)
        ).add_to(m)
    
    return m

def create_download_zip(communities_data: Dict, download_type: str = "all") -> bytes:
    """Create a zip file with community data"""
    
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        
        with tempfile.TemporaryDirectory() as temp_dir:
            
            for community_name, result in communities_data.items():
                if result['status'] != 'success':
                    continue
                
                buildings = result['buildings']
                if not buildings:
                    continue
                
                safe_name = sanitize_filename(community_name)
                
                # Create CSV
                csv_file = os.path.join(temp_dir, f"{safe_name}_buildings.csv")
                save_buildings_to_csv(buildings, csv_file)
                
                # Create KML (we need lat/lon from first result to get center)
                if buildings:
                    # Use the first building's data to estimate center
                    center_lat = sum(b['latitude'] for b in buildings) / len(buildings)
                    center_lon = sum(b['longitude'] for b in buildings) / len(buildings)
                    distance = max(b['distance_km'] for b in buildings) + 0.5  # Add buffer
                    
                    kml_file = os.path.join(temp_dir, f"{safe_name}_buildings.kml")
                    save_buildings_to_kml(buildings, kml_file, center_lat, center_lon, distance, community_name)
                    
                    # Create summary
                    stats = {
                        'count': len(buildings),
                        'types': pd.DataFrame(buildings)['building_category'].value_counts().to_dict(),
                        'distance_range': (min(b['distance_km'] for b in buildings), max(b['distance_km'] for b in buildings)),
                        'height_stats': None
                    }
                    
                    summary_file = os.path.join(temp_dir, f"{safe_name}_summary.txt")
                    generate_community_summary(buildings, stats, summary_file, community_name, center_lat, center_lon, distance)
                    
                    # Add files to zip
                    for file_path in [csv_file, kml_file, summary_file]:
                        if os.path.exists(file_path):
                            arcname = f"{safe_name}/{os.path.basename(file_path)}"
                            zip_file.write(file_path, arcname)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def main():
    # Header
    st.title("🏘️ Building Coordinate Extractor")
    st.markdown("Extract building coordinates with OpenStreetMap metadata for Australian communities")
    
    # Sidebar controls
    st.sidebar.header("🎛️ Controls")
    
    # Input method selection
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["📁 Upload CSV File", "✏️ Manual Entry"]
    )
    
    # Distance slider
    distance = st.sidebar.slider(
        "🎯 Search Radius (km)",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5,
        help="Radius around each community center to search for buildings"
    )
    
    # Cache controls
    st.sidebar.header("📦 Cache Controls")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🧹 Clear Cache"):
            st.session_state.cache_manager.clear_cache()
            st.sidebar.success("Cache cleared!")
    
    with col2:
        if st.button("📊 Cache Stats"):
            stats = st.session_state.cache_manager.get_cache_stats()
            st.sidebar.info(f"OSM: {stats['osm_hit_rate']:.1f}% hit rate\nGeo: {stats['geocoding_hit_rate']:.1f}% hit rate")
    
    # Main content area
    communities_to_process = []
    
    if input_method == "📁 Upload CSV File":
        st.header("📁 Upload CSV File")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="CSV should contain columns: Community Name, Latitude, Longitude"
        )
        
        if uploaded_file is not None:
            try:
                df = load_csv_data(uploaded_file)
                st.success(f"✅ Loaded {len(df)} communities from CSV")
                
                # Show preview
                with st.expander("📋 Preview Data"):
                    st.dataframe(df)
                
                # Add distance column
                df['distance_km'] = distance
                communities_to_process = df.to_dict('records')
                
            except Exception as e:
                st.error(f"❌ Error loading CSV: {e}")
                st.info("💡 Expected CSV format: Community Name, Latitude, Longitude")
    
    else:  # Manual Entry
        st.header("✏️ Manual Entry")
        
        col1, col2 = st.columns(2)
        
        with col1:
            community_name = st.text_input("Community Name", placeholder="e.g., Gunbalanya")
            latitude = st.number_input("Latitude", value=-12.324, format="%.6f")
        
        with col2:
            longitude = st.number_input("Longitude", value=133.056, format="%.6f")
            
        if community_name and latitude and longitude:
            communities_to_process = [{
                'community_name': community_name,
                'latitude': latitude,
                'longitude': longitude,
                'distance_km': distance
            }]
            
            st.success(f"✅ Ready to process: {community_name}")
    
    # Process button
    if communities_to_process:
        if st.button("🚀 Process Communities", type="primary"):
            
            # Initialize session state for results
            if 'processing_results' not in st.session_state:
                st.session_state.processing_results = {}
            
            try:
                # Load Australia dataset
                with st.spinner("📡 Loading Microsoft Australia building dataset..."):
                    australia_links = load_australia_dataset_links()
                
                st.success(f"✅ Loaded {len(australia_links)} Australian dataset files")
                
                # Process communities with progress bar
                total_communities = len(communities_to_process)
                
                overall_progress = st.progress(0)
                status_text = st.empty()
                
                results = {}
                
                for i, community in enumerate(communities_to_process):
                    community_name = community['community_name']
                    
                    # Update progress callback
                    def progress_callback(message, progress):
                        overall_progress.progress((i + progress) / total_communities)
                        status_text.text(f"[{i+1}/{total_communities}] {message}")
                    
                    result = process_single_community(
                        community_name,
                        community['latitude'],
                        community['longitude'], 
                        community['distance_km'],
                        australia_links,
                        progress_callback
                    )
                    
                    results[community_name] = result
                
                # Final progress update
                overall_progress.progress(1.0)
                status_text.text("✅ Processing complete!")
                
                # Save results to session state
                st.session_state.processing_results = results
                
                # Save cache
                st.session_state.cache_manager.save_all_caches()
                
                # Show summary
                successful = [r for r in results.values() if r['status'] == 'success']
                failed = [r for r in results.values() if r['status'] != 'success']
                cached = [r for r in successful if r.get('from_cache', False)]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Processed", len(results))
                with col2:
                    st.metric("Successful", len(successful))
                with col3:
                    st.metric("From Cache", len(cached))
                with col4:
                    total_buildings = sum(len(r['buildings']) for r in successful)
                    st.metric("Total Buildings", total_buildings)
                
                # Clean up temp files
                cleanup_temp_files("temp_buildings")
                
            except Exception as e:
                st.error(f"❌ Error during processing: {e}")
    
    # Results section
    if 'processing_results' in st.session_state and st.session_state.processing_results:
        
        st.header("📊 Results")
        
        successful_results = {k: v for k, v in st.session_state.processing_results.items() 
                            if v['status'] == 'success' and v['buildings']}
        
        if successful_results:
            
            # Community selector
            selected_community = st.selectbox(
                "🏘️ Select Community to View",
                options=list(successful_results.keys()),
                help="Choose a community to view its buildings on the map"
            )
            
            if selected_community:
                result = successful_results[selected_community]
                buildings = result['buildings']
                
                # Show community stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Buildings Found", len(buildings))
                with col2:
                    cache_indicator = "📦 (Cached)" if result.get('from_cache', False) else "🆕 (Fresh)"
                    st.metric("Data Source", cache_indicator)
                with col3:
                    if buildings:
                        categories = pd.DataFrame(buildings)['building_category'].value_counts()
                        top_category = categories.index[0]
                        st.metric("Top Category", f"{top_category} ({categories.iloc[0]})")
                
                # Create and display map
                if buildings:
                    # Calculate center from buildings
                    center_lat = sum(b['latitude'] for b in buildings) / len(buildings)
                    center_lon = sum(b['longitude'] for b in buildings) / len(buildings)
                    
                    building_map = create_building_map(buildings, selected_community, center_lat, center_lon)
                    
                    st.subheader(f"🗺️ {selected_community} Building Map")
                    st_folium(building_map, width=700, height=500, returned_objects=["last_object_clicked"])
                    
                    # Building category breakdown
                    st.subheader("📈 Building Categories")
                    category_df = pd.DataFrame(buildings)['building_category'].value_counts().reset_index()
                    category_df.columns = ['Category', 'Count']
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.dataframe(category_df, use_container_width=True)
                    with col2:
                        st.bar_chart(category_df.set_index('Category'))
            
            # Download section
            st.header("📥 Downloads")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if selected_community:
                    if st.button("📦 Download Current Community"):
                        current_data = {selected_community: successful_results[selected_community]}
                        zip_data = create_download_zip(current_data)
                        
                        safe_name = sanitize_filename(selected_community)
                        st.download_button(
                            label="⬇️ Download ZIP",
                            data=zip_data,
                            file_name=f"{safe_name}_buildings.zip",
                            mime="application/zip"
                        )
            
            with col2:
                if st.button("📦 Download All Communities"):
                    zip_data = create_download_zip(successful_results)
                    
                    st.download_button(
                        label="⬇️ Download All ZIP",
                        data=zip_data,
                        file_name="all_communities_buildings.zip",
                        mime="application/zip"
                    )
        
        else:
            st.warning("⚠️ No successful results with buildings found.")
            
            # Show failed results
            failed_results = {k: v for k, v in st.session_state.processing_results.items() 
                            if v['status'] != 'success'}
            
            if failed_results:
                st.subheader("❌ Failed Processing")
                for name, result in failed_results.items():
                    st.error(f"{name}: {result['status']} - {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()