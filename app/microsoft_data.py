#!/usr/bin/env python3
"""
Microsoft Building Data Module
Handles downloading and processing Microsoft Global Building Footprints data
"""

import os
import requests
import time
import json
import gzip
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

from .utils import vprint, get_quadkeys_for_area, calculate_distance, calculate_polygon_area
from .osm_integration import get_osm_metadata_cached, reverse_geocode_cached, classify_building

def load_australia_dataset_links() -> pd.DataFrame:
    """Load the Microsoft Australia building dataset links"""
    try:
        dataset_links_url = "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"
        df_links = pd.read_csv(dataset_links_url)
        
        australia_links = df_links[df_links['Location'] == 'Australia']
        if len(australia_links) == 0:
            australia_links = df_links[df_links['Country'] == 'Australia']
        
        if len(australia_links) == 0:
            raise ValueError("No Australian building data found in dataset")
        
        return australia_links
        
    except Exception as e:
        vprint(f"❌ Error loading dataset links: {e}", "ERROR")
        raise

def find_available_quadkeys(target_quadkeys: List[str], australia_links: pd.DataFrame) -> List[str]:
    """Find which target QuadKeys have available data"""
    try:
        available_quadkeys = set(australia_links['QuadKey'].astype(str))
        found_quadkeys = []
        
        for qk in target_quadkeys:
            qk_str = str(qk)
            if qk_str in available_quadkeys:
                found_quadkeys.append(qk_str)
            else:
                matches = [aq for aq in available_quadkeys if aq.startswith(qk_str) or qk_str.startswith(aq)]
                if matches:
                    found_quadkeys.extend(matches[:3])
        
        return list(set(found_quadkeys))
        
    except Exception as e:
        vprint(f"❌ Error checking QuadKeys: {e}", "ERROR")
        return []

def download_building_files(quadkeys: List[str], australia_links: pd.DataFrame, download_dir: str = "temp_buildings") -> List[str]:
    """Download building data files for the specified QuadKeys (with caching)"""
    Path(download_dir).mkdir(exist_ok=True)
    downloaded_files = []
    
    for i, quadkey in enumerate(quadkeys, 1):
        try:
            matching_rows = australia_links[australia_links['QuadKey'].astype(str) == str(quadkey)]
            if len(matching_rows) == 0:
                continue
            
            file_info = matching_rows.iloc[0]
            url = file_info['Url']
            filename = f"{download_dir}/quadkey_{quadkey}.csv.gz"
            
            # Skip if already downloaded (built-in caching)
            if os.path.exists(filename):
                downloaded_files.append(filename)
                vprint(f"    📦 Using cached file: {quadkey}", "DEBUG")
                continue
            
            vprint(f"    ⬇️  Downloading QuadKey {quadkey} ({i}/{len(quadkeys)})")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            downloaded_files.append(filename)
            time.sleep(0.1)  # Be nice to the server
            
        except Exception as e:
            vprint(f"❌ Failed to download {quadkey}: {e}", "ERROR")
            continue
    
    return downloaded_files

def process_building_file(filepath: str, center_lat: float, center_lon: float, max_distance_km: float, cache_manager=None) -> List[Dict[str, Any]]:
    """Process a single building data file and extract buildings within distance"""
    buildings_in_range = []
    building_count = 0
    
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        building = json.loads(line.strip())
                        
                        geometry = building.get('geometry', {})
                        if geometry.get('type') != 'Polygon':
                            continue
                        
                        coordinates = geometry.get('coordinates', [])
                        if not coordinates or not coordinates[0]:
                            continue
                        
                        # Calculate building center
                        polygon_coords = coordinates[0]
                        lats = [coord[1] for coord in polygon_coords]
                        lons = [coord[0] for coord in polygon_coords]
                        center_building_lat = sum(lats) / len(lats)
                        center_building_lon = sum(lons) / len(lons)
                        
                        # Calculate distance
                        distance_m = calculate_distance(center_lat, center_lon, center_building_lat, center_building_lon)
                        distance_km = distance_m / 1000.0
                        
                        if distance_km <= max_distance_km:
                            building_count += 1
                            
                            properties = building.get('properties', {})
                            if isinstance(properties, dict) and 'properties' in properties:
                                properties = properties['properties']
                            
                            height = properties.get('height', -1) if properties else -1
                            confidence = properties.get('confidence', -1) if properties else -1
                            
                            # Calculate building area
                            building_area = calculate_polygon_area(polygon_coords)
                            
                            # Get OSM metadata (cached)
                            osm_metadata = get_osm_metadata_cached(center_building_lat, center_building_lon, cache_manager)
                            
                            # Get address (cached)
                            address = reverse_geocode_cached(center_building_lat, center_building_lon, cache_manager)
                            
                            # Rate limiting only for non-cached requests
                            if cache_manager:
                                cache_stats = cache_manager.get_cache_stats()
                                # Only delay if we had cache misses (actual API calls)
                                if (cache_manager.stats['osm_cache_misses'] + cache_manager.stats['geocoding_cache_misses']) % 10 == 1:
                                    time.sleep(0.1)
                            else:
                                time.sleep(0.2)  # Default rate limiting without cache
                            
                            # Classify building
                            building_category = classify_building(osm_metadata)
                            
                            # Create description
                            desc_parts = []
                            if osm_metadata['name']:
                                desc_parts.append(f"Name: {osm_metadata['name']}")
                            if osm_metadata['amenity']:
                                desc_parts.append(f"Amenity: {osm_metadata['amenity']}")
                            if osm_metadata['building_type'] and osm_metadata['building_type'] != 'yes':
                                desc_parts.append(f"Building: {osm_metadata['building_type']}")
                            if address:
                                desc_parts.append(f"Address: {address}")
                            desc_parts.append(f"Category: {building_category}")
                            desc_parts.append(f"Area: {building_area:.1f}sqm")
                            desc_parts.append(f"Distance: {distance_km:.2f}km")
                            if height > 0:
                                desc_parts.append(f"Height: {height}m")
                            
                            description = " | ".join(desc_parts)
                            
                            building_record = {
                                'latitude': center_building_lat,
                                'longitude': center_building_lon,
                                'name': osm_metadata['name'],
                                'amenity': osm_metadata['amenity'],
                                'building_type': osm_metadata['building_type'],
                                'addr_street': osm_metadata['addr_street'],
                                'addr_housenumber': osm_metadata['addr_housenumber'],
                                'address': address,
                                'height': height,
                                'building_use': osm_metadata['building_use'],
                                'landuse': osm_metadata['landuse'],
                                'shop': osm_metadata['shop'],
                                'office': osm_metadata['office'],
                                'distance_km': distance_km,
                                'distance_m': distance_m,
                                'area_sqm': building_area,
                                'building_category': building_category,
                                'confidence': confidence,
                                'quadkey': os.path.basename(filepath).replace('quadkey_', '').replace('.csv.gz', ''),
                                'polygon_points': len(polygon_coords),
                                'description': description
                            }
                            
                            buildings_in_range.append(building_record)
                            
                            # Progress indicator
                            if building_count % 10 == 0:
                                vprint(f"    🏠 Processed {building_count} buildings...", "DEBUG")
                        
                    except json.JSONDecodeError:
                        continue
                    except Exception:
                        continue
        
    except Exception as e:
        vprint(f"❌ Error processing file {filepath}: {e}", "ERROR")
    
    return buildings_in_range

def cleanup_temp_files(download_dir: str):
    """Clean up temporary downloaded files"""
    try:
        if os.path.exists(download_dir):
            import shutil
            shutil.rmtree(download_dir)
            vprint(f"🧹 Cleaned up temporary directory: {download_dir}")
    except Exception as e:
        vprint(f"❌ Error cleaning up {download_dir}: {e}", "WARNING")