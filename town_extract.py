#!/usr/bin/env python3
"""
Batch Building Coordinate Extractor
Process multiple towns from CSV or single town via command line arguments

Usage:
  # Batch mode (reads TownData.csv by default)
  python batch_building_extractor.py --batch
  python batch_building_extractor.py --batch --input towns.csv --default-distance 10
  
  # Single town mode
  python batch_building_extractor.py --lat -16.95 --lon 122.86 --distance 5 --name "Beagle Bay"

Expected CSV format (TownData.csv):
  AGIL CODE,Community Name,Latitude,Longitude,State,LGA,ABS Remoteness
  Or with distance_km column:
  Community Name,Latitude,Longitude,distance_km,State,LGA,ABS Remoteness
"""

import argparse
import math
import os
import pandas as pd
import requests
import time
import json
import gzip
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Any
import sys
import re

# Global verbose flag
VERBOSE = False

def vprint(message: str, level: str = "INFO"):
    """Verbose print function"""
    if VERBOSE or level in ["ERROR", "WARNING"]:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")

def sanitize_filename(name: str) -> str:
    """Convert community name to safe filename"""
    # Remove special characters and replace spaces with underscores
    safe_name = re.sub(r'[^\w\s-]', '', name)
    safe_name = re.sub(r'[-\s]+', '_', safe_name)
    return safe_name.lower().strip('_')

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points on Earth"""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return 6371.0 * c

def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
    """Convert lat/lon to tile numbers at given zoom level"""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    x = int((lon_deg + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (x, y)

def tile_to_quadkey(x: int, y: int, zoom: int) -> str:
    """Convert tile coordinates to QuadKey"""
    quadkey = ""
    for i in range(zoom, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (x & mask) != 0:
            digit += 1
        if (y & mask) != 0:
            digit += 2
        quadkey += str(digit)
    return quadkey

def get_quadkeys_for_area(center_lat: float, center_lon: float, radius_km: float, zoom: int = 9) -> List[str]:
    """Get all QuadKeys that cover a circular area around a center point"""
    lat_deg_per_km = 1.0 / 111.0
    lon_deg_per_km = 1.0 / (111.0 * math.cos(math.radians(center_lat)))
    
    lat_offset = radius_km * lat_deg_per_km
    lon_offset = radius_km * lon_deg_per_km
    
    min_lat = center_lat - lat_offset
    max_lat = center_lat + lat_offset
    min_lon = center_lon - lon_offset
    max_lon = center_lon + lon_offset
    
    min_x, max_y = deg2num(min_lat, min_lon, zoom)
    max_x, min_y = deg2num(max_lat, max_lon, zoom)
    
    quadkeys = []
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            quadkey = tile_to_quadkey(x, y, zoom)
            quadkeys.append(quadkey)
    
    return quadkeys

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
    """Download building data files for the specified QuadKeys"""
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
            
            # Skip if already downloaded
            if os.path.exists(filename):
                downloaded_files.append(filename)
                continue
            
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

def process_building_file(filepath: str, center_lat: float, center_lon: float, max_distance_km: float) -> List[Dict[str, Any]]:
    """Process a single building data file and extract buildings within distance"""
    buildings_in_range = []
    
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
                        distance = calculate_distance(center_lat, center_lon, center_building_lat, center_building_lon)
                        
                        if distance <= max_distance_km:
                            properties = building.get('properties', {})
                            if isinstance(properties, dict) and 'properties' in properties:
                                properties = properties['properties']
                            
                            height = properties.get('height', -1) if properties else -1
                            confidence = properties.get('confidence', -1) if properties else -1
                            
                            # Calculate building area
                            building_area = calculate_polygon_area(polygon_coords)
                            building_type = classify_building_by_area(building_area)
                            
                            building_record = {
                                'latitude': center_building_lat,
                                'longitude': center_building_lon,
                                'height': height,
                                'distance_km': distance,
                                'area_sqm': building_area,
                                'building_type': building_type,
                                'confidence': confidence,
                                'quadkey': os.path.basename(filepath).replace('quadkey_', '').replace('.csv.gz', ''),
                                'polygon_points': len(polygon_coords)
                            }
                            
                            buildings_in_range.append(building_record)
                        
                    except json.JSONDecodeError:
                        continue
                    except Exception:
                        continue
        
    except Exception as e:
        vprint(f"❌ Error processing file {filepath}: {e}", "ERROR")
    
    return buildings_in_range

def calculate_polygon_area(polygon_coords: List[List[float]]) -> float:
    """Calculate approximate area of polygon in square meters"""
    if len(polygon_coords) < 3:
        return 0
    
    area_deg = 0
    n = len(polygon_coords)
    
    for i in range(n):
        j = (i + 1) % n
        area_deg += polygon_coords[i][0] * polygon_coords[j][1]
        area_deg -= polygon_coords[j][0] * polygon_coords[i][1]
    
    area_deg = abs(area_deg) / 2.0
    area_sqm = area_deg * 10000000  # Rough approximation
    
    return max(area_sqm, 1)

def classify_building_by_area(area_sqm: float) -> str:
    """Classify building type based on floor area"""
    if area_sqm < 50:
        return "small_structure"
    elif area_sqm < 200:
        return "residential"
    elif area_sqm < 1000:
        return "commercial"
    else:
        return "community"

def save_buildings_to_csv(buildings: List[Dict[str, Any]], output_file: str) -> Dict[str, Any]:
    """Save buildings to CSV and return summary statistics"""
    if not buildings:
        return {'count': 0, 'types': {}, 'distance_range': (0, 0), 'height_stats': None}
    
    df = pd.DataFrame(buildings)
    df = df.sort_values('distance_km')
    df['building_id'] = range(1, len(df) + 1)
    
    df['description'] = df.apply(lambda row: 
        f"Type: {row['building_type']}, "
        f"Area: {row['area_sqm']:.1f}sqm, "
        f"Distance: {row['distance_km']:.2f}km, "
        f"Height: {row['height']}m, "
        f"Confidence: {row['confidence']}, "
        f"QuadKey: {row['quadkey']}", axis=1)
    
    output_df = df[['building_id', 'latitude', 'longitude', 'height', 'description']].copy()
    output_df['latitude'] = output_df['latitude'].round(6)
    output_df['longitude'] = output_df['longitude'].round(6)
    
    output_df.to_csv(output_file, index=False)
    
    # Generate summary statistics
    type_counts = df['building_type'].value_counts().to_dict()
    distance_range = (df['distance_km'].min(), df['distance_km'].max())
    
    height_buildings = df[df['height'] > 0]
    height_stats = None
    if len(height_buildings) > 0:
        height_stats = {
            'count': len(height_buildings),
            'average': height_buildings['height'].mean(),
            'range': (height_buildings['height'].min(), height_buildings['height'].max())
        }
    
    return {
        'count': len(buildings),
        'types': type_counts,
        'distance_range': distance_range,
        'height_stats': height_stats
    }

def save_buildings_to_kml(buildings: List[Dict[str, Any]], kml_file: str, center_lat: float, center_lon: float, radius_km: float, community_name: str):
    """Save buildings to KML format for Google Earth visualization"""
    if not buildings:
        return
    
    buildings.sort(key=lambda x: x['distance_km'])
    
    kml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
    <name>{community_name} - Building Footprints</name>
    <description>Building locations for {community_name} extracted from Microsoft dataset</description>
    
    <!-- Style definitions -->
    <Style id="small_structure">
        <IconStyle>
            <color>ff0000ff</color>
            <scale>0.8</scale>
            <Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>
        </IconStyle>
    </Style>
    <Style id="residential">
        <IconStyle>
            <color>ff00ff00</color>
            <scale>1.0</scale>
            <Icon><href>http://maps.google.com/mapfiles/kml/shapes/homegardenbusiness.png</href></Icon>
        </IconStyle>
    </Style>
    <Style id="commercial">
        <IconStyle>
            <color>ff00ffff</color>
            <scale>1.2</scale>
            <Icon><href>http://maps.google.com/mapfiles/kml/shapes/shopping.png</href></Icon>
        </IconStyle>
    </Style>
    <Style id="community">
        <IconStyle>
            <color>ffff0000</color>
            <scale>1.4</scale>
            <Icon><href>http://maps.google.com/mapfiles/kml/shapes/schools.png</href></Icon>
        </IconStyle>
    </Style>
    <Style id="search_center">
        <IconStyle>
            <color>ffff00ff</color>
            <scale>2.0</scale>
            <Icon><href>http://maps.google.com/mapfiles/kml/shapes/target.png</href></Icon>
        </IconStyle>
    </Style>
    
    <!-- Search center point -->
    <Placemark>
        <name>{community_name} Center</name>
        <description>{community_name} search center: {radius_km}km radius</description>
        <styleUrl>#search_center</styleUrl>
        <Point>
            <coordinates>{center_lon},{center_lat},0</coordinates>
        </Point>
    </Placemark>
    
    <!-- Search radius circle -->
    <Placemark>
        <name>{community_name} Search Radius ({radius_km}km)</name>
        <description>Search area boundary for {community_name}</description>
        <Style>
            <LineStyle>
                <color>7fff0000</color>
                <width>2</width>
            </LineStyle>
            <PolyStyle>
                <color>1fff0000</color>
            </PolyStyle>
        </Style>
        <Polygon>
            <outerBoundaryIs>
                <LinearRing>
                    <coordinates>'''
    
    # Generate circle coordinates
    circle_points = []
    for i in range(36):
        angle = i * 10 * math.pi / 180
        lat_offset = (radius_km / 111.0) * math.cos(angle)
        lon_offset = (radius_km / 111.0) * math.sin(angle) / math.cos(math.radians(center_lat))
        circle_lat = center_lat + lat_offset
        circle_lon = center_lon + lon_offset
        circle_points.append(f"{circle_lon},{circle_lat},0")
    
    circle_points.append(circle_points[0])
    kml_content += "\n                        ".join(circle_points)
    
    kml_content += f'''
                    </coordinates>
                </LinearRing>
            </outerBoundaryIs>
        </Polygon>
    </Placemark>
    
    <!-- Buildings folder -->
    <Folder>
        <name>{community_name} Buildings ({len(buildings)})</name>
        <description>Building footprints for {community_name} by type</description>
        '''
    
    # Add each building
    for i, building in enumerate(buildings, 1):
        building_type = building['building_type']
        kml_content += f'''
        <Placemark>
            <name>{community_name} Building {i}</name>
            <description><![CDATA[
                <b>{community_name} Building #{i}</b><br/>
                Type: {building_type}<br/>
                Area: {building['area_sqm']:.1f} sqm<br/>
                Distance: {building['distance_km']:.3f} km<br/>
                Height: {building['height']}m<br/>
                Confidence: {building['confidence']}<br/>
                QuadKey: {building['quadkey']}<br/>
                Coordinates: {building['latitude']:.6f}, {building['longitude']:.6f}
            ]]></description>
            <styleUrl>#{building_type}</styleUrl>
            <Point>
                <coordinates>{building['longitude']},{building['latitude']},0</coordinates>
            </Point>
        </Placemark>'''
    
    kml_content += '''
    </Folder>
</Document>
</kml>'''
    
    with open(kml_file, 'w', encoding='utf-8') as f:
        f.write(kml_content)

def process_single_town(community_name: str, lat: float, lon: float, distance: float, 
                       australia_links: pd.DataFrame, output_dir: str, zoom: int = 9) -> Dict[str, Any]:
    """Process a single community and return results"""
    vprint(f"🏘️  Processing {community_name}...")
    vprint(f"    📍 Location: ({lat:.6f}, {lon:.6f})")
    vprint(f"    📏 Search radius: {distance}km")
    
    try:
        # Step 1: Get QuadKeys
        target_quadkeys = get_quadkeys_for_area(lat, lon, distance, zoom)
        vprint(f"    🗺️  Generated {len(target_quadkeys)} QuadKeys")
        
        # Step 2: Find available data
        available_quadkeys = find_available_quadkeys(target_quadkeys, australia_links)
        if not available_quadkeys:
            vprint(f"    ❌ No building data available for {community_name}", "WARNING")
            return {
                'community_name': community_name,
                'status': 'no_data',
                'buildings': [],
                'stats': {'count': 0, 'types': {}, 'distance_range': (0, 0), 'height_stats': None}
            }
        
        vprint(f"    📦 Found {len(available_quadkeys)} data files")
        
        # Step 3: Download data
        download_dir = "temp_buildings"
        downloaded_files = download_building_files(available_quadkeys, australia_links, download_dir)
        
        if not downloaded_files:
            vprint(f"    ❌ Failed to download data for {community_name}", "ERROR")
            return {
                'community_name': community_name,
                'status': 'download_failed',
                'buildings': [],
                'stats': {'count': 0, 'types': {}, 'distance_range': (0, 0), 'height_stats': None}
            }
        
        # Step 4: Process buildings
        all_buildings = []
        for filepath in downloaded_files:
            buildings = process_building_file(filepath, lat, lon, distance)
            all_buildings.extend(buildings)
        
        vprint(f"    🏠 Found {len(all_buildings)} buildings")
        
        # Step 5: Save files
        safe_name = sanitize_filename(community_name)
        
        # Save CSV
        csv_file = os.path.join(output_dir, 'csv', f"{safe_name}_buildings.csv")
        stats = save_buildings_to_csv(all_buildings, csv_file)
        
        # Save KML
        kml_file = os.path.join(output_dir, 'kml', f"{safe_name}_buildings.kml")
        save_buildings_to_kml(all_buildings, kml_file, lat, lon, distance, community_name)
        
        vprint(f"    ✅ {community_name} completed: {len(all_buildings)} buildings")
        
        return {
            'community_name': community_name,
            'status': 'success',
            'buildings': all_buildings,
            'stats': stats,
            'files': {
                'csv': csv_file,
                'kml': kml_file
            }
        }
        
    except Exception as e:
        vprint(f"    ❌ Error processing {community_name}: {e}", "ERROR")
        return {
            'community_name': community_name,
            'status': 'error',
            'error': str(e),
            'buildings': [],
            'stats': {'count': 0, 'types': {}, 'distance_range': (0, 0), 'height_stats': None}
        }

def generate_summary_report(results: List[Dict[str, Any]], output_file: str):
    """Generate a comprehensive summary report"""
    total_buildings = sum(len(r['buildings']) for r in results)
    successful_towns = [r for r in results if r['status'] == 'success']
    failed_towns = [r for r in results if r['status'] != 'success']
    
    report = f"""
🏗️  BUILDING EXTRACTION SUMMARY REPORT
{'=' * 50}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 OVERALL STATISTICS
{'=' * 30}
Total communities processed: {len(results)}
Successful extractions: {len(successful_towns)}
Failed extractions: {len(failed_towns)}
Total buildings found: {total_buildings:,}

📋 DETAILED RESULTS
{'=' * 30}
"""
    
    for result in results:
        community = result['community_name']
        status = result['status']
        
        if status == 'success':
            stats = result['stats']
            building_count = stats['count']
            types = stats['types']
            dist_range = stats['distance_range']
            
            report += f"""
🏘️  {community}
   Status: ✅ SUCCESS
   Buildings found: {building_count:,}
   Distance range: {dist_range[0]:.3f} - {dist_range[1]:.3f} km
   Building types:"""
            
            for building_type, count in types.items():
                report += f"\n      {building_type}: {count}"
            
            if stats['height_stats']:
                hs = stats['height_stats']
                report += f"\n   Height data: {hs['count']} buildings, avg {hs['average']:.1f}m"
            
            files = result.get('files', {})
            if 'csv' in files:
                report += f"\n   📁 CSV: {files['csv']}"
            if 'kml' in files:
                report += f"\n   🗺️  KML: {files['kml']}"
        
        else:
            report += f"""
🏘️  {community}
   Status: ❌ {status.upper()}"""
            if 'error' in result:
                report += f"\n   Error: {result['error']}"
    
    # Add building type summary
    if successful_towns:
        all_types = {}
        for result in successful_towns:
            for building_type, count in result['stats']['types'].items():
                all_types[building_type] = all_types.get(building_type, 0) + count
        
        report += f"""

📈 BUILDING TYPE SUMMARY
{'=' * 30}
Total buildings by type across all communities:"""
        
        for building_type, count in sorted(all_types.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_buildings) * 100
            report += f"\n   {building_type}: {count:,} ({percentage:.1f}%)"
    
    # Add recommendations
    report += f"""

💡 RECOMMENDATIONS
{'=' * 30}"""
    
    if failed_towns:
        report += f"\n⚠️  {len(failed_towns)} communities need manual survey or alternative data sources:"
        for result in failed_towns:
            report += f"\n   - {result['community_name']} ({result['status']})"
    
    if total_buildings > 0:
        avg_buildings_per_town = total_buildings / len(successful_towns) if successful_towns else 0
        report += f"\n📊 Average buildings per community: {avg_buildings_per_town:.1f}"
        
        if avg_buildings_per_town < 50:
            report += "\n💭 Consider larger search radius for small communities"
        
        report += f"\n🎯 Use KML files in Google Earth to validate building locations"
        report += f"\n📋 CSV files are ready for Starlink coverage planning software"
    
    report += f"""

🔧 NEXT STEPS
{'=' * 30}
1. Review failed communities and consider manual surveys
2. Import CSV files into coverage planning tools
3. Validate results using KML files in Google Earth
4. Cross-reference with field surveys during site visits

✅ Data extraction complete and ready for Starlink deployment planning!
"""
    
    # Save report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Also print to console
    print(report)

def cleanup_temp_files(download_dir: str):
    """Clean up temporary downloaded files"""
    try:
        if os.path.exists(download_dir):
            import shutil
            shutil.rmtree(download_dir)
            vprint(f"🧹 Cleaned up temporary directory: {download_dir}")
    except Exception as e:
        vprint(f"❌ Error cleaning up {download_dir}: {e}", "WARNING")

def load_towns_from_csv(csv_file: str, default_distance: float) -> pd.DataFrame:
    """Load towns from CSV and handle different column formats"""
    try:
        df = pd.read_csv(csv_file)
        
        # Print available columns for debugging
        vprint(f"Available columns in CSV: {list(df.columns)}")
        
        # Check for required columns and map them
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
        
        # Map distance column (optional)
        if 'distance_km' in df.columns:
            required_mapping['distance_km'] = 'distance_km'
        elif 'Distance' in df.columns:
            required_mapping['distance_km'] = 'Distance'
        elif 'radius' in df.columns:
            required_mapping['distance_km'] = 'radius'
        else:
            # No distance column found, will use default
            vprint(f"No distance column found, using default distance: {default_distance}km")
            df['distance_km'] = default_distance
            required_mapping['distance_km'] = 'distance_km'
        
        # Create new dataframe with standardized column names
        standardized_df = pd.DataFrame()
        for standard_name, original_name in required_mapping.items():
            standardized_df[standard_name] = df[original_name]
        
        # Add additional columns if they exist (for reference)
        additional_cols = ['State', 'LGA', 'ABS Remoteness', 'AGIL CODE']
        for col in additional_cols:
            if col in df.columns:
                standardized_df[col] = df[col]
        
        vprint(f"Successfully loaded {len(standardized_df)} communities from {csv_file}")
        return standardized_df
        
    except Exception as e:
        raise ValueError(f"Error loading CSV file {csv_file}: {e}")

def main():
    global VERBOSE
    
    parser = argparse.ArgumentParser(
        description="Extract building coordinates for multiple communities from CSV or single community",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Batch mode (processes TownData.csv)
  python batch_building_extractor.py --batch --verbose
  
  # Custom input file with default distance
  python batch_building_extractor.py --batch --input my_towns.csv --default-distance 2
  
  # Single community mode
  python batch_building_extractor.py --lat -12.324 --lon 133.056 --distance 5 --name "Gunbalanya"
  
Expected CSV format (flexible column names):
  Community Name,Latitude,Longitude[,distance_km]
  or
  AGIL CODE,Community Name,Latitude,Longitude,State,LGA,ABS Remoteness
        """
    )
    
    # Mode selection
    parser.add_argument('--batch', action='store_true',
                        help='Batch mode: process communities from CSV file')
    
    # Batch mode arguments
    parser.add_argument('--input', type=str, default='TownData.csv',
                        help='Input CSV file for batch mode (default: TownData.csv)')
    parser.add_argument('--default-distance', type=float, default=2.0,
                        help='Default search radius in km if not specified in CSV (default: 10.0)')
    
    # Single community mode arguments
    parser.add_argument('--lat', type=float,
                        help='Latitude of center point (single community mode)')
    parser.add_argument('--lon', type=float,
                        help='Longitude of center point (single community mode)')
    parser.add_argument('--distance', type=float,
                        help='Search radius in kilometers (single community mode)')
    parser.add_argument('--name', type=str,
                        help='Community name (single community mode)')
    
    # Common arguments
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory (default: output)')
    parser.add_argument('--zoom', type=int, default=9,
                        help='QuadKey zoom level for tile selection (default: 9)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--keep-temp', action='store_true',
                        help='Keep temporary downloaded files (for debugging)')
    
    args = parser.parse_args()
    VERBOSE = args.verbose
    
    # Validate arguments
    if not args.batch and not all([args.lat, args.lon, args.distance, args.name]):
        print("❌ Error: Either use --batch mode or provide --lat, --lon, --distance, and --name")
        sys.exit(1)
    
    # Create output directories
    output_dir = args.output_dir
    Path(output_dir).mkdir(exist_ok=True)
    Path(os.path.join(output_dir, 'csv')).mkdir(exist_ok=True)
    Path(os.path.join(output_dir, 'kml')).mkdir(exist_ok=True)
    
    print("🏗️  Microsoft Building Coordinate Extractor - Batch Mode")
    print("=" * 60)
    
    try:
        # Load Australia dataset once
        vprint("📡 Loading Microsoft Australia building dataset...")
        australia_links = load_australia_dataset_links()
        vprint(f"✅ Loaded {len(australia_links)} Australian dataset files")
        
        results = []
        
        if args.batch:
            # Batch mode: process communities from CSV
            vprint(f"📋 Reading communities from: {args.input}")
            
            if not os.path.exists(args.input):
                print(f"❌ Input file not found: {args.input}")
                print("💡 Create a CSV file with columns: Community Name,Latitude,Longitude[,distance_km]")
                sys.exit(1)
            
            try:
                communities_df = load_towns_from_csv(args.input, args.default_distance)
            except ValueError as e:
                print(f"❌ Error loading CSV: {e}")
                sys.exit(1)
            
            vprint(f"📊 Found {len(communities_df)} communities to process")
            
            for _, row in communities_df.iterrows():
                result = process_single_town(
                    row['community_name'], 
                    row['latitude'], 
                    row['longitude'], 
                    row['distance_km'],
                    australia_links,
                    output_dir,
                    args.zoom
                )
                results.append(result)
        
        else:
            # Single community mode
            result = process_single_town(
                args.name,
                args.lat,
                args.lon,
                args.distance,
                australia_links,
                output_dir,
                args.zoom
            )
            results.append(result)
        
        # Generate summary report
        summary_file = os.path.join(output_dir, 'summary_report.txt')
        generate_summary_report(results, summary_file)
        
        # Cleanup
        if not args.keep_temp:
            cleanup_temp_files("temp_buildings")
        
        print(f"\n🎉 Processing complete!")
        print(f"📁 Output directory: {output_dir}/")
        print(f"📊 Summary report: {summary_file}")
        
    except KeyboardInterrupt:
        vprint("\n❌ Process interrupted by user", "ERROR")
        sys.exit(1)
    except Exception as e:
        vprint(f"\n❌ Fatal error: {e}", "ERROR")
        if VERBOSE:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()