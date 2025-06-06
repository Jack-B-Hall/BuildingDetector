#!/usr/bin/env python3
"""
Enhanced Community Building Extractor
Features:
- Custom CSV input support
- Data caching (save/load to avoid re-downloading)
- Address lookup using reverse geocoding
- Batch processing with progress tracking
"""

import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import json
import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import time
import logging
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
from xml.dom import minidom
import pickle
import hashlib
from urllib.parse import quote

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('building_extraction.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class CommunityBuildingExtractor:
    """
    Enhanced building extractor with caching and address lookup
    """
    
    def __init__(self, csv_file: str = "TownData.csv", output_dir: str = "output", 
                 cache_dir: str = "cache", default_radius: int = 1500, 
                 delay_between_requests: float = 2.0, enable_geocoding: bool = True,
                 cache_expiry_days: int = 30, force_refresh: bool = False):
        """
        Initialize the enhanced extractor
        
        Args:
            csv_file: Path to CSV file with community data
            output_dir: Base output directory 
            cache_dir: Directory for caching downloaded data
            default_radius: Default search radius in meters
            delay_between_requests: Delay between API requests
            enable_geocoding: Whether to lookup addresses for buildings
            cache_expiry_days: How many days to keep cached data
            force_refresh: Force refresh of cached data
        """
        self.csv_file = csv_file
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.default_radius = default_radius
        self.delay_between_requests = delay_between_requests
        self.enable_geocoding = enable_geocoding
        self.cache_expiry_days = cache_expiry_days
        self.force_refresh = force_refresh
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        (self.cache_dir / "buildings").mkdir(exist_ok=True)
        (self.cache_dir / "geocoding").mkdir(exist_ok=True)
        
        # Initialize statistics
        self.stats = {
            'total_communities': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'total_buildings_found': 0,
            'communities_with_buildings': 0,
            'communities_without_buildings': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'geocoding_success': 0,
            'geocoding_failed': 0
        }
        
        # Load community data
        self.communities_df = self.load_community_data()
        
        logger.info(f"🏘️ Enhanced extractor initialized for {len(self.communities_df)} communities")
        logger.info(f"📁 Output: {self.output_dir.absolute()}")
        logger.info(f"💾 Cache: {self.cache_dir.absolute()}")
        logger.info(f"🔍 Geocoding: {'Enabled' if self.enable_geocoding else 'Disabled'}")

    def load_community_data(self) -> pd.DataFrame:
        """Load and validate community data from CSV"""
        try:
            if not os.path.exists(self.csv_file):
                raise FileNotFoundError(f"CSV file not found: {self.csv_file}")
            
            df = pd.read_csv(self.csv_file)
            
            required_columns = ['Community Name', 'Latitude', 'Longitude']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Clean data
            df = df.dropna(subset=['Community Name', 'Latitude', 'Longitude'])
            
            # Create safe filenames
            df['safe_name'] = df['Community Name'].apply(self.create_safe_filename)
            
            logger.info(f"✅ Loaded {len(df)} valid communities from {self.csv_file}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading community data: {e}")
            raise

    @staticmethod
    def create_safe_filename(name: str) -> str:
        """Create filesystem-safe filename from community name"""
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_', '.')).strip()
        safe_name = safe_name.replace(' ', '_').replace('__', '_')
        return safe_name.lower()

    def create_cache_key(self, community_name: str, latitude: float, longitude: float, radius: int) -> str:
        """Create unique cache key for community building data"""
        key_string = f"{community_name}_{latitude}_{longitude}_{radius}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def get_cached_buildings(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Retrieve cached building data if available and not expired"""
        cache_file = self.cache_dir / "buildings" / f"{cache_key}.pkl"
        
        if not cache_file.exists() or self.force_refresh:
            return None
        
        try:
            # Check if cache is expired
            cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - cache_time > timedelta(days=self.cache_expiry_days):
                logger.info(f"🕐 Cache expired for {cache_key}")
                return None
            
            # Load cached data
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            self.stats['cache_hits'] += 1
            logger.info(f"💾 Cache hit: {cache_key}")
            return cached_data
            
        except Exception as e:
            logger.warning(f"⚠️ Error loading cache {cache_key}: {e}")
            return None

    def save_buildings_to_cache(self, cache_key: str, buildings_df: pd.DataFrame):
        """Save building data to cache"""
        try:
            cache_file = self.cache_dir / "buildings" / f"{cache_key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(buildings_df, f)
            
            logger.info(f"💾 Cached buildings data: {cache_key}")
            
        except Exception as e:
            logger.warning(f"⚠️ Error saving to cache {cache_key}: {e}")

    def extract_buildings_for_community(self, row: pd.Series) -> Optional[pd.DataFrame]:
        """Extract buildings for a single community with caching"""
        community_name = row['Community Name']
        latitude = row['Latitude']
        longitude = row['Longitude']
        
        logger.info(f"🏠 Processing: {community_name}")
        logger.info(f"📍 Coordinates: {latitude}, {longitude}")
        
        # Check cache first
        cache_key = self.create_cache_key(community_name, latitude, longitude, self.default_radius)
        cached_buildings = self.get_cached_buildings(cache_key)
        
        if cached_buildings is not None:
            logger.info(f"📦 Using cached data for {community_name}")
            return cached_buildings
        
        # No cache hit, extract fresh data
        self.stats['cache_misses'] += 1
        
        try:
            buildings_df = None
            
            # Approach 1: OSMnx point-based query
            try:
                center_point = (latitude, longitude)
                buildings = ox.features_from_point(
                    center_point,
                    tags={'building': True},
                    dist=self.default_radius
                )
                
                if not buildings.empty:
                    buildings_df = self.process_building_data(buildings, row)
                    logger.info(f"✅ OSMnx point query: {len(buildings_df)} buildings")
                
            except Exception as e:
                logger.warning(f"⚠️ OSMnx point query failed: {e}")
            
            # Approach 2: Overpass API direct query (if OSMnx failed or found few buildings)
            if buildings_df is None or len(buildings_df) < 3:
                try:
                    overpass_buildings = self.query_overpass_api(latitude, longitude, self.default_radius)
                    if overpass_buildings is not None and not overpass_buildings.empty:
                        if buildings_df is None or len(overpass_buildings) > len(buildings_df):
                            buildings_df = overpass_buildings
                            buildings_df = self.add_community_metadata(buildings_df, row)
                            logger.info(f"✅ Overpass API: {len(buildings_df)} buildings")
                
                except Exception as e:
                    logger.warning(f"⚠️ Overpass API query failed: {e}")
            
            # Add address lookup if enabled and buildings found
            if buildings_df is not None and not buildings_df.empty and self.enable_geocoding:
                buildings_df = self.add_addresses_to_buildings(buildings_df)
            
            # Cache the results
            if buildings_df is not None and not buildings_df.empty:
                self.save_buildings_to_cache(cache_key, buildings_df)
            
            # Delay between requests to be respectful to APIs
            time.sleep(self.delay_between_requests)
            
            return buildings_df
            
        except Exception as e:
            logger.error(f"❌ Error extracting buildings for {community_name}: {e}")
            return None

    def add_addresses_to_buildings(self, buildings_df: pd.DataFrame) -> pd.DataFrame:
        """Add address information to buildings using reverse geocoding"""
        logger.info(f"🔍 Looking up addresses for {len(buildings_df)} buildings...")
        
        addresses = []
        
        for idx, building in buildings_df.iterrows():
            lat = building.get('building_lat')
            lon = building.get('building_lon')
            
            if pd.isna(lat) or pd.isna(lon):
                addresses.append('')
                continue
            
            # Check geocoding cache
            geocode_key = f"{lat:.6f}_{lon:.6f}"
            cached_address = self.get_cached_address(geocode_key)
            
            if cached_address is not None:
                addresses.append(cached_address)
                continue
            
            # Perform reverse geocoding
            address = self.reverse_geocode(lat, lon)
            addresses.append(address)
            
            # Cache the result
            self.save_address_to_cache(geocode_key, address)
            
            # Small delay between geocoding requests
            time.sleep(0.5)
        
        buildings_df['address'] = addresses
        
        successful_geocodes = sum(1 for addr in addresses if addr)
        self.stats['geocoding_success'] += successful_geocodes
        self.stats['geocoding_failed'] += len(addresses) - successful_geocodes
        
        logger.info(f"🏠 Successfully geocoded {successful_geocodes}/{len(addresses)} addresses")
        
        return buildings_df

    def get_cached_address(self, geocode_key: str) -> Optional[str]:
        """Get cached geocoding result"""
        cache_file = self.cache_dir / "geocoding" / f"{geocode_key}.txt"
        
        if not cache_file.exists():
            return None
        
        try:
            # Check if cache is expired (shorter expiry for geocoding)
            cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - cache_time > timedelta(days=7):  # 7 day expiry for addresses
                return None
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
                
        except Exception:
            return None

    def save_address_to_cache(self, geocode_key: str, address: str):
        """Save geocoding result to cache"""
        try:
            cache_file = self.cache_dir / "geocoding" / f"{geocode_key}.txt"
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(address)
        except Exception as e:
            logger.warning(f"⚠️ Error caching address: {e}")

    def reverse_geocode(self, latitude: float, longitude: float) -> str:
        """Perform reverse geocoding using Nominatim"""
        try:
            # Use Nominatim (OpenStreetMap's geocoding service)
            url = "https://nominatim.openstreetmap.org/reverse"
            params = {
                'lat': latitude,
                'lon': longitude,
                'format': 'json',
                'addressdetails': 1,
                'zoom': 18,  # Building level detail
                'extratags': 1
            }
            
            headers = {
                'User-Agent': 'CommunityBuildingExtractor/1.0 (Contact: your-email@example.com)'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'address' in data:
                    addr_parts = []
                    address = data['address']
                    
                    # Build address string from components
                    if 'house_number' in address:
                        addr_parts.append(address['house_number'])
                    if 'road' in address:
                        addr_parts.append(address['road'])
                    elif 'street' in address:
                        addr_parts.append(address['street'])
                    
                    if 'suburb' in address:
                        addr_parts.append(address['suburb'])
                    elif 'city' in address:
                        addr_parts.append(address['city'])
                    elif 'town' in address:
                        addr_parts.append(address['town'])
                    
                    if 'state' in address:
                        addr_parts.append(address['state'])
                    
                    if 'postcode' in address:
                        addr_parts.append(address['postcode'])
                    
                    return ', '.join(addr_parts) if addr_parts else data.get('display_name', '')
                
                return data.get('display_name', '')
            
            return ''
            
        except Exception as e:
            logger.warning(f"⚠️ Geocoding failed for {latitude}, {longitude}: {e}")
            return ''

    def process_building_data(self, buildings_gdf: gpd.GeoDataFrame, community_row: pd.Series) -> pd.DataFrame:
        """Process raw building data from OSMnx"""
        if buildings_gdf.empty:
            return pd.DataFrame()
        
        buildings = buildings_gdf.copy()
        
        # Add community metadata
        buildings = self.add_community_metadata(buildings, community_row)
        
        # Extract building coordinates (centroid)
        buildings['building_lat'] = buildings.geometry.centroid.y
        buildings['building_lon'] = buildings.geometry.centroid.x
        
        # Calculate distance from community center
        center_lat, center_lon = community_row['Latitude'], community_row['Longitude']
        buildings['distance_from_center_m'] = buildings.apply(
            lambda row: self.calculate_distance(
                center_lat, center_lon, row['building_lat'], row['building_lon']
            ), axis=1
        )
        
        # Classify buildings
        buildings['building_category'] = buildings.apply(self.classify_building, axis=1)
        
        # Calculate area if possible
        try:
            buildings_projected = buildings.to_crs('EPSG:3577')  # Australian Albers
            buildings['area_sqm'] = buildings_projected.geometry.area
        except:
            buildings['area_sqm'] = None
        
        # Create building IDs
        buildings['building_id'] = range(1, len(buildings) + 1)
        
        # Select relevant columns for output
        output_columns = [
            'building_id', 'community_name', 'agil_code', 'state', 'lga', 
            'abs_remoteness', 'building_category', 'building_lat', 'building_lon',
            'distance_from_center_m', 'area_sqm', 'building', 'community_lat', 
            'community_lon', 'search_radius_m'
        ]
        
        # Add optional columns if they exist
        optional_columns = ['name', 'addr:street', 'addr:housenumber', 'amenity', 'landuse', 'height']
        for col in optional_columns:
            if col in buildings.columns:
                output_columns.append(col)
        
        available_columns = [col for col in output_columns if col in buildings.columns]
        return buildings[available_columns].copy()

    def add_community_metadata(self, buildings_df: pd.DataFrame, community_row: pd.Series) -> pd.DataFrame:
        """Add community metadata to buildings DataFrame"""
        buildings_df['community_name'] = community_row['Community Name']
        buildings_df['agil_code'] = community_row.get('AGIL CODE', '')
        buildings_df['state'] = community_row.get('State', '')
        buildings_df['lga'] = community_row.get('LGA', '')
        buildings_df['abs_remoteness'] = community_row.get('ABS Remoteness', '')
        buildings_df['community_lat'] = community_row['Latitude']
        buildings_df['community_lon'] = community_row['Longitude']
        buildings_df['search_radius_m'] = self.default_radius
        return buildings_df

    def query_overpass_api(self, latitude: float, longitude: float, radius_m: int) -> Optional[pd.DataFrame]:
        """Direct Overpass API query"""
        # Calculate bounding box
        lat_offset = radius_m / 111320
        lon_offset = radius_m / (111320 * abs(np.cos(np.radians(latitude))))
        
        south = latitude - lat_offset
        west = longitude - lon_offset
        north = latitude + lat_offset
        east = longitude + lon_offset
        
        # Overpass QL query
        overpass_query = f"""
        [out:json][timeout:25];
        (
          way["building"]({south},{west},{north},{east});
          relation["building"]({south},{west},{north},{east});
        );
        out geom;
        """
        
        try:
            response = requests.post(
                "http://overpass-api.de/api/interpreter",
                data=overpass_query,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                elements = data.get('elements', [])
                
                buildings_list = []
                for i, element in enumerate(elements, 1):
                    if element['type'] == 'way' and 'tags' in element and 'geometry' in element:
                        coords = [(node['lon'], node['lat']) for node in element['geometry']]
                        if coords:
                            center_lon = sum(coord[0] for coord in coords) / len(coords)
                            center_lat = sum(coord[1] for coord in coords) / len(coords)
                            
                            building_info = {
                                'building_id': i,
                                'building_lat': center_lat,
                                'building_lon': center_lon,
                                'building': element['tags'].get('building', ''),
                                'name': element['tags'].get('name', ''),
                                'amenity': element['tags'].get('amenity', ''),
                                'landuse': element['tags'].get('landuse', ''),
                                'osm_id': element['id']
                            }
                            buildings_list.append(building_info)
                
                if buildings_list:
                    return pd.DataFrame(buildings_list)
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.warning(f"Overpass API error: {e}")
            return pd.DataFrame()

    @staticmethod
    def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in meters using Haversine formula"""
        R = 6371000  # Earth's radius in meters
        
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c

    @staticmethod
    def classify_building(row) -> str:
        """Classify building into category"""
        building_tag = str(row.get('building', '')).lower()
        amenity_tag = str(row.get('amenity', '')).lower()
        
        # Residential
        if any(t in building_tag for t in ['house', 'residential', 'detached', 'cabin', 'hut']):
            return 'Residential'
        
        # Community/Public
        if any(t in building_tag for t in ['school', 'hospital', 'church', 'community']) or \
           amenity_tag in ['school', 'hospital', 'place_of_worship', 'community_centre']:
            return 'Community'
        
        # Commercial
        if any(t in building_tag for t in ['commercial', 'retail', 'shop', 'office']) or \
           amenity_tag in ['shop', 'restaurant', 'cafe']:
            return 'Commercial'
        
        return 'Unknown'

    def create_community_output_dir(self, community_name: str) -> Path:
        """Create output directory for specific community"""
        safe_name = self.create_safe_filename(community_name)
        community_dir = self.output_dir / safe_name
        community_dir.mkdir(exist_ok=True)
        return community_dir

    def save_csv_output(self, buildings_df: pd.DataFrame, community_dir: Path, community_name: str) -> Path:
        """Save buildings data as CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{self.create_safe_filename(community_name)}_buildings_{timestamp}.csv"
        csv_path = community_dir / csv_filename
        
        # Add metadata header
        with open(csv_path, 'w') as f:
            f.write(f"# Building extraction for {community_name}\n")
            f.write(f"# Extraction date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Search radius: {self.default_radius}m\n")
            f.write(f"# Total buildings found: {len(buildings_df)}\n")
            f.write(f"# Geocoding enabled: {self.enable_geocoding}\n")
            f.write("# \n")
        
        buildings_df.to_csv(csv_path, mode='a', index=False)
        logger.info(f"💾 CSV saved: {csv_path}")
        return csv_path

    def save_kml_output(self, buildings_df: pd.DataFrame, community_dir: Path, 
                       community_name: str, community_lat: float, community_lon: float) -> Path:
        """Save buildings data as KML with search radius and proper building names"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        kml_filename = f"{self.create_safe_filename(community_name)}_buildings_{timestamp}.kml"
        kml_path = community_dir / kml_filename
        
        # Create KML structure
        kml = ET.Element('kml', xmlns='http://www.opengis.net/kml/2.2')
        document = ET.SubElement(kml, 'Document')
        
        # Add document info
        ET.SubElement(document, 'name').text = f"{community_name} Buildings"
        ET.SubElement(document, 'description').text = f"""
        Building footprints for {community_name}
        Extracted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Total buildings: {len(buildings_df)}
        Search radius: {self.default_radius}m
        Geocoding: {'Enabled' if self.enable_geocoding else 'Disabled'}
        """
        
        # Define styles for different building categories
        styles = {
            'Residential': {'color': 'ff0000ff', 'icon': 'http://maps.google.com/mapfiles/kml/pushpin/red-pushpin.png'},
            'Commercial': {'color': 'ff00ff00', 'icon': 'http://maps.google.com/mapfiles/kml/pushpin/grn-pushpin.png'},
            'Community': {'color': 'ffff0000', 'icon': 'http://maps.google.com/mapfiles/kml/pushpin/blue-pushpin.png'},
            'Unknown': {'color': 'ff888888', 'icon': 'http://maps.google.com/mapfiles/kml/pushpin/wht-pushpin.png'},
            'Center': {'color': 'ffff00ff', 'icon': 'http://maps.google.com/mapfiles/kml/shapes/target.png', 'scale': '1.2'}
        }
        
        # Add styles
        for category, style in styles.items():
            style_elem = ET.SubElement(document, 'Style', id=f'{category}Style')
            icon_style = ET.SubElement(style_elem, 'IconStyle')
            ET.SubElement(icon_style, 'color').text = style['color']
            if 'scale' in style:
                ET.SubElement(icon_style, 'scale').text = style['scale']
            icon = ET.SubElement(icon_style, 'Icon')
            ET.SubElement(icon, 'href').text = style['icon']
        
        # Add search radius circle style
        circle_style = ET.SubElement(document, 'Style', id='SearchRadiusStyle')
        line_style = ET.SubElement(circle_style, 'LineStyle')
        ET.SubElement(line_style, 'color').text = 'ff0000ff'  # Blue line
        ET.SubElement(line_style, 'width').text = '2'
        poly_style = ET.SubElement(circle_style, 'PolyStyle')
        ET.SubElement(poly_style, 'color').text = '330000ff'  # Semi-transparent blue fill
        ET.SubElement(poly_style, 'outline').text = '1'
        
        # Add community center placemark with distinctive style
        center_placemark = ET.SubElement(document, 'Placemark')
        ET.SubElement(center_placemark, 'name').text = f"🎯 {community_name} Center"
        ET.SubElement(center_placemark, 'description').text = f"""
        <![CDATA[
        <b>Community Center Point</b><br/>
        Search radius: {self.default_radius}m<br/>
        Coordinates: {community_lat:.6f}, {community_lon:.6f}<br/>
        Buildings found: {len(buildings_df)}
        ]]>
        """
        ET.SubElement(center_placemark, 'styleUrl').text = '#CenterStyle'
        center_point = ET.SubElement(center_placemark, 'Point')
        ET.SubElement(center_point, 'coordinates').text = f"{community_lon},{community_lat},0"
        
        # Add search radius circle
        radius_placemark = ET.SubElement(document, 'Placemark')
        ET.SubElement(radius_placemark, 'name').text = f"Search Radius ({self.default_radius}m)"
        ET.SubElement(radius_placemark, 'description').text = f"Search area boundary for {community_name}"
        ET.SubElement(radius_placemark, 'styleUrl').text = '#SearchRadiusStyle'
        
        # Create circle polygon (approximate circle using linear ring)
        polygon = ET.SubElement(radius_placemark, 'Polygon')
        ET.SubElement(polygon, 'extrude').text = '0'
        ET.SubElement(polygon, 'altitudeMode').text = 'clampToGround'
        outer_boundary = ET.SubElement(polygon, 'outerBoundaryIs')
        linear_ring = ET.SubElement(outer_boundary, 'LinearRing')
        
        # Generate circle coordinates (360 points for smooth circle)
        circle_coords = []
        for i in range(361):  # 361 to close the circle
            angle = i * np.pi / 180
            # Convert radius from meters to degrees (approximate)
            lat_offset = (self.default_radius / 111320) * np.cos(angle)
            lon_offset = (self.default_radius / (111320 * np.cos(np.radians(community_lat)))) * np.sin(angle)
            
            circle_lat = community_lat + lat_offset
            circle_lon = community_lon + lon_offset
            circle_coords.append(f"{circle_lon},{circle_lat},0")
        
        ET.SubElement(linear_ring, 'coordinates').text = ' '.join(circle_coords)
        
        # Add building placemarks
        for _, building in buildings_df.iterrows():
            placemark = ET.SubElement(document, 'Placemark')
            
            # Smart building name generation
            building_id = building.get('building_id', 'Unknown')
            original_name = building.get('name', '')
            
            # Check if name is missing, empty, or "nan"
            if pd.isna(original_name) or str(original_name).lower() in ['nan', 'none', '', 'null']:
                building_name = f"Building {building_id}"
            else:
                building_name = str(original_name)
            
            ET.SubElement(placemark, 'name').text = building_name
            
            # Description with building details
            address_info = ""
            if self.enable_geocoding and 'address' in building and building['address']:
                if not pd.isna(building['address']) and str(building['address']).lower() not in ['nan', 'none', '']:
                    address_info = f"Address: {building['address']}<br/>"
            
            description = f"""
            <![CDATA[
            <b>Building Details:</b><br/>
            ID: {building_id}<br/>
            Category: {building.get('building_category', 'Unknown')}<br/>
            Type: {building.get('building', 'Unknown')}<br/>
            {address_info}Distance from center: {building.get('distance_from_center_m', 0):.0f}m<br/>
            Coordinates: {building.get('building_lat', 0):.6f}, {building.get('building_lon', 0):.6f}
            ]]>
            """
            ET.SubElement(placemark, 'description').text = description
            
            # Apply style based on category
            category = building.get('building_category', 'Unknown')
            if category in styles:
                ET.SubElement(placemark, 'styleUrl').text = f'#{category}Style'
            
            # Add point geometry
            point = ET.SubElement(placemark, 'Point')
            ET.SubElement(point, 'coordinates').text = f"{building.get('building_lon', 0)},{building.get('building_lat', 0)},0"
        
        # Write KML file
        rough_string = ET.tostring(kml, 'unicode')
        reparsed = minidom.parseString(rough_string)
        
        with open(kml_path, 'w', encoding='utf-8') as f:
            f.write(reparsed.toprettyxml(indent="  "))
        
        logger.info(f"🗺️ KML saved: {kml_path}")
        return kml_path

    def create_summary_report(self, buildings_df: pd.DataFrame, community_dir: Path, community_name: str):
        """Create summary report for community"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = community_dir / f"{self.create_safe_filename(community_name)}_summary_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write(f"BUILDING EXTRACTION SUMMARY\n")
            f.write(f"{'=' * 50}\n")
            f.write(f"Community: {community_name}\n")
            f.write(f"Extraction date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Search radius: {self.default_radius}m\n")
            f.write(f"Geocoding enabled: {self.enable_geocoding}\n")
            f.write(f"Total buildings found: {len(buildings_df)}\n\n")
            
            if not buildings_df.empty:
                # Category breakdown
                if 'building_category' in buildings_df.columns:
                    f.write("Buildings by category:\n")
                    category_counts = buildings_df['building_category'].value_counts()
                    for category, count in category_counts.items():
                        f.write(f"  {category}: {count}\n")
                    f.write("\n")
                
                # Distance analysis
                if 'distance_from_center_m' in buildings_df.columns:
                    f.write("Distance analysis:\n")
                    f.write(f"  Average distance from center: {buildings_df['distance_from_center_m'].mean():.0f}m\n")
                    f.write(f"  Maximum distance from center: {buildings_df['distance_from_center_m'].max():.0f}m\n")
                    f.write(f"  Minimum distance from center: {buildings_df['distance_from_center_m'].min():.0f}m\n\n")
                
                # Address analysis (if geocoding enabled)
                if self.enable_geocoding and 'address' in buildings_df.columns:
                    addresses_found = buildings_df['address'].notna().sum()
                    f.write("Address lookup results:\n")
                    f.write(f"  Buildings with addresses: {addresses_found}/{len(buildings_df)}\n")
                    f.write(f"  Success rate: {addresses_found/len(buildings_df)*100:.1f}%\n")
        
        logger.info(f"📄 Summary saved: {report_path}")

    def process_all_communities(self) -> Dict:
        """Process all communities and generate outputs"""
        logger.info(f"🚀 Starting batch processing of {len(self.communities_df)} communities")
        logger.info(f"💾 Cache directory: {self.cache_dir.absolute()}")
        logger.info(f"🔄 Force refresh: {self.force_refresh}")
        
        self.stats['total_communities'] = len(self.communities_df)
        results = []
        
        for idx, community_row in self.communities_df.iterrows():
            community_name = community_row['Community Name']
            
            try:
                # Create output directory for this community
                community_dir = self.create_community_output_dir(community_name)
                
                # Extract buildings
                buildings_df = self.extract_buildings_for_community(community_row)
                
                if buildings_df is not None and not buildings_df.empty:
                    # Save outputs
                    csv_path = self.save_csv_output(buildings_df, community_dir, community_name)
                    kml_path = self.save_kml_output(
                        buildings_df, community_dir, community_name,
                        community_row['Latitude'], community_row['Longitude']
                    )
                    self.create_summary_report(buildings_df, community_dir, community_name)
                    
                    # Update statistics
                    self.stats['successful_extractions'] += 1
                    self.stats['total_buildings_found'] += len(buildings_df)
                    self.stats['communities_with_buildings'] += 1
                    
                    result = {
                        'community': community_name,
                        'status': 'success',
                        'buildings_found': len(buildings_df),
                        'csv_path': str(csv_path),
                        'kml_path': str(kml_path)
                    }
                    
                    logger.info(f"✅ {community_name}: {len(buildings_df)} buildings found")
                    
                else:
                    # No buildings found
                    self.stats['communities_without_buildings'] += 1
                    result = {
                        'community': community_name,
                        'status': 'no_buildings',
                        'buildings_found': 0,
                        'csv_path': None,
                        'kml_path': None
                    }
                    
                    logger.info(f"⚠️ {community_name}: No buildings found")
                
                results.append(result)
                
            except Exception as e:
                self.stats['failed_extractions'] += 1
                logger.error(f"❌ {community_name}: Extraction failed - {e}")
                
                result = {
                    'community': community_name,
                    'status': 'failed',
                    'buildings_found': 0,
                    'error': str(e),
                    'csv_path': None,
                    'kml_path': None
                }
                results.append(result)
        
        # Generate final summary
        self.generate_final_summary(results)
        
        return {
            'results': results,
            'statistics': self.stats
        }

    def generate_final_summary(self, results: List[Dict]):
        """Generate final summary report"""
        summary_path = self.output_dir / f"batch_extraction_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(summary_path, 'w') as f:
            f.write("BATCH BUILDING EXTRACTION SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(f"Extraction date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input file: {self.csv_file}\n")
            f.write(f"Search radius: {self.default_radius}m\n")
            f.write(f"Geocoding enabled: {self.enable_geocoding}\n")
            f.write(f"Cache directory: {self.cache_dir}\n")
            f.write(f"Force refresh: {self.force_refresh}\n\n")
            
            f.write("STATISTICS:\n")
            f.write("-" * 20 + "\n")
            for key, value in self.stats.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            
            # Cache efficiency
            total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
            if total_requests > 0:
                cache_efficiency = self.stats['cache_hits'] / total_requests * 100
                f.write(f"Cache efficiency: {cache_efficiency:.1f}%\n")
            
            f.write(f"\nDETAILED RESULTS:\n")
            f.write("-" * 20 + "\n")
            for result in results:
                status_emoji = {"success": "✅", "no_buildings": "⚠️", "failed": "❌"}
                emoji = status_emoji.get(result['status'], "❓")
                f.write(f"{emoji} {result['community']}: {result['buildings_found']} buildings ({result['status']})\n")
        
        logger.info(f"📊 Final summary saved: {summary_path}")

def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description='Enhanced Community Building Extractor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python community_building_extractor.py                                    # Use default TownData.csv
  python community_building_extractor.py --csv MyTowns.csv                  # Use custom CSV
  python community_building_extractor.py --radius 2000 --no-geocoding      # 2km radius, no addresses
  python community_building_extractor.py --force-refresh                    # Ignore cache, fresh data
  python community_building_extractor.py --csv QuickTest.csv --radius 1000  # Quick test with small dataset
        """
    )
    
    parser.add_argument('--csv', '-c', default='TownData.csv',
                        help='CSV file with community data (default: TownData.csv)')
    parser.add_argument('--output', '-o', default='output',
                        help='Output directory (default: output)')
    parser.add_argument('--cache', default='cache',
                        help='Cache directory (default: cache)')
    parser.add_argument('--radius', '-r', type=int, default=1500,
                        help='Search radius in meters (default: 1500)')
    parser.add_argument('--delay', '-d', type=float, default=2.0,
                        help='Delay between API requests in seconds (default: 2.0)')
    parser.add_argument('--no-geocoding', action='store_true',
                        help='Disable address lookup (faster processing)')
    parser.add_argument('--cache-expiry', type=int, default=30,
                        help='Cache expiry in days (default: 30)')
    parser.add_argument('--force-refresh', action='store_true',
                        help='Force refresh of cached data')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: smaller radius, no geocoding, faster processing')
    
    return parser

def main():
    """Main function with command line interface"""
    parser = create_parser()
    args = parser.parse_args()
    
    print("🏘️ ENHANCED COMMUNITY BUILDING EXTRACTOR")
    print("=" * 60)
    print("✨ New Features:")
    print("   🔧 Custom CSV input support")
    print("   💾 Smart caching (saves re-downloading)")
    print("   🏠 Address lookup with reverse geocoding")
    print("   ⚡ Faster processing with cached data")
    print("=" * 60)
    
    # Quick mode adjustments
    if args.quick:
        args.radius = min(args.radius, 1000)  # Max 1km in quick mode
        args.no_geocoding = True
        args.delay = 1.0
        print("⚡ QUICK MODE: Smaller radius, no geocoding, faster processing")
    
    # Check if CSV file exists
    if not os.path.exists(args.csv):
        print(f"❌ Error: CSV file not found: {args.csv}")
        print(f"💡 Please ensure {args.csv} exists in the current directory")
        print(f"📋 Expected columns: Community Name, Latitude, Longitude")
        return
    
    try:
        # Initialize extractor with command line arguments
        extractor = CommunityBuildingExtractor(
            csv_file=args.csv,
            output_dir=args.output,
            cache_dir=args.cache,
            default_radius=args.radius,
            delay_between_requests=args.delay,
            enable_geocoding=not args.no_geocoding,
            cache_expiry_days=args.cache_expiry,
            force_refresh=args.force_refresh
        )
        
        # Process all communities
        results = extractor.process_all_communities()
        
        # Print final statistics
        print(f"\n🎉 BATCH EXTRACTION COMPLETE!")
        print("=" * 40)
        stats = results['statistics']
        print(f"📊 Total communities processed: {stats['total_communities']}")
        print(f"✅ Successful extractions: {stats['successful_extractions']}")
        print(f"❌ Failed extractions: {stats['failed_extractions']}")
        print(f"🏠 Total buildings found: {stats['total_buildings_found']}")
        print(f"🏘️ Communities with buildings: {stats['communities_with_buildings']}")
        print(f"🔍 Communities without buildings: {stats['communities_without_buildings']}")
        
        # Cache statistics
        total_requests = stats['cache_hits'] + stats['cache_misses']
        if total_requests > 0:
            cache_efficiency = stats['cache_hits'] / total_requests * 100
            print(f"💾 Cache efficiency: {cache_efficiency:.1f}% ({stats['cache_hits']}/{total_requests} hits)")
        
        # Geocoding statistics
        if not args.no_geocoding:
            total_geocoding = stats['geocoding_success'] + stats['geocoding_failed']
            if total_geocoding > 0:
                geocoding_rate = stats['geocoding_success'] / total_geocoding * 100
                print(f"🏠 Address lookup success: {geocoding_rate:.1f}% ({stats['geocoding_success']}/{total_geocoding})")
        
        print(f"\n📁 Results saved in: {extractor.output_dir.absolute()}")
        print(f"💾 Cache saved in: {extractor.cache_dir.absolute()}")
        print("📋 Each community has its own folder with CSV, KML, and summary files")
        
        if stats['cache_hits'] > 0:
            print(f"\n💡 Next run will be faster thanks to cached data!")
        
    except Exception as e:
        logger.error(f"❌ Batch extraction failed: {e}")
        raise

if __name__ == "__main__":
    main()