#!/usr/bin/env python3
"""
Enhanced Building Coordinate Extractor with OSM Metadata and Caching
Process multiple towns from CSV with OpenStreetMap enrichment and intelligent caching

Usage:
  # Batch mode (reads TownData.csv by default)
  python batch_building_extractor.py --batch
  python batch_building_extractor.py --batch --input towns.csv --default-distance 5
  
  # Single town mode
  python batch_building_extractor.py --lat -16.95 --lon 122.86 --distance 2 --name "Beagle Bay"
  
  # Cache management
  python batch_building_extractor.py --clear-cache
  python batch_building_extractor.py --batch --no-cache

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
import pickle
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import sys
import re
import logging

# OSM Libraries for metadata enrichment
try:
    import osmnx as ox
    import overpy
    OSM_AVAILABLE = True
except ImportError:
    OSM_AVAILABLE = False
    print("⚠️  OSM libraries not available. Install with: pip install osmnx overpy")

# Global settings
VERBOSE = True  # Default verbose on
DEFAULT_DISTANCE = 2.0  # Default 2km radius
CACHE_DIR = "cache"  # Cache directory
CACHE_EXPIRY_DAYS = 30  # Cache expires after 30 days

class CacheManager:
    """Manages caching for OSM metadata, geocoding, and building data"""
    
    def __init__(self, cache_dir: str = CACHE_DIR, expiry_days: int = CACHE_EXPIRY_DAYS):
        self.cache_dir = Path(cache_dir)
        self.expiry_days = expiry_days
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache files
        self.osm_metadata_cache_file = self.cache_dir / "osm_metadata.json"
        self.geocoding_cache_file = self.cache_dir / "geocoding.json"
        self.building_cache_file = self.cache_dir / "buildings.pickle"
        
        # Load existing caches
        self.osm_metadata_cache = self._load_json_cache(self.osm_metadata_cache_file)
        self.geocoding_cache = self._load_json_cache(self.geocoding_cache_file)
        self.building_cache = self._load_pickle_cache(self.building_cache_file)
        
        # Statistics
        self.stats = {
            'osm_cache_hits': 0,
            'osm_cache_misses': 0,
            'geocoding_cache_hits': 0,
            'geocoding_cache_misses': 0,
            'building_cache_hits': 0,
            'building_cache_misses': 0
        }
        
        vprint(f"📦 Cache initialized: {len(self.osm_metadata_cache)} OSM entries, "
               f"{len(self.geocoding_cache)} geocoding entries, "
               f"{len(self.building_cache)} building areas")
    
    def _load_json_cache(self, cache_file: Path) -> dict:
        """Load JSON cache file"""
        try:
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                    # Clean expired entries
                    return self._clean_expired_entries(cache)
            return {}
        except Exception as e:
            vprint(f"Warning: Could not load cache {cache_file}: {e}", "WARNING")
            return {}
    
    def _load_pickle_cache(self, cache_file: Path) -> dict:
        """Load pickle cache file"""
        try:
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    cache = pickle.load(f)
                    return self._clean_expired_entries(cache)
            return {}
        except Exception as e:
            vprint(f"Warning: Could not load cache {cache_file}: {e}", "WARNING")
            return {}
    
    def _clean_expired_entries(self, cache: dict) -> dict:
        """Remove expired cache entries"""
        cutoff_time = datetime.now().timestamp() - (self.expiry_days * 24 * 3600)
        cleaned = {}
        expired_count = 0
        
        for key, value in cache.items():
            if isinstance(value, dict) and 'timestamp' in value:
                if value['timestamp'] > cutoff_time:
                    cleaned[key] = value
                else:
                    expired_count += 1
            else:
                # Old format without timestamp, keep it but add timestamp
                cleaned[key] = {
                    'data': value,
                    'timestamp': datetime.now().timestamp()
                }
        
        if expired_count > 0:
            vprint(f"🧹 Cleaned {expired_count} expired cache entries")
        
        return cleaned
    
    def _save_json_cache(self, cache: dict, cache_file: Path):
        """Save JSON cache file"""
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, indent=2)
        except Exception as e:
            vprint(f"Warning: Could not save cache {cache_file}: {e}", "WARNING")
    
    def _save_pickle_cache(self, cache: dict, cache_file: Path):
        """Save pickle cache file"""
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache, f)
        except Exception as e:
            vprint(f"Warning: Could not save cache {cache_file}: {e}", "WARNING")
    
    def _create_cache_entry(self, data: Any) -> dict:
        """Create cache entry with timestamp"""
        return {
            'data': data,
            'timestamp': datetime.now().timestamp()
        }
    
    def get_osm_metadata(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """Get cached OSM metadata"""
        key = f"{lat:.6f}_{lon:.6f}"
        if key in self.osm_metadata_cache:
            self.stats['osm_cache_hits'] += 1
            return self.osm_metadata_cache[key]['data']
        
        self.stats['osm_cache_misses'] += 1
        return None
    
    def save_osm_metadata(self, lat: float, lon: float, metadata: Dict[str, Any]):
        """Cache OSM metadata"""
        key = f"{lat:.6f}_{lon:.6f}"
        self.osm_metadata_cache[key] = self._create_cache_entry(metadata)
    
    def get_geocoding(self, lat: float, lon: float) -> Optional[str]:
        """Get cached geocoding result"""
        key = f"{lat:.6f}_{lon:.6f}"
        if key in self.geocoding_cache:
            self.stats['geocoding_cache_hits'] += 1
            return self.geocoding_cache[key]['data']
        
        self.stats['geocoding_cache_misses'] += 1
        return None
    
    def save_geocoding(self, lat: float, lon: float, address: str):
        """Cache geocoding result"""
        key = f"{lat:.6f}_{lon:.6f}"
        self.geocoding_cache[key] = self._create_cache_entry(address)
    
    def get_building_area(self, lat: float, lon: float, radius: float) -> Optional[List[Dict[str, Any]]]:
        """Get cached building data for an area"""
        key = self._area_cache_key(lat, lon, radius)
        if key in self.building_cache:
            self.stats['building_cache_hits'] += 1
            return self.building_cache[key]['data']
        
        self.stats['building_cache_misses'] += 1
        return None
    
    def save_building_area(self, lat: float, lon: float, radius: float, buildings: List[Dict[str, Any]]):
        """Cache building data for an area"""
        key = self._area_cache_key(lat, lon, radius)
        self.building_cache[key] = self._create_cache_entry(buildings)
    
    def _area_cache_key(self, lat: float, lon: float, radius: float) -> str:
        """Generate cache key for area"""
        return f"{lat:.4f}_{lon:.4f}_{radius:.1f}"
    
    def save_all_caches(self):
        """Save all caches to disk"""
        self._save_json_cache(self.osm_metadata_cache, self.osm_metadata_cache_file)
        self._save_json_cache(self.geocoding_cache, self.geocoding_cache_file)
        self._save_pickle_cache(self.building_cache, self.building_cache_file)
        
        vprint(f"💾 Cache saved: {len(self.osm_metadata_cache)} OSM entries, "
               f"{len(self.geocoding_cache)} geocoding entries, "
               f"{len(self.building_cache)} building areas")
    
    def clear_cache(self):
        """Clear all cache files"""
        try:
            for cache_file in [self.osm_metadata_cache_file, self.geocoding_cache_file, self.building_cache_file]:
                if cache_file.exists():
                    cache_file.unlink()
            
            # Clear in-memory caches
            self.osm_metadata_cache = {}
            self.geocoding_cache = {}
            self.building_cache = {}
            
            vprint("🧹 Cache cleared successfully")
            
        except Exception as e:
            vprint(f"❌ Error clearing cache: {e}", "ERROR")
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        total_osm = self.stats['osm_cache_hits'] + self.stats['osm_cache_misses']
        total_geocoding = self.stats['geocoding_cache_hits'] + self.stats['geocoding_cache_misses']
        total_building = self.stats['building_cache_hits'] + self.stats['building_cache_misses']
        
        return {
            'osm_hit_rate': (self.stats['osm_cache_hits'] / total_osm * 100) if total_osm > 0 else 0,
            'geocoding_hit_rate': (self.stats['geocoding_cache_hits'] / total_geocoding * 100) if total_geocoding > 0 else 0,
            'building_hit_rate': (self.stats['building_cache_hits'] / total_building * 100) if total_building > 0 else 0,
            'total_entries': len(self.osm_metadata_cache) + len(self.geocoding_cache) + len(self.building_cache)
        }

# Global cache manager
cache_manager: Optional[CacheManager] = None

def vprint(message: str, level: str = "INFO"):
    """Verbose print function"""
    if VERBOSE or level in ["ERROR", "WARNING"]:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")

def sanitize_filename(name: str) -> str:
    """Convert community name to safe filename"""
    safe_name = re.sub(r'[^\w\s-]', '', name)
    safe_name = re.sub(r'[-\s]+', '_', safe_name)
    return safe_name.lower().strip('_')

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points on Earth (meters)"""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return 6371000.0 * c  # Return meters

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

def get_osm_metadata(lat: float, lon: float, radius_m: float = 50) -> Dict[str, Any]:
    """Get OSM metadata for a building location (with caching)"""
    global cache_manager
    
    # Check cache first
    if cache_manager:
        cached_metadata = cache_manager.get_osm_metadata(lat, lon)
        if cached_metadata is not None:
            return cached_metadata
    
    metadata = {
        'name': '',
        'amenity': '',
        'building_type': '',
        'addr_street': '',
        'addr_housenumber': '',
        'height': '',
        'building_use': '',
        'landuse': '',
        'shop': '',
        'office': ''
    }
    
    if not OSM_AVAILABLE:
        if cache_manager:
            cache_manager.save_osm_metadata(lat, lon, metadata)
        return metadata
    
    try:
        # Method 1: OSMnx query for nearest features
        try:
            # Query for building features near the point
            tags = {
                'building': True,
                'amenity': True,
                'name': True,
                'shop': True,
                'office': True
            }
            
            gdf = ox.features_from_point((lat, lon), tags=tags, dist=radius_m)
            
            if not gdf.empty:
                # Find the nearest feature
                gdf['distance'] = gdf.geometry.apply(
                    lambda geom: calculate_distance(lat, lon, geom.centroid.y, geom.centroid.x)
                )
                nearest = gdf.loc[gdf['distance'].idxmin()]
                
                # Extract metadata
                metadata['name'] = str(nearest.get('name', '')).replace('nan', '')
                metadata['amenity'] = str(nearest.get('amenity', '')).replace('nan', '')
                metadata['building_type'] = str(nearest.get('building', '')).replace('nan', '')
                metadata['addr_street'] = str(nearest.get('addr:street', '')).replace('nan', '')
                metadata['addr_housenumber'] = str(nearest.get('addr:housenumber', '')).replace('nan', '')
                metadata['height'] = str(nearest.get('height', '')).replace('nan', '')
                metadata['building_use'] = str(nearest.get('building:use', '')).replace('nan', '')
                metadata['landuse'] = str(nearest.get('landuse', '')).replace('nan', '')
                metadata['shop'] = str(nearest.get('shop', '')).replace('nan', '')
                metadata['office'] = str(nearest.get('office', '')).replace('nan', '')
                
        except Exception as e:
            vprint(f"OSMnx query failed: {e}", "DEBUG")
        
        time.sleep(0.05)  # Reduced delay for cached operations
        
    except Exception as e:
        vprint(f"OSM metadata extraction failed for {lat}, {lon}: {e}", "DEBUG")
    
    # Cache the result
    if cache_manager:
        cache_manager.save_osm_metadata(lat, lon, metadata)
    
    return metadata

def reverse_geocode(lat: float, lon: float) -> str:
    """Get address using Nominatim reverse geocoding (with caching)"""
    global cache_manager
    
    # Check cache first
    if cache_manager:
        cached_address = cache_manager.get_geocoding(lat, lon)
        if cached_address is not None:
            return cached_address
    
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            'lat': lat,
            'lon': lon,
            'format': 'json',
            'addressdetails': 1,
            'zoom': 18
        }
        
        headers = {
            'User-Agent': 'CommunityBuildingExtractor/2.0 (community-mapping@example.com)'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        address = ''
        if response.status_code == 200:
            data = response.json()
            
            if 'address' in data:
                addr_parts = []
                address_data = data['address']
                
                if 'house_number' in address_data:
                    addr_parts.append(address_data['house_number'])
                if 'road' in address_data:
                    addr_parts.append(address_data['road'])
                
                for key in ['suburb', 'city', 'town', 'village']:
                    if key in address_data:
                        addr_parts.append(address_data[key])
                        break
                
                if 'state' in address_data:
                    addr_parts.append(address_data['state'])
                if 'postcode' in address_data:
                    addr_parts.append(address_data['postcode'])
                
                address = ', '.join(addr_parts) if addr_parts else ''
            
            if not address:
                address = data.get('display_name', '')
        
        # Cache the result
        if cache_manager:
            cache_manager.save_geocoding(lat, lon, address)
        
        return address
        
    except Exception as e:
        vprint(f"Reverse geocoding failed for {lat}, {lon}: {e}", "DEBUG")
        if cache_manager:
            cache_manager.save_geocoding(lat, lon, '')
        return ''

def classify_building(metadata: Dict[str, Any]) -> str:
    """Classify building based on OSM metadata"""
    
    # Community/Public building indicators
    community_amenities = [
        'school', 'hospital', 'clinic', 'place_of_worship', 'church', 
        'community_centre', 'library', 'townhall', 'police', 'fire_station',
        'post_office', 'courthouse', 'government', 'public_building',
        'kindergarten', 'college', 'university'
    ]
    
    community_buildings = [
        'school', 'hospital', 'church', 'community', 'government', 'public',
        'kindergarten', 'college', 'university', 'library', 'townhall',
        'civic', 'fire_station', 'police'
    ]
    
    # Commercial building indicators
    commercial_amenities = [
        'shop', 'restaurant', 'cafe', 'fast_food', 'bar', 'pub',
        'bank', 'pharmacy', 'fuel', 'marketplace', 'retail'
    ]
    
    commercial_buildings = [
        'commercial', 'retail', 'shop', 'office', 'warehouse', 'industrial',
        'supermarket', 'store', 'kiosk', 'service', 'factory'
    ]
    
    # Residential building indicators
    residential_buildings = [
        'house', 'residential', 'detached', 'cabin', 'hut', 'apartments',
        'terrace', 'dormitory', 'bungalow', 'semidetached_house'
    ]
    
    # Check amenity first (most reliable)
    amenity = metadata.get('amenity', '').lower()
    if amenity in community_amenities:
        return 'Community'
    if amenity in commercial_amenities:
        return 'Commercial'
    
    # Check building type
    building_type = metadata.get('building_type', '').lower()
    if building_type in community_buildings:
        return 'Community'
    if building_type in commercial_buildings:
        return 'Commercial'
    if building_type in residential_buildings:
        return 'Residential'
    
    # Check shop/office tags
    if metadata.get('shop', '') or metadata.get('office', ''):
        return 'Commercial'
    
    # Check name for hints
    name = metadata.get('name', '').lower()
    if name:
        community_keywords = ['school', 'church', 'hospital', 'clinic', 'council',
                            'community', 'centre', 'center', 'library', 'government']
        if any(keyword in name for keyword in community_keywords):
            return 'Community'
        
        commercial_keywords = ['shop', 'store', 'market', 'office', 'business',
                             'company', 'warehouse', 'factory']
        if any(keyword in name for keyword in commercial_keywords):
            return 'Commercial'
    
    # Default to Residential
    return 'Residential'

def process_building_file(filepath: str, center_lat: float, center_lon: float, max_distance_km: float) -> List[Dict[str, Any]]:
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
                            osm_metadata = get_osm_metadata(center_building_lat, center_building_lon)
                            
                            # Get address (cached)
                            address = reverse_geocode(center_building_lat, center_building_lon)
                            
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

def save_buildings_to_csv(buildings: List[Dict[str, Any]], output_file: str) -> Dict[str, Any]:
    """Save buildings to CSV and return summary statistics"""
    if not buildings:
        return {'count': 0, 'types': {}, 'distance_range': (0, 0), 'height_stats': None}
    
    df = pd.DataFrame(buildings)
    df = df.sort_values('distance_km')
    df['building_id'] = range(1, len(df) + 1)
    
    # Reorder columns for better readability
    column_order = [
        'building_id', 'latitude', 'longitude', 'name', 'building_category',
        'amenity', 'building_type', 'addr_street', 'addr_housenumber', 'address',
        'height', 'building_use', 'landuse', 'shop', 'office', 'distance_km',
        'distance_m', 'area_sqm', 'confidence', 'quadkey', 'polygon_points', 'description'
    ]
    
    # Only include columns that exist
    final_columns = [col for col in column_order if col in df.columns]
    output_df = df[final_columns].copy()
    
    # Round coordinates
    output_df['latitude'] = output_df['latitude'].round(6)
    output_df['longitude'] = output_df['longitude'].round(6)
    
    output_df.to_csv(output_file, index=False)
    
    # Generate summary statistics
    type_counts = df['building_category'].value_counts().to_dict()
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
    """Save buildings to KML format with color-coded categories"""
    if not buildings:
        return
    
    buildings.sort(key=lambda x: x['distance_km'])
    
    kml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
    <name>{community_name} - Building Footprints</name>
    <description>Building locations for {community_name} with OSM metadata and classification</description>
    
    <!-- Style definitions -->
    <Style id="Residential">
        <IconStyle>
            <color>ff00ff00</color>
            <scale>1.0</scale>
            <Icon><href>http://maps.google.com/mapfiles/kml/shapes/homegardenbusiness.png</href></Icon>
        </IconStyle>
    </Style>
    <Style id="Community">
        <IconStyle>
            <color>ffff0000</color>
            <scale>1.4</scale>
            <Icon><href>http://maps.google.com/mapfiles/kml/shapes/schools.png</href></Icon>
        </IconStyle>
    </Style>
    <Style id="Commercial">
        <IconStyle>
            <color>ff00ffff</color>
            <scale>1.2</scale>
            <Icon><href>http://maps.google.com/mapfiles/kml/shapes/shopping.png</href></Icon>
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
    
    <!-- Buildings by category -->'''
    
    # Group buildings by category
    categories = {}
    for building in buildings:
        category = building['building_category']
        if category not in categories:
            categories[category] = []
        categories[category].append(building)
    
    for category, cat_buildings in categories.items():
        kml_content += f'''
    <Folder>
        <name>{category} Buildings ({len(cat_buildings)})</name>
        <description>{category} buildings in {community_name}</description>'''
        
        for i, building in enumerate(cat_buildings, 1):
            building_name = building.get('name', f'{community_name} Building {i}')
            if not building_name:
                building_name = f'{community_name} Building {i}'
            
            kml_content += f'''
        <Placemark>
            <name>{building_name}</name>
            <description><![CDATA[
                <b>{building_name}</b><br/>
                <b>Category:</b> {building['building_category']}<br/>
                <b>Description:</b> {building['description']}<br/>
                <br/>
                <b>Details:</b><br/>
                Coordinates: {building['latitude']:.6f}, {building['longitude']:.6f}<br/>
                Distance: {building['distance_km']:.3f} km<br/>
                Area: {building['area_sqm']:.1f} sqm<br/>
                Height: {building['height']}m<br/>
                Confidence: {building['confidence']}<br/>
                QuadKey: {building['quadkey']}
            ]]></description>
            <styleUrl>#{building['building_category']}</styleUrl>
            <Point>
                <coordinates>{building['longitude']},{building['latitude']},0</coordinates>
            </Point>
        </Placemark>'''
        
        kml_content += '''
    </Folder>'''
    
    kml_content += '''
</Document>
</kml>'''
    
    with open(kml_file, 'w', encoding='utf-8') as f:
        f.write(kml_content)

def process_single_town(community_name: str, lat: float, lon: float, distance: float, 
                       australia_links: pd.DataFrame, output_dir: str, zoom: int = 9) -> Dict[str, Any]:
    """Process a single community and return results"""
    global cache_manager
    
    vprint(f"🏘️  Processing {community_name}...")
    vprint(f"    📍 Location: ({lat:.6f}, {lon:.6f})")
    vprint(f"    📏 Search radius: {distance}km")
    
    # Check if we have cached results for this area
    if cache_manager:
        cached_buildings = cache_manager.get_building_area(lat, lon, distance)
        if cached_buildings is not None:
            vprint(f"    📦 Using cached building data ({len(cached_buildings)} buildings)")
            
            # Create community-specific output directory
            safe_name = sanitize_filename(community_name)
            community_output_dir = os.path.join(output_dir, safe_name)
            Path(community_output_dir).mkdir(exist_ok=True)
            
            # Save files directly from cache
            csv_file = os.path.join(community_output_dir, f"{safe_name}_buildings.csv")
            kml_file = os.path.join(community_output_dir, f"{safe_name}_buildings.kml")
            summary_file = os.path.join(community_output_dir, f"{safe_name}_summary.txt")
            
            stats = save_buildings_to_csv(cached_buildings, csv_file)
            save_buildings_to_kml(cached_buildings, kml_file, lat, lon, distance, community_name)
            generate_community_summary(cached_buildings, stats, summary_file, community_name, lat, lon, distance)
            
            vprint(f"    ✅ {community_name} completed from cache: {len(cached_buildings)} buildings")
            
            return {
                'community_name': community_name,
                'status': 'success',
                'buildings': cached_buildings,
                'stats': stats,
                'output_dir': community_output_dir,
                'files': {
                    'csv': csv_file,
                    'kml': kml_file,
                    'summary': summary_file
                },
                'from_cache': True
            }
    
    # Create community-specific output directory
    safe_name = sanitize_filename(community_name)
    community_output_dir = os.path.join(output_dir, safe_name)
    Path(community_output_dir).mkdir(exist_ok=True)
    
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
        
        # Step 4: Process buildings with OSM metadata
        vprint(f"    🏠 Processing buildings with OSM metadata...")
        all_buildings = []
        for filepath in downloaded_files:
            buildings = process_building_file(filepath, lat, lon, distance)
            all_buildings.extend(buildings)
        
        vprint(f"    🏠 Found {len(all_buildings)} buildings with metadata")
        
        # Cache the results for this area
        if cache_manager and all_buildings:
            cache_manager.save_building_area(lat, lon, distance, all_buildings)
        
        # Step 5: Save files in community directory
        csv_file = os.path.join(community_output_dir, f"{safe_name}_buildings.csv")
        kml_file = os.path.join(community_output_dir, f"{safe_name}_buildings.kml")
        summary_file = os.path.join(community_output_dir, f"{safe_name}_summary.txt")
        
        stats = save_buildings_to_csv(all_buildings, csv_file)
        save_buildings_to_kml(all_buildings, kml_file, lat, lon, distance, community_name)
        generate_community_summary(all_buildings, stats, summary_file, community_name, lat, lon, distance)
        
        vprint(f"    ✅ {community_name} completed: {len(all_buildings)} buildings")
        vprint(f"    📁 Files saved to: {community_output_dir}/")
        
        return {
            'community_name': community_name,
            'status': 'success',
            'buildings': all_buildings,
            'stats': stats,
            'output_dir': community_output_dir,
            'files': {
                'csv': csv_file,
                'kml': kml_file,
                'summary': summary_file
            },
            'from_cache': False
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

def generate_community_summary(buildings: List[Dict[str, Any]], stats: Dict[str, Any], 
                             output_file: str, community_name: str, lat: float, lon: float, radius: float):
    """Generate individual community summary"""
    
    summary = f"""
🏘️  {community_name.upper()} - BUILDING SUMMARY
{'=' * 60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📍 LOCATION DETAILS
{'=' * 30}
Community: {community_name}
Coordinates: {lat:.6f}, {lon:.6f}
Search radius: {radius}km
Total buildings found: {stats['count']}

📊 BUILDING CATEGORIES
{'=' * 30}"""
    
    if stats['count'] > 0:
        for category, count in stats['types'].items():
            percentage = (count / stats['count']) * 100
            summary += f"\n{category}: {count} ({percentage:.1f}%)"
        
        summary += f"""

📏 DISTANCE & SIZE ANALYSIS
{'=' * 30}
Distance range: {stats['distance_range'][0]:.3f} - {stats['distance_range'][1]:.3f} km"""
        
        if stats['height_stats']:
            hs = stats['height_stats']
            summary += f"""
Buildings with height data: {hs['count']}
Average height: {hs['average']:.1f}m
Height range: {hs['range'][0]:.1f} - {hs['range'][1]:.1f}m"""
        
        # Add detailed building list
        summary += f"""

📋 DETAILED BUILDING LIST
{'=' * 30}"""
        
        buildings_by_category = {}
        for building in buildings:
            category = building['building_category']
            if category not in buildings_by_category:
                buildings_by_category[category] = []
            buildings_by_category[category].append(building)
        
        for category in ['Community', 'Commercial', 'Residential']:
            if category in buildings_by_category:
                summary += f"\n\n{category.upper()} BUILDINGS ({len(buildings_by_category[category])}):"
                summary += "\n" + "-" * 40
                
                for i, building in enumerate(buildings_by_category[category][:10], 1):  # Show first 10
                    name = building.get('name', 'Unnamed Building')
                    if not name or name == 'Unnamed Building':
                        name = f"Building #{i}"
                    
                    summary += f"\n{i:2d}. {name}"
                    if building.get('amenity'):
                        summary += f" ({building['amenity']})"
                    summary += f" - {building['distance_km']:.2f}km"
                    if building.get('address'):
                        summary += f" - {building['address'][:50]}{'...' if len(building['address']) > 50 else ''}"
                
                if len(buildings_by_category[category]) > 10:
                    summary += f"\n    ... and {len(buildings_by_category[category]) - 10} more {category.lower()} buildings"
    else:
        summary += "\nNo buildings found in the specified radius."
    
    summary += f"""

🎯 RECOMMENDATIONS
{'=' * 30}
- Use the KML file in Google Earth for visual verification
- Import CSV file into Starlink coverage planning tools
- Review Community buildings for priority coverage
- Consider field verification for buildings without clear classification

📁 FILES GENERATED
{'=' * 30}
- CSV: {os.path.basename(output_file).replace('_summary.txt', '_buildings.csv')}
- KML: {os.path.basename(output_file).replace('_summary.txt', '_buildings.kml')}
- Summary: {os.path.basename(output_file)}

✅ Building extraction complete for {community_name}!
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(summary)

def generate_summary_report(results: List[Dict[str, Any]], output_file: str):
    """Generate overall batch summary report"""
    total_buildings = sum(len(r['buildings']) for r in results)
    successful_towns = [r for r in results if r['status'] == 'success']
    failed_towns = [r for r in results if r['status'] != 'success']
    cached_towns = [r for r in successful_towns if r.get('from_cache', False)]
    
    # Cache statistics
    cache_stats_text = ""
    if cache_manager:
        cache_stats = cache_manager.get_cache_stats()
        cache_stats_text = f"""

📦 CACHE PERFORMANCE
{'=' * 30}
OSM metadata hit rate: {cache_stats['osm_hit_rate']:.1f}%
Geocoding hit rate: {cache_stats['geocoding_hit_rate']:.1f}%
Building area hit rate: {cache_stats['building_hit_rate']:.1f}%
Total cached entries: {cache_stats['total_entries']}
Communities served from cache: {len(cached_towns)}"""
    
    report = f"""
🏗️  BATCH BUILDING EXTRACTION SUMMARY REPORT
{'=' * 60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 OVERALL STATISTICS
{'=' * 30}
Total communities processed: {len(results)}
Successful extractions: {len(successful_towns)}
Failed extractions: {len(failed_towns)}
Total buildings found: {total_buildings:,}
Communities served from cache: {len(cached_towns)}{cache_stats_text}

📋 COMMUNITY RESULTS
{'=' * 30}
"""
    
    for result in results:
        community = result['community_name']
        status = result['status']
        cache_indicator = " 📦" if result.get('from_cache', False) else ""
        
        if status == 'success':
            stats = result['stats']
            building_count = stats['count']
            types = stats['types']
            
            report += f"""
🏘️  {community}{cache_indicator}
   Status: ✅ SUCCESS
   Buildings found: {building_count:,}
   Categories: R:{types.get('Residential', 0)} C:{types.get('Community', 0)} M:{types.get('Commercial', 0)}
   Output: {result.get('output_dir', 'output/' + sanitize_filename(community))}"""
        else:
            report += f"""
🏘️  {community}
   Status: ❌ {status.upper()}"""
    
    # Add category summary
    if successful_towns:
        all_types = {}
        for result in successful_towns:
            for category, count in result['stats']['types'].items():
                all_types[category] = all_types.get(category, 0) + count
        
        report += f"""

📈 BUILDING CATEGORY SUMMARY
{'=' * 30}
Total buildings by category across all communities:"""
        
        for category, count in sorted(all_types.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_buildings) * 100
            report += f"\n   {category}: {count:,} ({percentage:.1f}%)"
    
    report += f"""

🔧 NEXT STEPS
{'=' * 30}
1. Review individual community summaries in each folder
2. Import CSV files into coverage planning tools  
3. Validate results using KML files in Google Earth
4. Cross-reference with field surveys during site visits

📁 OUTPUT STRUCTURE
{'=' * 30}
output/
├── <community_name>/
│   ├── <community_name>_buildings.csv
│   ├── <community_name>_buildings.kml  
│   └── <community_name>_summary.txt
└── batch_summary_report.txt

💡 CACHE TIPS
{'=' * 30}
- Cache significantly speeds up re-runs
- Clear cache with --clear-cache if data seems outdated
- Cache expires automatically after {CACHE_EXPIRY_DAYS} days
- Disable cache with --no-cache for fresh data

✅ Enhanced building extraction with OSM metadata and caching complete!
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
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
        
        vprint(f"Available columns in CSV: {list(df.columns)}")
        
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
            vprint(f"No distance column found, using default distance: {default_distance}km")
            df['distance_km'] = default_distance
            required_mapping['distance_km'] = 'distance_km'
        
        # Create standardized dataframe
        standardized_df = pd.DataFrame()
        for standard_name, original_name in required_mapping.items():
            standardized_df[standard_name] = df[original_name]
        
        # Add additional columns if they exist
        additional_cols = ['State', 'LGA', 'ABS Remoteness', 'AGIL CODE']
        for col in additional_cols:
            if col in df.columns:
                standardized_df[col] = df[col]
        
        vprint(f"Successfully loaded {len(standardized_df)} communities from {csv_file}")
        return standardized_df
        
    except Exception as e:
        raise ValueError(f"Error loading CSV file {csv_file}: {e}")

def main():
    global VERBOSE, cache_manager
    
    parser = argparse.ArgumentParser(
        description="Extract building coordinates with OSM metadata and intelligent caching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Batch mode (processes TownData.csv with 2km radius)
  python batch_building_extractor.py --batch
  
  # Custom input file with different default distance
  python batch_building_extractor.py --batch --input my_towns.csv --default-distance 5
  
  # Single community mode
  python batch_building_extractor.py --lat -12.324 --lon 133.056 --distance 2 --name "Gunbalanya"
  
  # Cache management
  python batch_building_extractor.py --clear-cache
  python batch_building_extractor.py --batch --no-cache
  
Expected CSV format:
  Community Name,Latitude,Longitude[,distance_km]
  AGIL CODE,Community Name,Latitude,Longitude,State,LGA,ABS Remoteness
        """
    )
    
    # Mode selection
    parser.add_argument('--batch', action='store_true',
                        help='Batch mode: process communities from CSV file')
    
    # Batch mode arguments
    parser.add_argument('--input', type=str, default='TownData.csv',
                        help='Input CSV file for batch mode (default: TownData.csv)')
    parser.add_argument('--default-distance', type=float, default=DEFAULT_DISTANCE,
                        help=f'Default search radius in km if not specified in CSV (default: {DEFAULT_DISTANCE})')
    
    # Single community mode arguments
    parser.add_argument('--lat', type=float,
                        help='Latitude of center point (single community mode)')
    parser.add_argument('--lon', type=float,
                        help='Longitude of center point (single community mode)')
    parser.add_argument('--distance', type=float,
                        help='Search radius in kilometers (single community mode)')
    parser.add_argument('--name', type=str,
                        help='Community name (single community mode)')
    
    # Cache arguments
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable caching (fetch fresh data)')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Clear all cached data and exit')
    parser.add_argument('--cache-dir', type=str, default=CACHE_DIR,
                        help=f'Cache directory (default: {CACHE_DIR})')
    
    # Common arguments
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory (default: output)')
    parser.add_argument('--zoom', type=int, default=9,
                        help='QuadKey zoom level for tile selection (default: 9)')
    parser.add_argument('--quiet', action='store_true',
                        help='Disable verbose logging')
    parser.add_argument('--keep-temp', action='store_true',
                        help='Keep temporary downloaded files (for debugging)')
    
    args = parser.parse_args()
    if args.quiet:
        VERBOSE = False
    
    # Handle cache clearing
    if args.clear_cache:
        temp_cache = CacheManager(args.cache_dir)
        temp_cache.clear_cache()
        print("✅ Cache cleared successfully!")
        sys.exit(0)
    
    # Initialize cache manager
    if not args.no_cache:
        cache_manager = CacheManager(args.cache_dir)
        vprint(f"📦 Cache enabled: {args.cache_dir}/")
    else:
        vprint("🚫 Cache disabled - fetching fresh data")
    
    # Validate arguments
    if not args.batch and not all([args.lat, args.lon, args.distance, args.name]):
        print("❌ Error: Either use --batch mode or provide --lat, --lon, --distance, and --name")
        sys.exit(1)
    
    # Create output directory
    output_dir = args.output_dir
    Path(output_dir).mkdir(exist_ok=True)
    
    print("🏗️  Enhanced Building Coordinate Extractor with OSM Metadata & Caching")
    print("=" * 75)
    if not OSM_AVAILABLE:
        print("⚠️  OSM libraries not available - metadata collection will be limited")
        print("   Install with: pip install osmnx overpy")
        print()
    
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
        
        # Save cache before generating reports
        if cache_manager:
            cache_manager.save_all_caches()
        
        # Generate batch summary report
        summary_file = os.path.join(output_dir, 'batch_summary_report.txt')
        generate_summary_report(results, summary_file)
        
        # Cleanup
        if not args.keep_temp:
            cleanup_temp_files("temp_buildings")
        
        print(f"\n🎉 Processing complete!")
        print(f"📁 Output directory: {output_dir}/")
        print(f"📊 Batch summary: {summary_file}")
        
        # Show cache statistics
        if cache_manager:
            cache_stats = cache_manager.get_cache_stats()
            print(f"📦 Cache performance: OSM {cache_stats['osm_hit_rate']:.1f}%, "
                  f"Geocoding {cache_stats['geocoding_hit_rate']:.1f}%, "
                  f"Areas {cache_stats['building_hit_rate']:.1f}%")
        
        # Show successful extractions
        successful = [r for r in results if r['status'] == 'success']
        if successful:
            print(f"\n✅ Successfully processed {len(successful)} communities:")
            for result in successful:
                cache_indicator = " 📦" if result.get('from_cache', False) else ""
                print(f"   📁 {result['community_name']}{cache_indicator}: {result['output_dir']}/")
        
    except KeyboardInterrupt:
        if cache_manager:
            cache_manager.save_all_caches()
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