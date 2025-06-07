#!/usr/bin/env python3
"""
Utility Functions Module
Helper functions for the building extractor
"""

import math
import re
import html
from datetime import datetime
from typing import Tuple, List

# Global settings
VERBOSE = True

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

def escape_xml_text(text: str) -> str:
    """Escape XML/HTML characters in text for safe KML output"""
    if not text or text == 'nan':
        return ''
    
    # Convert to string and escape HTML entities
    text_str = str(text)
    escaped = html.escape(text_str, quote=True)
    return escaped

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

def classify_building(metadata: dict) -> str:
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