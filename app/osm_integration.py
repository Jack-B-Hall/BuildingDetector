#!/usr/bin/env python3
"""
OpenStreetMap Integration Module
Handles OSM metadata collection and geocoding with caching
"""

import time
import requests
from typing import Dict, Any, Optional

from .utils import vprint, calculate_distance, classify_building

# OSM Libraries for metadata enrichment
try:
    import osmnx as ox
    import overpy
    OSM_AVAILABLE = True
except ImportError:
    OSM_AVAILABLE = False

def get_osm_metadata_cached(lat: float, lon: float, cache_manager=None, radius_m: float = 50) -> Dict[str, Any]:
    """Get OSM metadata for a building location (with caching)"""
    
    # Check cache first
    if cache_manager:
        cached_metadata = cache_manager.get_osm_metadata(lat, lon)
        if cached_metadata is not None:
            return cached_metadata
    
    # Get fresh metadata
    metadata = get_osm_metadata(lat, lon, radius_m)
    
    # Cache the result
    if cache_manager:
        cache_manager.save_osm_metadata(lat, lon, metadata)
    
    return metadata

def get_osm_metadata(lat: float, lon: float, radius_m: float = 50) -> Dict[str, Any]:
    """Get OSM metadata for a building location using multiple methods"""
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
                
                # Extract metadata with safe string conversion
                metadata['name'] = _safe_str(nearest.get('name', ''))
                metadata['amenity'] = _safe_str(nearest.get('amenity', ''))
                metadata['building_type'] = _safe_str(nearest.get('building', ''))
                metadata['addr_street'] = _safe_str(nearest.get('addr:street', ''))
                metadata['addr_housenumber'] = _safe_str(nearest.get('addr:housenumber', ''))
                metadata['height'] = _safe_str(nearest.get('height', ''))
                metadata['building_use'] = _safe_str(nearest.get('building:use', ''))
                metadata['landuse'] = _safe_str(nearest.get('landuse', ''))
                metadata['shop'] = _safe_str(nearest.get('shop', ''))
                metadata['office'] = _safe_str(nearest.get('office', ''))
                
        except Exception as e:
            vprint(f"OSMnx query failed: {e}", "DEBUG")
        
        time.sleep(0.05)  # Reduced delay for cached operations
        
    except Exception as e:
        vprint(f"OSM metadata extraction failed for {lat}, {lon}: {e}", "DEBUG")
    
    return metadata

def _safe_str(value) -> str:
    """Safely convert value to string, handling NaN and None"""
    if value is None or str(value).lower() in ['nan', 'none', '']:
        return ''
    return str(value)

def reverse_geocode_cached(lat: float, lon: float, cache_manager=None) -> str:
    """Get address using Nominatim reverse geocoding (with caching)"""
    
    # Check cache first
    if cache_manager:
        cached_address = cache_manager.get_geocoding(lat, lon)
        if cached_address is not None:
            return cached_address
    
    # Get fresh address
    address = reverse_geocode(lat, lon)
    
    # Cache the result
    if cache_manager:
        cache_manager.save_geocoding(lat, lon, address)
    
    return address

def reverse_geocode(lat: float, lon: float) -> str:
    """Get address using Nominatim reverse geocoding"""
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
        
        return address
        
    except Exception as e:
        vprint(f"Reverse geocoding failed for {lat}, {lon}: {e}", "DEBUG")
        return ''