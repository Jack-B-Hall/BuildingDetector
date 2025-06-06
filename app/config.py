#!/usr/bin/env python3
"""
Configuration Module
Central location for all configurable settings
"""

import os
from pathlib import Path

# API Settings
API_SETTINGS = {
    'overpass_url': 'http://overpass-api.de/api/interpreter',
    'nominatim_url': 'https://nominatim.openstreetmap.org/reverse',
    'user_agent': 'CommunityBuildingExtractor/2.0 (Contact: your-email@example.com)',
    'timeout': 30,
    'max_retries': 3
}

# Delay Settings (in seconds)
DELAY_SETTINGS = {
    'between_communities': 2.0,
    'between_geocoding': 0.5,
    'after_error': 5.0,
    'after_rate_limit': 60.0
}

# Search Settings
SEARCH_SETTINGS = {
    'default_radius': 1500,
    'max_radius': 5000,
    'min_buildings_threshold': 3,  # Minimum buildings before trying alternative methods
    'coordinate_precision': 6      # Decimal places for coordinates
}

# Cache Settings
CACHE_SETTINGS = {
    'building_cache_days': 30,
    'geocoding_cache_days': 7,
    'enable_cache': True,
    'cache_dir': 'cache'
}

# Output Settings
OUTPUT_SETTINGS = {
    'output_dir': 'output',
    'timestamp_format': '%Y%m%d_%H%M%S',
    'csv_encoding': 'utf-8',
    'kml_circle_points': 361  # Points to draw search radius circle
}

# Classification Rules
CLASSIFICATION_RULES = {
    'residential_tags': [
        'house', 'residential', 'detached', 'cabin', 'hut', 'apartments',
        'terrace', 'dormitory', 'bungalow', 'semidetached_house'
    ],
    'community_tags': [
        'school', 'hospital', 'church', 'community', 'government', 'public',
        'kindergarten', 'college', 'university', 'library', 'townhall',
        'civic', 'fire_station', 'police'
    ],
    'commercial_tags': [
        'commercial', 'retail', 'shop', 'office', 'warehouse', 'industrial',
        'supermarket', 'store', 'kiosk', 'service', 'factory'
    ],
    'default_category': 'Residential'  # For unknown building types
}

# Logging Settings
LOGGING_SETTINGS = {
    'log_file': 'building_extraction.log',
    'log_level': 'INFO',
    'log_format': '%(asctime)s - %(levelname)s - %(message)s'
}

# OSMnx Settings
OSMNX_SETTINGS = {
    'use_cache': True,
    'log_console': False,
    'cache_folder': './cache/osmnx'
}

# Place name variations for searching
PLACE_NAME_TEMPLATES = [
    "{name}, Northern Territory, Australia",
    "{name}, NT, Australia",
    "{name}, Australia",
    "{name}"
]

# File paths
def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent

def get_cache_dir():
    """Get the cache directory path"""
    return get_project_root() / CACHE_SETTINGS['cache_dir']

def get_output_dir():
    """Get the output directory path"""
    return get_project_root() / OUTPUT_SETTINGS['output_dir']

# Environment variable overrides
def load_env_overrides():
    """Load settings from environment variables if set"""
    if 'BUILDING_EXTRACTOR_RADIUS' in os.environ:
        SEARCH_SETTINGS['default_radius'] = int(os.environ['BUILDING_EXTRACTOR_RADIUS'])
    
    if 'BUILDING_EXTRACTOR_CACHE_DIR' in os.environ:
        CACHE_SETTINGS['cache_dir'] = os.environ['BUILDING_EXTRACTOR_CACHE_DIR']
    
    if 'BUILDING_EXTRACTOR_OUTPUT_DIR' in os.environ:
        OUTPUT_SETTINGS['output_dir'] = os.environ['BUILDING_EXTRACTOR_OUTPUT_DIR']

# Load environment overrides on import
load_env_overrides()