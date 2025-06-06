#!/usr/bin/env python3
"""
Data Extractor Module
Handles building extraction from OpenStreetMap sources
"""

import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import logging
import time
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)


class BuildingExtractor:
    """Extracts building data from OpenStreetMap using multiple methods"""
    
    def __init__(self, default_radius: int = 1500, delay_between_requests: float = 2.0):
        """
        Initialize building extractor
        
        Args:
            default_radius: Default search radius in meters
            delay_between_requests: Delay between API requests in seconds
        """
        self.default_radius = default_radius
        self.delay_between_requests = delay_between_requests
        
        # Configure osmnx
        ox.settings.use_cache = True
        ox.settings.log_console = False
    
    def extract_buildings(self, community_name: str, latitude: float, 
                         longitude: float, radius: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Extract buildings using multiple fallback methods
        
        Args:
            community_name: Name of the community
            latitude: Community center latitude
            longitude: Community center longitude
            radius: Search radius (uses default if None)
            
        Returns:
            DataFrame of buildings or None if extraction fails
        """
        radius = radius or self.default_radius
        buildings_df = None
        
        logger.info(f"🔍 Extracting buildings for {community_name} at {latitude}, {longitude}")
        
        # Method 1: OSMnx point-based query
        buildings_df = self._extract_via_osmnx_point(latitude, longitude, radius)
        
        # Method 2: Try place name query if point query fails or returns few results
        if buildings_df is None or len(buildings_df) < 3:
            logger.info(f"📍 Trying place name query for {community_name}")
            place_buildings = self._extract_via_osmnx_place(community_name)
            if place_buildings is not None and (buildings_df is None or len(place_buildings) > len(buildings_df)):
                buildings_df = place_buildings
        
        # Method 3: Direct Overpass API query
        if buildings_df is None or len(buildings_df) < 3:
            logger.info(f"🌐 Trying Overpass API for {community_name}")
            overpass_buildings = self._extract_via_overpass(latitude, longitude, radius)
            if overpass_buildings is not None and (buildings_df is None or len(overpass_buildings) > len(buildings_df)):
                buildings_df = overpass_buildings
        
        # Add delay to be respectful to APIs
        time.sleep(self.delay_between_requests)
        
        if buildings_df is not None and not buildings_df.empty:
            logger.info(f"✅ Found {len(buildings_df)} buildings for {community_name}")
        else:
            logger.warning(f"⚠️ No buildings found for {community_name}")
        
        return buildings_df
    
    def _extract_via_osmnx_point(self, latitude: float, longitude: float, 
                                 radius: int) -> Optional[pd.DataFrame]:
        """Extract buildings using OSMnx point query"""
        try:
            center_point = (latitude, longitude)
            
            # First try with standard building tag
            buildings = ox.features_from_point(
                center_point,
                tags={'building': True},
                dist=radius
            )
            
            # Log what we found
            logger.info(f"📊 OSMnx point query returned {len(buildings)} features")
            
            if not buildings.empty:
                # Check what types of buildings we found
                if 'building' in buildings.columns:
                    building_types = buildings['building'].value_counts()
                    logger.info(f"📊 Building types found: {building_types.to_dict()}")
                
                return self._process_osmnx_buildings(buildings, latitude, longitude)
            
            # If no buildings found, try a broader query
            logger.info("🔄 No buildings found, trying broader feature search...")
            all_features = ox.features_from_point(
                center_point,
                tags={'building': ['yes', 'house', 'residential', 'commercial', 'industrial']},
                dist=radius
            )
            
            if not all_features.empty:
                logger.info(f"📊 Broader search found {len(all_features)} features")
                return self._process_osmnx_buildings(all_features, latitude, longitude)
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ OSMnx point query failed: {e}")
            return None
    
    def _extract_via_osmnx_place(self, place_name: str) -> Optional[pd.DataFrame]:
        """Extract buildings using OSMnx place name query"""
        try:
            # Try variations of the place name
            place_queries = [
                f"{place_name}, Northern Territory, Australia",
                f"{place_name}, NT, Australia",
                f"{place_name}, Australia",
                place_name
            ]
            
            for query in place_queries:
                try:
                    buildings = ox.features_from_place(
                        query,
                        tags={'building': True}
                    )
                    
                    if not buildings.empty:
                        logger.info(f"✅ Found buildings with query: {query}")
                        # Get the place boundary to find center
                        place = ox.geocode_to_gdf(query)
                        if not place.empty:
                            centroid = place.geometry.centroid.iloc[0]
                            return self._process_osmnx_buildings(
                                buildings, centroid.y, centroid.x
                            )
                        return self._process_osmnx_buildings(buildings, None, None)
                        
                except Exception:
                    continue
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ OSMnx place query failed: {e}")
            return None
    
    def _extract_via_overpass(self, latitude: float, longitude: float, 
                             radius: int) -> Optional[pd.DataFrame]:
        """Extract buildings using direct Overpass API query"""
        try:
            # Calculate bounding box
            lat_offset = radius / 111320
            lon_offset = radius / (111320 * abs(np.cos(np.radians(latitude))))
            
            south = latitude - lat_offset
            west = longitude - lon_offset
            north = latitude + lat_offset
            east = longitude + lon_offset
            
            # Overpass QL query - expanded to catch more building types
            overpass_query = f"""
            [out:json][timeout:30];
            (
              way["building"]({south},{west},{north},{east});
              relation["building"]({south},{west},{north},{east});
              node["building"]({south},{west},{north},{east});
            );
            out body;
            >;
            out skel qt;
            """
            
            response = requests.post(
                "http://overpass-api.de/api/interpreter",
                data=overpass_query,
                timeout=35,
                headers={'User-Agent': 'CommunityBuildingExtractor/1.0'}
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._process_overpass_response(data, latitude, longitude)
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Overpass API error: {e}")
            return None
    
    def _process_osmnx_buildings(self, buildings_gdf: gpd.GeoDataFrame, 
                                center_lat: Optional[float], 
                                center_lon: Optional[float]) -> pd.DataFrame:
        """Process raw building data from OSMnx"""
        if buildings_gdf.empty:
            return pd.DataFrame()
        
        buildings = buildings_gdf.copy()
        
        # Extract building coordinates (centroid)
        buildings['building_lat'] = buildings.geometry.centroid.y
        buildings['building_lon'] = buildings.geometry.centroid.x
        
        # Calculate distance from center if available
        if center_lat and center_lon:
            buildings['distance_from_center_m'] = buildings.apply(
                lambda row: self._calculate_distance(
                    center_lat, center_lon, row['building_lat'], row['building_lon']
                ), axis=1
            )
        else:
            buildings['distance_from_center_m'] = None
        
        # Calculate area if possible
        try:
            buildings_projected = buildings.to_crs('EPSG:3577')  # Australian Albers
            buildings['area_sqm'] = buildings_projected.geometry.area
        except:
            buildings['area_sqm'] = None
        
        # Create building IDs
        buildings['building_id'] = range(1, len(buildings) + 1)
        
        # Select relevant columns
        columns_to_keep = [
            'building_id', 'building_lat', 'building_lon', 'distance_from_center_m',
            'area_sqm', 'building', 'name', 'amenity', 'landuse', 'height',
            'addr:street', 'addr:housenumber'
        ]
        
        available_columns = [col for col in columns_to_keep if col in buildings.columns]
        return buildings[available_columns].copy()
    
    def _process_overpass_response(self, data: Dict, center_lat: float, 
                                  center_lon: float) -> Optional[pd.DataFrame]:
        """Process Overpass API response into DataFrame"""
        elements = data.get('elements', [])
        if not elements:
            return None
        
        # Build node lookup
        nodes = {}
        for element in elements:
            if element['type'] == 'node':
                nodes[element['id']] = (element['lon'], element['lat'])
        
        buildings_list = []
        building_id = 1
        
        for element in elements:
            if element['type'] == 'way' and 'tags' in element:
                # Get way nodes
                way_nodes = element.get('nodes', [])
                if not way_nodes:
                    continue
                
                # Calculate centroid
                coords = []
                for node_id in way_nodes:
                    if node_id in nodes:
                        coords.append(nodes[node_id])
                
                if coords:
                    center_lon_way = sum(coord[0] for coord in coords) / len(coords)
                    center_lat_way = sum(coord[1] for coord in coords) / len(coords)
                    
                    building_info = {
                        'building_id': building_id,
                        'building_lat': center_lat_way,
                        'building_lon': center_lon_way,
                        'building': element['tags'].get('building', 'yes'),
                        'name': element['tags'].get('name', ''),
                        'amenity': element['tags'].get('amenity', ''),
                        'landuse': element['tags'].get('landuse', ''),
                        'height': element['tags'].get('height', ''),
                        'addr:street': element['tags'].get('addr:street', ''),
                        'addr:housenumber': element['tags'].get('addr:housenumber', ''),
                        'osm_id': element['id'],
                        'distance_from_center_m': self._calculate_distance(
                            center_lat, center_lon, center_lat_way, center_lon_way
                        )
                    }
                    buildings_list.append(building_info)
                    building_id += 1
        
        if buildings_list:
            return pd.DataFrame(buildings_list)
        
        return None
    
    @staticmethod
    def _calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in meters using Haversine formula"""
        R = 6371000  # Earth's radius in meters
        
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c