#!/usr/bin/env python3
"""
Community Extractor Module
Main orchestrator class that ties all modules together
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional
import os

from .cache_manager import CacheManager
from .data_extractor import BuildingExtractor
from .geocoding_service import GeocodingService
from .building_classifier import BuildingClassifier
from .output_generator import OutputGenerator

logger = logging.getLogger(__name__)


class CommunityBuildingExtractor:
    """Main orchestrator for community building extraction"""
    
    def __init__(self, csv_file: str = "TownData.csv", output_dir: str = "output",
                 cache_dir: str = "cache", default_radius: int = 1500,
                 delay_between_requests: float = 2.0, enable_geocoding: bool = True,
                 cache_expiry_days: int = 30, force_refresh: bool = False):
        """
        Initialize the community building extractor
        
        Args:
            csv_file: Path to CSV file with community data
            output_dir: Base output directory
            cache_dir: Directory for caching data
            default_radius: Default search radius in meters
            delay_between_requests: Delay between API requests
            enable_geocoding: Whether to lookup addresses
            cache_expiry_days: Days to keep cached data
            force_refresh: Force refresh of cached data
        """
        self.csv_file = csv_file
        self.default_radius = default_radius
        self.enable_geocoding = enable_geocoding
        self.force_refresh = force_refresh
        
        # Initialize modules
        self.cache_manager = CacheManager(cache_dir, cache_expiry_days)
        self.building_extractor = BuildingExtractor(default_radius, delay_between_requests)
        self.geocoding_service = GeocodingService(delay_between_requests=0.5)
        self.building_classifier = BuildingClassifier()
        self.output_generator = OutputGenerator(output_dir)
        
        # Initialize statistics
        self.stats = {
            'total_communities': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'total_buildings_found': 0,
            'communities_with_buildings': 0,
            'communities_without_buildings': 0
        }
        
        # Load community data
        self.communities_df = self.load_community_data()
        
        logger.info(f"🏘️ Extractor initialized for {len(self.communities_df)} communities")
    
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
            
            logger.info(f"✅ Loaded {len(df)} valid communities from {self.csv_file}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading community data: {e}")
            raise
    
    def extract_buildings_for_community(self, row: pd.Series) -> Optional[pd.DataFrame]:
        """Extract and process buildings for a single community"""
        community_name = row['Community Name']
        latitude = row['Latitude']
        longitude = row['Longitude']
        
        # Check cache first
        cache_key = self.cache_manager.create_cache_key(
            community_name, latitude, longitude, self.default_radius
        )
        cached_buildings = self.cache_manager.get_cached_buildings(cache_key, self.force_refresh)
        
        if cached_buildings is not None:
            logger.info(f"📦 Using cached data for {community_name}")
            return cached_buildings
        
        # Extract fresh building data
        buildings_df = self.building_extractor.extract_buildings(
            community_name, latitude, longitude, self.default_radius
        )
        
        if buildings_df is None or buildings_df.empty:
            return None
        
        # Add community metadata
        buildings_df = self._add_community_metadata(buildings_df, row)
        
        # Classify buildings
        buildings_df = self.building_classifier.classify_buildings(buildings_df)
        
        # Add addresses if enabled
        if self.enable_geocoding:
            buildings_df = self.geocoding_service.add_addresses_to_buildings(
                buildings_df, self.cache_manager
            )
        
        # Cache the results
        self.cache_manager.save_buildings_to_cache(cache_key, buildings_df)
        
        return buildings_df
    
    def _add_community_metadata(self, buildings_df: pd.DataFrame, 
                               community_row: pd.Series) -> pd.DataFrame:
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
    
    def process_all_communities(self) -> Dict:
        """Process all communities and generate outputs"""
        logger.info(f"🚀 Starting batch processing of {len(self.communities_df)} communities")
        
        self.stats['total_communities'] = len(self.communities_df)
        results = []
        
        for idx, community_row in self.communities_df.iterrows():
            community_name = community_row['Community Name']
            
            try:
                # Create output directory
                community_dir = self.output_generator.create_community_output_dir(community_name)
                
                # Extract buildings
                buildings_df = self.extract_buildings_for_community(community_row)
                
                if buildings_df is not None and not buildings_df.empty:
                    # Generate outputs
                    metadata = {
                        'search_radius': self.default_radius,
                        'geocoding_enabled': self.enable_geocoding
                    }
                    
                    csv_path = self.output_generator.save_csv_output(
                        buildings_df, community_dir, community_name, metadata
                    )
                    
                    kml_path = self.output_generator.save_kml_output(
                        buildings_df, community_dir, community_name,
                        community_row['Latitude'], community_row['Longitude'],
                        self.default_radius, self.enable_geocoding
                    )
                    
                    self.output_generator.create_summary_report(
                        buildings_df, community_dir, community_name, metadata
                    )
                    
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
        
        # Merge all statistics
        all_stats = {
            **self.stats,
            **self.cache_manager.get_cache_statistics(),
            **self.geocoding_service.get_statistics(),
            **self.building_classifier.get_statistics()
        }
        
        # Generate final summary
        self.output_generator.generate_batch_summary(
            results, all_stats, self.csv_file, self.default_radius,
            self.enable_geocoding, str(self.cache_manager.cache_dir),
            self.force_refresh
        )
        
        return {
            'results': results,
            'statistics': all_stats
        }