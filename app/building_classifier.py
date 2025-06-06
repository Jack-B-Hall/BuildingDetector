#!/usr/bin/env python3
"""
Building Classifier Module
Handles classification of buildings into categories
"""

import pandas as pd
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class BuildingClassifier:
    """Classifies buildings into categories based on OSM tags"""
    
    # Classification rules
    RESIDENTIAL_TAGS = [
        'house', 'residential', 'detached', 'cabin', 'hut', 'apartments',
        'terrace', 'dormitory', 'bungalow', 'semidetached_house'
    ]
    
    COMMUNITY_TAGS = [
        'school', 'hospital', 'church', 'community', 'government', 'public',
        'kindergarten', 'college', 'university', 'library', 'townhall',
        'civic', 'fire_station', 'police'
    ]
    
    COMMERCIAL_TAGS = [
        'commercial', 'retail', 'shop', 'office', 'warehouse', 'industrial',
        'supermarket', 'store', 'kiosk', 'service', 'factory'
    ]
    
    # Amenity tags for classification
    COMMUNITY_AMENITIES = [
        'school', 'hospital', 'place_of_worship', 'community_centre',
        'library', 'townhall', 'police', 'fire_station', 'post_office',
        'courthouse', 'prison', 'embassy', 'government'
    ]
    
    COMMERCIAL_AMENITIES = [
        'shop', 'restaurant', 'cafe', 'fast_food', 'bar', 'pub',
        'bank', 'pharmacy', 'fuel', 'marketplace'
    ]
    
    def __init__(self):
        """Initialize building classifier"""
        self.stats = {
            'residential': 0,
            'community': 0,
            'commercial': 0,
            'unknown': 0,
            'unknown_defaulted_residential': 0
        }
    
    def classify_buildings(self, buildings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify buildings and add category information
        
        Args:
            buildings_df: DataFrame with building data
            
        Returns:
            DataFrame with added classification columns
        """
        if buildings_df.empty:
            return buildings_df
        
        logger.info(f"🏷️ Classifying {len(buildings_df)} buildings...")
        
        # Apply classification
        classifications = buildings_df.apply(self._classify_building, axis=1)
        buildings_df['building_category'] = classifications.apply(lambda x: x[0])
        buildings_df['classification_unknown'] = classifications.apply(lambda x: x[1])
        
        # Update statistics
        category_counts = buildings_df['building_category'].value_counts()
        for category, count in category_counts.items():
            self.stats[category.lower()] = self.stats.get(category.lower(), 0) + count
        
        unknown_count = buildings_df['classification_unknown'].sum()
        self.stats['unknown_defaulted_residential'] += unknown_count
        
        logger.info(f"📊 Classification complete: "
                   f"R:{category_counts.get('Residential', 0)} "
                   f"C:{category_counts.get('Community', 0)} "
                   f"M:{category_counts.get('Commercial', 0)}")
        
        return buildings_df
    
    def _classify_building(self, row) -> Tuple[str, bool]:
        """
        Classify a single building
        
        Args:
            row: Building data row
            
        Returns:
            Tuple of (category, is_unknown_defaulted)
        """
        building_tag = str(row.get('building', '')).lower()
        amenity_tag = str(row.get('amenity', '')).lower()
        name = str(row.get('name', '')).lower()
        
        # Priority 1: Check amenity tags (most reliable)
        if amenity_tag and amenity_tag != 'nan':
            if amenity_tag in self.COMMUNITY_AMENITIES:
                return ('Community', False)
            if amenity_tag in self.COMMERCIAL_AMENITIES:
                return ('Commercial', False)
        
        # Priority 2: Check building tags
        if building_tag and building_tag != 'nan':
            # Check community buildings
            if any(tag in building_tag for tag in self.COMMUNITY_TAGS):
                return ('Community', False)
            
            # Check commercial buildings
            if any(tag in building_tag for tag in self.COMMERCIAL_TAGS):
                return ('Commercial', False)
            
            # Check residential buildings
            if any(tag in building_tag for tag in self.RESIDENTIAL_TAGS):
                return ('Residential', False)
        
        # Priority 3: Check name for hints
        if name and name != 'nan':
            # Community keywords in name
            community_keywords = ['school', 'church', 'hospital', 'clinic', 'council',
                                'community', 'centre', 'center', 'library', 'government']
            if any(keyword in name for keyword in community_keywords):
                return ('Community', False)
            
            # Commercial keywords in name
            commercial_keywords = ['shop', 'store', 'market', 'office', 'business',
                                 'company', 'warehouse', 'factory']
            if any(keyword in name for keyword in commercial_keywords):
                return ('Commercial', False)
        
        # Default: Unknown buildings to Residential (but flag them)
        if building_tag == 'yes' or not building_tag or building_tag == 'nan':
            return ('Residential', True)  # True indicates it was defaulted
        
        # If we have a specific building tag we don't recognize, still default to Residential
        return ('Residential', True)
    
    def get_statistics(self) -> dict:
        """Get classification statistics"""
        total = sum(self.stats.values()) - self.stats.get('unknown_defaulted_residential', 0)
        
        return {
            'total_classified': total,
            'residential': self.stats.get('residential', 0),
            'community': self.stats.get('community', 0),
            'commercial': self.stats.get('commercial', 0),
            'unknown_defaulted': self.stats.get('unknown_defaulted_residential', 0)
        }