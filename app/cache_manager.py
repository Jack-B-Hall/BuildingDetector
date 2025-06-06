#!/usr/bin/env python3
"""
Cache Manager Module
Handles all caching operations for building data and geocoding results
"""

import pickle
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching of building data and geocoding results"""
    
    def __init__(self, cache_dir: str = "cache", cache_expiry_days: int = 30, 
                 geocoding_expiry_days: int = 7):
        """
        Initialize cache manager
        
        Args:
            cache_dir: Base cache directory
            cache_expiry_days: Days to keep building data cached
            geocoding_expiry_days: Days to keep geocoding results cached
        """
        self.cache_dir = Path(cache_dir)
        self.cache_expiry_days = cache_expiry_days
        self.geocoding_expiry_days = geocoding_expiry_days
        
        # Create cache directories
        self.cache_dir.mkdir(exist_ok=True)
        self.buildings_cache_dir = self.cache_dir / "buildings"
        self.geocoding_cache_dir = self.cache_dir / "geocoding"
        self.buildings_cache_dir.mkdir(exist_ok=True)
        self.geocoding_cache_dir.mkdir(exist_ok=True)
        
        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'geocoding_cache_hits': 0,
            'geocoding_cache_misses': 0
        }
    
    def create_cache_key(self, community_name: str, latitude: float, 
                        longitude: float, radius: int) -> str:
        """Create unique cache key for community building data"""
        key_string = f"{community_name}_{latitude}_{longitude}_{radius}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_cached_buildings(self, cache_key: str, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Retrieve cached building data if available and not expired
        
        Args:
            cache_key: Unique cache identifier
            force_refresh: Force cache miss if True
            
        Returns:
            Cached DataFrame or None if not found/expired
        """
        cache_file = self.buildings_cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists() or force_refresh:
            self.stats['cache_misses'] += 1
            return None
        
        try:
            # Check if cache is expired
            cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - cache_time > timedelta(days=self.cache_expiry_days):
                logger.info(f"🕐 Cache expired for {cache_key}")
                self.stats['cache_misses'] += 1
                return None
            
            # Load cached data
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            self.stats['cache_hits'] += 1
            logger.info(f"💾 Cache hit: {cache_key}")
            return cached_data
            
        except Exception as e:
            logger.warning(f"⚠️ Error loading cache {cache_key}: {e}")
            self.stats['cache_misses'] += 1
            return None
    
    def save_buildings_to_cache(self, cache_key: str, buildings_df: pd.DataFrame):
        """Save building data to cache"""
        try:
            cache_file = self.buildings_cache_dir / f"{cache_key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(buildings_df, f)
            
            logger.info(f"💾 Cached buildings data: {cache_key}")
            
        except Exception as e:
            logger.warning(f"⚠️ Error saving to cache {cache_key}: {e}")
    
    def get_cached_address(self, geocode_key: str) -> Optional[str]:
        """Get cached geocoding result"""
        cache_file = self.geocoding_cache_dir / f"{geocode_key}.txt"
        
        if not cache_file.exists():
            self.stats['geocoding_cache_misses'] += 1
            return None
        
        try:
            # Check if cache is expired
            cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - cache_time > timedelta(days=self.geocoding_expiry_days):
                self.stats['geocoding_cache_misses'] += 1
                return None
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                address = f.read().strip()
            
            self.stats['geocoding_cache_hits'] += 1
            return address
                
        except Exception:
            self.stats['geocoding_cache_misses'] += 1
            return None
    
    def save_address_to_cache(self, geocode_key: str, address: str):
        """Save geocoding result to cache"""
        try:
            cache_file = self.geocoding_cache_dir / f"{geocode_key}.txt"
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(address)
        except Exception as e:
            logger.warning(f"⚠️ Error caching address: {e}")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_building_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        building_hit_rate = (self.stats['cache_hits'] / total_building_requests * 100 
                           if total_building_requests > 0 else 0)
        
        total_geocoding_requests = (self.stats['geocoding_cache_hits'] + 
                                  self.stats['geocoding_cache_misses'])
        geocoding_hit_rate = (self.stats['geocoding_cache_hits'] / total_geocoding_requests * 100
                            if total_geocoding_requests > 0 else 0)
        
        return {
            'building_cache_hits': self.stats['cache_hits'],
            'building_cache_misses': self.stats['cache_misses'],
            'building_cache_hit_rate': building_hit_rate,
            'geocoding_cache_hits': self.stats['geocoding_cache_hits'],
            'geocoding_cache_misses': self.stats['geocoding_cache_misses'],
            'geocoding_cache_hit_rate': geocoding_hit_rate,
            'total_cached_buildings': len(list(self.buildings_cache_dir.glob('*.pkl'))),
            'total_cached_addresses': len(list(self.geocoding_cache_dir.glob('*.txt')))
        }
    
    def clear_expired_cache(self):
        """Remove expired cache files"""
        now = datetime.now()
        cleared_buildings = 0
        cleared_addresses = 0
        
        # Clear expired building cache
        for cache_file in self.buildings_cache_dir.glob('*.pkl'):
            cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if now - cache_time > timedelta(days=self.cache_expiry_days):
                cache_file.unlink()
                cleared_buildings += 1
        
        # Clear expired geocoding cache
        for cache_file in self.geocoding_cache_dir.glob('*.txt'):
            cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if now - cache_time > timedelta(days=self.geocoding_expiry_days):
                cache_file.unlink()
                cleared_addresses += 1
        
        if cleared_buildings > 0 or cleared_addresses > 0:
            logger.info(f"🧹 Cleared {cleared_buildings} expired building caches and "
                       f"{cleared_addresses} expired address caches")