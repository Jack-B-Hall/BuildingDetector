#!/usr/bin/env python3
"""
Cache Manager Module
Handles caching for OSM metadata, geocoding, and building data
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from .utils import vprint

class CacheManager:
    """Manages caching for OSM metadata, geocoding, and building data"""
    
    def __init__(self, cache_dir: str = "cache", expiry_days: int = 30):
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