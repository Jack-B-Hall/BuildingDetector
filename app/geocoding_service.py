#!/usr/bin/env python3
"""
Geocoding Service Module
Handles reverse geocoding for building addresses
"""

import requests
import logging
import time
import pandas as pd
from typing import Optional, List

logger = logging.getLogger(__name__)


class GeocodingService:
    """Provides reverse geocoding services for building addresses"""
    
    def __init__(self, delay_between_requests: float = 0.5):
        """
        Initialize geocoding service
        
        Args:
            delay_between_requests: Delay between geocoding requests
        """
        self.delay_between_requests = delay_between_requests
        self.user_agent = 'CommunityBuildingExtractor/1.0 (Contact: your-email@example.com)'
        
        # Statistics
        self.stats = {
            'geocoding_success': 0,
            'geocoding_failed': 0
        }
    
    def add_addresses_to_buildings(self, buildings_df: pd.DataFrame, 
                                 cache_manager=None) -> pd.DataFrame:
        """
        Add address information to buildings using reverse geocoding
        
        Args:
            buildings_df: DataFrame with building data
            cache_manager: Optional cache manager for caching results
            
        Returns:
            DataFrame with added address column
        """
        if buildings_df.empty:
            return buildings_df
        
        logger.info(f"🔍 Looking up addresses for {len(buildings_df)} buildings...")
        
        addresses = []
        
        for idx, building in buildings_df.iterrows():
            lat = building.get('building_lat')
            lon = building.get('building_lon')
            
            if pd.isna(lat) or pd.isna(lon):
                addresses.append('')
                continue
            
            # Check cache if available
            address = ''
            geocode_key = f"{lat:.6f}_{lon:.6f}"
            
            if cache_manager:
                cached_address = cache_manager.get_cached_address(geocode_key)
                if cached_address is not None:
                    addresses.append(cached_address)
                    continue
            
            # Perform reverse geocoding
            address = self.reverse_geocode(lat, lon)
            addresses.append(address)
            
            # Cache the result if cache manager available
            if cache_manager and address:
                cache_manager.save_address_to_cache(geocode_key, address)
            
            # Delay between requests
            time.sleep(self.delay_between_requests)
        
        buildings_df['address'] = addresses
        
        # Update statistics
        successful_geocodes = sum(1 for addr in addresses if addr)
        self.stats['geocoding_success'] += successful_geocodes
        self.stats['geocoding_failed'] += len(addresses) - successful_geocodes
        
        logger.info(f"🏠 Successfully geocoded {successful_geocodes}/{len(addresses)} addresses")
        
        return buildings_df
    
    def reverse_geocode(self, latitude: float, longitude: float) -> str:
        """
        Perform reverse geocoding using Nominatim
        
        Args:
            latitude: Building latitude
            longitude: Building longitude
            
        Returns:
            Formatted address string or empty string if failed
        """
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
                'User-Agent': self.user_agent
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'address' in data:
                    return self._format_address(data['address'])
                
                return data.get('display_name', '')
            
            return ''
            
        except Exception as e:
            logger.warning(f"⚠️ Geocoding failed for {latitude}, {longitude}: {e}")
            return ''
    
    def _format_address(self, address_data: dict) -> str:
        """
        Format address components into readable string
        
        Args:
            address_data: Address components from Nominatim
            
        Returns:
            Formatted address string
        """
        addr_parts = []
        
        # Build address string from components
        if 'house_number' in address_data:
            addr_parts.append(address_data['house_number'])
        
        if 'road' in address_data:
            addr_parts.append(address_data['road'])
        elif 'street' in address_data:
            addr_parts.append(address_data['street'])
        
        # Add suburb/city
        for key in ['suburb', 'city', 'town', 'village']:
            if key in address_data:
                addr_parts.append(address_data[key])
                break
        
        if 'state' in address_data:
            addr_parts.append(address_data['state'])
        
        if 'postcode' in address_data:
            addr_parts.append(address_data['postcode'])
        
        return ', '.join(addr_parts) if addr_parts else ''
    
    def get_statistics(self) -> dict:
        """Get geocoding statistics"""
        total = self.stats['geocoding_success'] + self.stats['geocoding_failed']
        success_rate = (self.stats['geocoding_success'] / total * 100 
                       if total > 0 else 0)
        
        return {
            'total_geocoded': total,
            'successful': self.stats['geocoding_success'],
            'failed': self.stats['geocoding_failed'],
            'success_rate': success_rate
        }