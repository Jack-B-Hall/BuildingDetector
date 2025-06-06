#!/usr/bin/env python3
"""
Output Generator Module
Handles generation of CSV, KML, and summary reports
"""

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class OutputGenerator:
    """Generates various output formats for building data"""
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize output generator
        
        Args:
            output_dir: Base output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    @staticmethod
    def create_safe_filename(name: str) -> str:
        """Create filesystem-safe filename from community name"""
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_', '.')).strip()
        safe_name = safe_name.replace(' ', '_').replace('__', '_')
        return safe_name.lower()
    
    def create_community_output_dir(self, community_name: str) -> Path:
        """Create output directory for specific community"""
        safe_name = self.create_safe_filename(community_name)
        community_dir = self.output_dir / safe_name
        community_dir.mkdir(exist_ok=True)
        return community_dir
    
    def save_csv_output(self, buildings_df: pd.DataFrame, community_dir: Path, 
                       community_name: str, metadata: Dict) -> Path:
        """
        Save buildings data as CSV with metadata header
        
        Args:
            buildings_df: Building data
            community_dir: Output directory
            community_name: Community name
            metadata: Additional metadata to include
            
        Returns:
            Path to saved CSV file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{self.create_safe_filename(community_name)}_buildings_{timestamp}.csv"
        csv_path = community_dir / csv_filename
        
        # Add metadata header
        with open(csv_path, 'w') as f:
            f.write(f"# Building extraction for {community_name}\n")
            f.write(f"# Extraction date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Search radius: {metadata.get('search_radius', 'unknown')}m\n")
            f.write(f"# Total buildings found: {len(buildings_df)}\n")
            f.write(f"# Geocoding enabled: {metadata.get('geocoding_enabled', False)}\n")
            f.write("# \n")
        
        buildings_df.to_csv(csv_path, mode='a', index=False)
        logger.info(f"💾 CSV saved: {csv_path}")
        return csv_path
    
    def save_kml_output(self, buildings_df: pd.DataFrame, community_dir: Path,
                       community_name: str, community_lat: float, 
                       community_lon: float, search_radius: int,
                       geocoding_enabled: bool = False) -> Path:
        """
        Save buildings data as KML for visualization
        
        Args:
            buildings_df: Building data
            community_dir: Output directory
            community_name: Community name
            community_lat: Community center latitude
            community_lon: Community center longitude
            search_radius: Search radius in meters
            geocoding_enabled: Whether geocoding was used
            
        Returns:
            Path to saved KML file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        kml_filename = f"{self.create_safe_filename(community_name)}_buildings_{timestamp}.kml"
        kml_path = community_dir / kml_filename
        
        # Create KML structure
        kml = ET.Element('kml', xmlns='http://www.opengis.net/kml/2.2')
        document = ET.SubElement(kml, 'Document')
        
        # Add document info
        ET.SubElement(document, 'name').text = f"{community_name} Buildings"
        ET.SubElement(document, 'description').text = f"""
        Building footprints for {community_name}
        Extracted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Total buildings: {len(buildings_df)}
        Search radius: {search_radius}m
        Geocoding: {'Enabled' if geocoding_enabled else 'Disabled'}
        """
        
        # Add styles
        self._add_kml_styles(document)
        
        # Add community center
        self._add_center_placemark(document, community_name, community_lat, 
                                  community_lon, len(buildings_df), search_radius)
        
        # Add search radius circle
        self._add_search_radius(document, community_lat, community_lon, search_radius)
        
        # Add building placemarks
        for _, building in buildings_df.iterrows():
            self._add_building_placemark(document, building, geocoding_enabled)
        
        # Write KML file
        rough_string = ET.tostring(kml, 'unicode')
        reparsed = minidom.parseString(rough_string)
        
        with open(kml_path, 'w', encoding='utf-8') as f:
            f.write(reparsed.toprettyxml(indent="  "))
        
        logger.info(f"🗺️ KML saved: {kml_path}")
        return kml_path
    
    def _add_kml_styles(self, document):
        """Add KML styles for different building categories"""
        styles = {
            'Residential': {
                'color': 'ff0000ff',  # Red
                'icon': 'http://maps.google.com/mapfiles/kml/pushpin/red-pushpin.png'
            },
            'Commercial': {
                'color': 'ff00ff00',  # Green
                'icon': 'http://maps.google.com/mapfiles/kml/pushpin/grn-pushpin.png'
            },
            'Community': {
                'color': 'ffff0000',  # Blue
                'icon': 'http://maps.google.com/mapfiles/kml/pushpin/blue-pushpin.png'
            },
            'Unknown': {
                'color': 'ff888888',  # Gray
                'icon': 'http://maps.google.com/mapfiles/kml/pushpin/wht-pushpin.png'
            },
            'Center': {
                'color': 'ffff00ff',  # Magenta
                'icon': 'http://maps.google.com/mapfiles/kml/shapes/target.png',
                'scale': '1.2'
            }
        }
        
        for category, style in styles.items():
            style_elem = ET.SubElement(document, 'Style', id=f'{category}Style')
            icon_style = ET.SubElement(style_elem, 'IconStyle')
            ET.SubElement(icon_style, 'color').text = style['color']
            if 'scale' in style:
                ET.SubElement(icon_style, 'scale').text = style['scale']
            icon = ET.SubElement(icon_style, 'Icon')
            ET.SubElement(icon, 'href').text = style['icon']
        
        # Add search radius circle style
        circle_style = ET.SubElement(document, 'Style', id='SearchRadiusStyle')
        line_style = ET.SubElement(circle_style, 'LineStyle')
        ET.SubElement(line_style, 'color').text = 'ff0000ff'  # Blue line
        ET.SubElement(line_style, 'width').text = '2'
        poly_style = ET.SubElement(circle_style, 'PolyStyle')
        ET.SubElement(poly_style, 'color').text = '330000ff'  # Semi-transparent blue
        ET.SubElement(poly_style, 'outline').text = '1'
    
    def _add_center_placemark(self, document, community_name: str, lat: float, 
                             lon: float, building_count: int, radius: int):
        """Add community center placemark"""
        placemark = ET.SubElement(document, 'Placemark')
        ET.SubElement(placemark, 'name').text = f"🎯 {community_name} Center"
        ET.SubElement(placemark, 'description').text = f"""
        <![CDATA[
        <b>Community Center Point</b><br/>
        Search radius: {radius}m<br/>
        Coordinates: {lat:.6f}, {lon:.6f}<br/>
        Buildings found: {building_count}
        ]]>
        """
        ET.SubElement(placemark, 'styleUrl').text = '#CenterStyle'
        point = ET.SubElement(placemark, 'Point')
        ET.SubElement(point, 'coordinates').text = f"{lon},{lat},0"
    
    def _add_search_radius(self, document, center_lat: float, center_lon: float, 
                          radius: int):
        """Add search radius circle to KML"""
        placemark = ET.SubElement(document, 'Placemark')
        ET.SubElement(placemark, 'name').text = f"Search Radius ({radius}m)"
        ET.SubElement(placemark, 'styleUrl').text = '#SearchRadiusStyle'
        
        polygon = ET.SubElement(placemark, 'Polygon')
        ET.SubElement(polygon, 'extrude').text = '0'
        ET.SubElement(polygon, 'altitudeMode').text = 'clampToGround'
        outer_boundary = ET.SubElement(polygon, 'outerBoundaryIs')
        linear_ring = ET.SubElement(outer_boundary, 'LinearRing')
        
        # Generate circle coordinates
        circle_coords = []
        for i in range(361):  # 361 to close the circle
            angle = i * np.pi / 180
            lat_offset = (radius / 111320) * np.cos(angle)
            lon_offset = (radius / (111320 * np.cos(np.radians(center_lat)))) * np.sin(angle)
            
            circle_lat = center_lat + lat_offset
            circle_lon = center_lon + lon_offset
            circle_coords.append(f"{circle_lon},{circle_lat},0")
        
        ET.SubElement(linear_ring, 'coordinates').text = ' '.join(circle_coords)
    
    def _add_building_placemark(self, document, building, geocoding_enabled: bool):
        """Add individual building placemark"""
        placemark = ET.SubElement(document, 'Placemark')
        
        # Generate building name
        building_id = building.get('building_id', 'Unknown')
        original_name = building.get('name', '')
        
        if pd.isna(original_name) or str(original_name).lower() in ['nan', 'none', '', 'null']:
            building_name = f"Building {building_id}"
        else:
            building_name = str(original_name)
        
        ET.SubElement(placemark, 'name').text = building_name
        
        # Create description
        address_info = ""
        if geocoding_enabled and 'address' in building and building['address']:
            if not pd.isna(building['address']) and str(building['address']).lower() not in ['nan', 'none', '']:
                address_info = f"Address: {building['address']}<br/>"
        
        # Check if this was an unknown building defaulted to residential
        category_display = building.get('building_category', 'Unknown')
        if building.get('classification_unknown', False) and category_display == 'Residential':
            category_display = 'Residential (Unknown type)'
        
        description = f"""
        <![CDATA[
        <b>Building Details:</b><br/>
        ID: {building_id}<br/>
        Category: {category_display}<br/>
        Type: {building.get('building', 'Unknown')}<br/>
        {address_info}Distance from center: {building.get('distance_from_center_m', 0):.0f}m<br/>
        Coordinates: {building.get('building_lat', 0):.6f}, {building.get('building_lon', 0):.6f}
        ]]>
        """
        ET.SubElement(placemark, 'description').text = description
        
        # Apply style
        category = building.get('building_category', 'Unknown')
        ET.SubElement(placemark, 'styleUrl').text = f'#{category}Style'
        
        # Add point
        point = ET.SubElement(placemark, 'Point')
        ET.SubElement(point, 'coordinates').text = (
            f"{building.get('building_lon', 0)},{building.get('building_lat', 0)},0"
        )
    
    def create_summary_report(self, buildings_df: pd.DataFrame, community_dir: Path,
                            community_name: str, metadata: Dict):
        """Create summary report for community"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = community_dir / f"{self.create_safe_filename(community_name)}_summary_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write(f"BUILDING EXTRACTION SUMMARY\n")
            f.write(f"{'=' * 50}\n")
            f.write(f"Community: {community_name}\n")
            f.write(f"Extraction date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Search radius: {metadata.get('search_radius', 'unknown')}m\n")
            f.write(f"Geocoding enabled: {metadata.get('geocoding_enabled', False)}\n")
            f.write(f"Total buildings found: {len(buildings_df)}\n\n")
            
            if not buildings_df.empty:
                # Category breakdown
                if 'building_category' in buildings_df.columns:
                    f.write("Buildings by category:\n")
                    category_counts = buildings_df['building_category'].value_counts()
                    for category, count in category_counts.items():
                        if 'classification_unknown' in buildings_df.columns:
                            unknown_count = len(buildings_df[
                                (buildings_df['building_category'] == category) & 
                                (buildings_df['classification_unknown'] == True)
                            ])
                            if unknown_count > 0:
                                f.write(f"  {category}: {count} ({unknown_count} unknown types)\n")
                            else:
                                f.write(f"  {category}: {count}\n")
                        else:
                            f.write(f"  {category}: {count}\n")
                    f.write("\n")
                
                # Distance analysis
                if 'distance_from_center_m' in buildings_df.columns:
                    f.write("Distance analysis:\n")
                    f.write(f"  Average distance: {buildings_df['distance_from_center_m'].mean():.0f}m\n")
                    f.write(f"  Maximum distance: {buildings_df['distance_from_center_m'].max():.0f}m\n")
                    f.write(f"  Minimum distance: {buildings_df['distance_from_center_m'].min():.0f}m\n\n")
                
                # Address analysis
                if metadata.get('geocoding_enabled') and 'address' in buildings_df.columns:
                    addresses_found = buildings_df['address'].notna().sum()
                    f.write("Address lookup results:\n")
                    f.write(f"  Buildings with addresses: {addresses_found}/{len(buildings_df)}\n")
                    f.write(f"  Success rate: {addresses_found/len(buildings_df)*100:.1f}%\n")
        
        logger.info(f"📄 Summary saved: {report_path}")
    
    def generate_batch_summary(self, results: List[Dict], statistics: Dict, 
                             input_file: str, search_radius: int, 
                             geocoding_enabled: bool, cache_dir: str,
                             force_refresh: bool):
        """Generate final batch summary report"""
        summary_path = self.output_dir / f"batch_extraction_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(summary_path, 'w') as f:
            f.write("BATCH BUILDING EXTRACTION SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(f"Extraction date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input file: {input_file}\n")
            f.write(f"Search radius: {search_radius}m\n")
            f.write(f"Geocoding enabled: {geocoding_enabled}\n")
            f.write(f"Cache directory: {cache_dir}\n")
            f.write(f"Force refresh: {force_refresh}\n\n")
            
            f.write("STATISTICS:\n")
            f.write("-" * 20 + "\n")
            for key, value in statistics.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            
            f.write(f"\nDETAILED RESULTS:\n")
            f.write("-" * 20 + "\n")
            for result in results:
                status_emoji = {"success": "✅", "no_buildings": "⚠️", "failed": "❌"}
                emoji = status_emoji.get(result['status'], "❓")
                f.write(f"{emoji} {result['community']}: {result['buildings_found']} buildings ({result['status']})\n")
        
        logger.info(f"📊 Final summary saved: {summary_path}")