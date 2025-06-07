#!/usr/bin/env python3
"""
Output Generators Module
Handles CSV, KML, and summary file generation with proper XML escaping
"""

import os
import math
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any

from .utils import vprint, escape_xml_text, sanitize_filename

def save_buildings_to_csv(buildings: List[Dict[str, Any]], output_file: str) -> Dict[str, Any]:
    """Save buildings to CSV and return summary statistics"""
    if not buildings:
        return {'count': 0, 'types': {}, 'distance_range': (0, 0), 'height_stats': None}
    
    df = pd.DataFrame(buildings)
    df = df.sort_values('distance_km')
    df['building_id'] = range(1, len(df) + 1)
    
    # Reorder columns for better readability
    column_order = [
        'building_id', 'latitude', 'longitude', 'name', 'building_category',
        'amenity', 'building_type', 'addr_street', 'addr_housenumber', 'address',
        'height', 'building_use', 'landuse', 'shop', 'office', 'distance_km',
        'distance_m', 'area_sqm', 'confidence', 'quadkey', 'polygon_points', 'description'
    ]
    
    # Only include columns that exist
    final_columns = [col for col in column_order if col in df.columns]
    output_df = df[final_columns].copy()
    
    # Round coordinates
    output_df['latitude'] = output_df['latitude'].round(6)
    output_df['longitude'] = output_df['longitude'].round(6)
    
    output_df.to_csv(output_file, index=False)
    
    # Generate summary statistics
    type_counts = df['building_category'].value_counts().to_dict()
    distance_range = (df['distance_km'].min(), df['distance_km'].max())
    
    height_buildings = df[df['height'] > 0]
    height_stats = None
    if len(height_buildings) > 0:
        height_stats = {
            'count': len(height_buildings),
            'average': height_buildings['height'].mean(),
            'range': (height_buildings['height'].min(), height_buildings['height'].max())
        }
    
    return {
        'count': len(buildings),
        'types': type_counts,
        'distance_range': distance_range,
        'height_stats': height_stats
    }

def save_buildings_to_kml(buildings: List[Dict[str, Any]], kml_file: str, center_lat: float, center_lon: float, radius_km: float, community_name: str):
    """Save buildings to KML format with color-coded categories and proper XML escaping"""
    if not buildings:
        return
    
    buildings.sort(key=lambda x: x['distance_km'])
    
    # Escape community name for XML
    escaped_community_name = escape_xml_text(community_name)
    
    kml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
    <name>{escaped_community_name} - Building Footprints</name>
    <description>Building locations for {escaped_community_name} with OSM metadata and classification</description>
    
    <!-- Style definitions -->
    <Style id="Residential">
        <IconStyle>
            <color>ff00ff00</color>
            <scale>1.0</scale>
            <Icon><href>http://maps.google.com/mapfiles/kml/shapes/homegardenbusiness.png</href></Icon>
        </IconStyle>
    </Style>
    <Style id="Community">
        <IconStyle>
            <color>ffff0000</color>
            <scale>1.4</scale>
            <Icon><href>http://maps.google.com/mapfiles/kml/shapes/schools.png</href></Icon>
        </IconStyle>
    </Style>
    <Style id="Commercial">
        <IconStyle>
            <color>ff00ffff</color>
            <scale>1.2</scale>
            <Icon><href>http://maps.google.com/mapfiles/kml/shapes/shopping.png</href></Icon>
        </IconStyle>
    </Style>
    <Style id="search_center">
        <IconStyle>
            <color>ffff00ff</color>
            <scale>2.0</scale>
            <Icon><href>http://maps.google.com/mapfiles/kml/shapes/target.png</href></Icon>
        </IconStyle>
    </Style>
    
    <!-- Search center point -->
    <Placemark>
        <name>{escaped_community_name} Center</name>
        <description>{escaped_community_name} search center: {radius_km}km radius</description>
        <styleUrl>#search_center</styleUrl>
        <Point>
            <coordinates>{center_lon},{center_lat},0</coordinates>
        </Point>
    </Placemark>
    
    <!-- Search radius circle -->
    <Placemark>
        <name>{escaped_community_name} Search Radius ({radius_km}km)</name>
        <description>Search area boundary for {escaped_community_name}</description>
        <Style>
            <LineStyle>
                <color>7fff0000</color>
                <width>2</width>
            </LineStyle>
            <PolyStyle>
                <color>1fff0000</color>
            </PolyStyle>
        </Style>
        <Polygon>
            <outerBoundaryIs>
                <LinearRing>
                    <coordinates>'''
    
    # Generate circle coordinates
    circle_points = []
    for i in range(36):
        angle = i * 10 * math.pi / 180
        lat_offset = (radius_km / 111.0) * math.cos(angle)
        lon_offset = (radius_km / 111.0) * math.sin(angle) / math.cos(math.radians(center_lat))
        circle_lat = center_lat + lat_offset
        circle_lon = center_lon + lon_offset
        circle_points.append(f"{circle_lon},{circle_lat},0")
    
    circle_points.append(circle_points[0])
    kml_content += "\n                        ".join(circle_points)
    
    kml_content += f'''
                    </coordinates>
                </LinearRing>
            </outerBoundaryIs>
        </Polygon>
    </Placemark>
    
    <!-- Buildings by category -->'''
    
    # Group buildings by category
    categories = {}
    for building in buildings:
        category = building['building_category']
        if category not in categories:
            categories[category] = []
        categories[category].append(building)
    
    for category, cat_buildings in categories.items():
        escaped_category = escape_xml_text(category)
        kml_content += f'''
    <Folder>
        <name>{escaped_category} Buildings ({len(cat_buildings)})</name>
        <description>{escaped_category} buildings in {escaped_community_name}</description>'''
        
        for i, building in enumerate(cat_buildings, 1):
            # Safely get and escape building name
            building_name = building.get('name', '')
            if not building_name or building_name.lower() in ['', 'nan', 'none']:
                building_name = f'{community_name} Building {i}'
            
            escaped_building_name = escape_xml_text(building_name)
            escaped_description = escape_xml_text(building['description'])
            
            kml_content += f'''
        <Placemark>
            <name>{escaped_building_name}</name>
            <description><![CDATA[
                <b>{escaped_building_name}</b><br/>
                <b>Category:</b> {escape_xml_text(building['building_category'])}<br/>
                <b>Description:</b> {escaped_description}<br/>
                <br/>
                <b>Details:</b><br/>
                Coordinates: {building['latitude']:.6f}, {building['longitude']:.6f}<br/>
                Distance: {building['distance_km']:.3f} km<br/>
                Area: {building['area_sqm']:.1f} sqm<br/>
                Height: {building['height']}m<br/>
                Confidence: {building['confidence']}<br/>
                QuadKey: {building['quadkey']}
            ]]></description>
            <styleUrl>#{building['building_category']}</styleUrl>
            <Point>
                <coordinates>{building['longitude']},{building['latitude']},0</coordinates>
            </Point>
        </Placemark>'''
        
        kml_content += '''
    </Folder>'''
    
    kml_content += '''
</Document>
</kml>'''
    
    with open(kml_file, 'w', encoding='utf-8') as f:
        f.write(kml_content)

def generate_community_summary(buildings: List[Dict[str, Any]], stats: Dict[str, Any], 
                             output_file: str, community_name: str, lat: float, lon: float, radius: float):
    """Generate individual community summary"""
    
    summary = f"""
🏘️  {community_name.upper()} - BUILDING SUMMARY
{'=' * 60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📍 LOCATION DETAILS
{'=' * 30}
Community: {community_name}
Coordinates: {lat:.6f}, {lon:.6f}
Search radius: {radius}km
Total buildings found: {stats['count']}

📊 BUILDING CATEGORIES
{'=' * 30}"""
    
    if stats['count'] > 0:
        for category, count in stats['types'].items():
            percentage = (count / stats['count']) * 100
            summary += f"\n{category}: {count} ({percentage:.1f}%)"
        
        summary += f"""

📏 DISTANCE & SIZE ANALYSIS
{'=' * 30}
Distance range: {stats['distance_range'][0]:.3f} - {stats['distance_range'][1]:.3f} km"""
        
        if stats['height_stats']:
            hs = stats['height_stats']
            summary += f"""
Buildings with height data: {hs['count']}
Average height: {hs['average']:.1f}m
Height range: {hs['range'][0]:.1f} - {hs['range'][1]:.1f}m"""
        
        # Add detailed building list
        summary += f"""

📋 DETAILED BUILDING LIST
{'=' * 30}"""
        
        buildings_by_category = {}
        for building in buildings:
            category = building['building_category']
            if category not in buildings_by_category:
                buildings_by_category[category] = []
            buildings_by_category[category].append(building)
        
        for category in ['Community', 'Commercial', 'Residential']:
            if category in buildings_by_category:
                summary += f"\n\n{category.upper()} BUILDINGS ({len(buildings_by_category[category])}):"
                summary += "\n" + "-" * 40
                
                for i, building in enumerate(buildings_by_category[category][:10], 1):  # Show first 10
                    name = building.get('name', 'Unnamed Building')
                    if not name or name.lower() in ['unnamed building', '', 'nan', 'none']:
                        name = f"Building #{i}"
                    
                    summary += f"\n{i:2d}. {name}"
                    if building.get('amenity'):
                        summary += f" ({building['amenity']})"
                    summary += f" - {building['distance_km']:.2f}km"
                    if building.get('address'):
                        address_preview = building['address'][:50]
                        if len(building['address']) > 50:
                            address_preview += '...'
                        summary += f" - {address_preview}"
                
                if len(buildings_by_category[category]) > 10:
                    remaining = len(buildings_by_category[category]) - 10
                    summary += f"\n    ... and {remaining} more {category.lower()} buildings"
    else:
        summary += "\nNo buildings found in the specified radius."
    
    summary += f"""

🎯 RECOMMENDATIONS
{'=' * 30}
- Use the KML file in Google Earth for visual verification
- Import CSV file into Starlink coverage planning tools
- Review Community buildings for priority coverage
- Consider field verification for buildings without clear classification

📁 FILES GENERATED
{'=' * 30}
- CSV: {os.path.basename(output_file).replace('_summary.txt', '_buildings.csv')}
- KML: {os.path.basename(output_file).replace('_summary.txt', '_buildings.kml')}
- Summary: {os.path.basename(output_file)}

✅ Building extraction complete for {community_name}!
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(summary)

def generate_summary_report(results: List[Dict[str, Any]], output_file: str, cache_manager=None):
    """Generate overall batch summary report"""
    total_buildings = sum(len(r['buildings']) for r in results)
    successful_towns = [r for r in results if r['status'] == 'success']
    failed_towns = [r for r in results if r['status'] != 'success']
    cached_towns = [r for r in successful_towns if r.get('from_cache', False)]
    
    # Cache statistics
    cache_stats_text = ""
    if cache_manager:
        cache_stats = cache_manager.get_cache_stats()
        cache_stats_text = f"""

📦 CACHE PERFORMANCE
{'=' * 30}
OSM metadata hit rate: {cache_stats['osm_hit_rate']:.1f}%
Geocoding hit rate: {cache_stats['geocoding_hit_rate']:.1f}%
Building area hit rate: {cache_stats['building_hit_rate']:.1f}%
Total cached entries: {cache_stats['total_entries']}
Communities served from cache: {len(cached_towns)}"""
    
    report = f"""
🏗️  BATCH BUILDING EXTRACTION SUMMARY REPORT
{'=' * 60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 OVERALL STATISTICS
{'=' * 30}
Total communities processed: {len(results)}
Successful extractions: {len(successful_towns)}
Failed extractions: {len(failed_towns)}
Total buildings found: {total_buildings:,}
Communities served from cache: {len(cached_towns)}{cache_stats_text}

📋 COMMUNITY RESULTS
{'=' * 30}
"""
    
    for result in results:
        community = result['community_name']
        status = result['status']
        cache_indicator = " 📦" if result.get('from_cache', False) else ""
        
        if status == 'success':
            stats = result['stats']
            building_count = stats['count']
            types = stats['types']
            
            report += f"""
🏘️  {community}{cache_indicator}
   Status: ✅ SUCCESS
   Buildings found: {building_count:,}
   Categories: R:{types.get('Residential', 0)} C:{types.get('Community', 0)} M:{types.get('Commercial', 0)}
   Output: {result.get('output_dir', 'output/' + sanitize_filename(community))}"""
        else:
            report += f"""
🏘️  {community}
   Status: ❌ {status.upper()}"""
    
    # Add category summary
    if successful_towns:
        all_types = {}
        for result in successful_towns:
            for category, count in result['stats']['types'].items():
                all_types[category] = all_types.get(category, 0) + count
        
        report += f"""

📈 BUILDING CATEGORY SUMMARY
{'=' * 30}
Total buildings by category across all communities:"""
        
        for category, count in sorted(all_types.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_buildings) * 100
            report += f"\n   {category}: {count:,} ({percentage:.1f}%)"
    
    cache_expiry_days = cache_manager.expiry_days if cache_manager else 30
    
    report += f"""

🔧 NEXT STEPS
{'=' * 30}
1. Review individual community summaries in each folder
2. Import CSV files into coverage planning tools  
3. Validate results using KML files in Google Earth
4. Cross-reference with field surveys during site visits

📁 OUTPUT STRUCTURE
{'=' * 30}
output/
├── <community_name>/
│   ├── <community_name>_buildings.csv
│   ├── <community_name>_buildings.kml  
│   └── <community_name>_summary.txt
└── batch_summary_report.txt

💡 CACHE TIPS
{'=' * 30}
- Cache significantly speeds up re-runs
- Clear cache with --clear-cache if data seems outdated
- Cache expires automatically after {cache_expiry_days} days
- Disable cache with --no-cache for fresh data

✅ Enhanced building extraction with OSM metadata and caching complete!
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)