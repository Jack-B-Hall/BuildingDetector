#!/usr/bin/env python3
"""
Enhanced Building Coordinate Extractor with OSM Metadata and Caching (Modular Version)
Process multiple towns from CSV with OpenStreetMap enrichment and intelligent caching

Usage:
  # Batch mode (reads TownData.csv by default)
  python town_extract.py --batch
  python town_extract.py --batch --input towns.csv --default-distance 5
  
  # Single town mode
  python town_extract.py --lat -16.95 --lon 122.86 --distance 2 --name "Beagle Bay"
  
  # Cache management
  python town_extract.py --clear-cache
  python town_extract.py --batch --no-cache

Expected CSV format (TownData.csv):
  AGIL CODE,Community Name,Latitude,Longitude,State,LGA,ABS Remoteness
  Or with distance_km column:
  Community Name,Latitude,Longitude,distance_km,State,LGA,ABS Remoteness
"""

import argparse
import os
import pandas as pd
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import our modular components
from app.utils import vprint, sanitize_filename, get_quadkeys_for_area
from app.cache_manager import CacheManager
from app.microsoft_data import (
    load_australia_dataset_links, 
    find_available_quadkeys, 
    download_building_files, 
    process_building_file,
    cleanup_temp_files
)
from app.output_generators import (
    save_buildings_to_csv,
    save_buildings_to_kml, 
    generate_community_summary,
    generate_summary_report
)

# Global settings
DEFAULT_DISTANCE = 2.0  # Default 2km radius
CACHE_DIR = "cache"  # Cache directory
CACHE_EXPIRY_DAYS = 30  # Cache expires after 30 days

# Global cache manager
cache_manager: Optional[CacheManager] = None

def load_towns_from_csv(csv_file: str, default_distance: float) -> pd.DataFrame:
    """Load towns from CSV and handle different column formats"""
    try:
        df = pd.read_csv(csv_file)
        
        vprint(f"Available columns in CSV: {list(df.columns)}")
        
        required_mapping = {}
        
        # Map community name column
        if 'Community Name' in df.columns:
            required_mapping['community_name'] = 'Community Name'
        elif 'town_name' in df.columns:
            required_mapping['community_name'] = 'town_name'
        elif 'name' in df.columns:
            required_mapping['community_name'] = 'name'
        else:
            raise ValueError("Could not find community name column. Expected 'Community Name', 'town_name', or 'name'")
        
        # Map latitude column
        if 'Latitude' in df.columns:
            required_mapping['latitude'] = 'Latitude'
        elif 'latitude' in df.columns:
            required_mapping['latitude'] = 'latitude'
        elif 'lat' in df.columns:
            required_mapping['latitude'] = 'lat'
        else:
            raise ValueError("Could not find latitude column. Expected 'Latitude', 'latitude', or 'lat'")
        
        # Map longitude column
        if 'Longitude' in df.columns:
            required_mapping['longitude'] = 'Longitude'
        elif 'longitude' in df.columns:
            required_mapping['longitude'] = 'longitude'
        elif 'lon' in df.columns:
            required_mapping['longitude'] = 'lon'
        else:
            raise ValueError("Could not find longitude column. Expected 'Longitude', 'longitude', or 'lon'")
        
        # Map distance column (optional)
        if 'distance_km' in df.columns:
            required_mapping['distance_km'] = 'distance_km'
        elif 'Distance' in df.columns:
            required_mapping['distance_km'] = 'Distance'
        elif 'radius' in df.columns:
            required_mapping['distance_km'] = 'radius'
        else:
            vprint(f"No distance column found, using default distance: {default_distance}km")
            df['distance_km'] = default_distance
            required_mapping['distance_km'] = 'distance_km'
        
        # Create standardized dataframe
        standardized_df = pd.DataFrame()
        for standard_name, original_name in required_mapping.items():
            standardized_df[standard_name] = df[original_name]
        
        # Add additional columns if they exist
        additional_cols = ['State', 'LGA', 'ABS Remoteness', 'AGIL CODE']
        for col in additional_cols:
            if col in df.columns:
                standardized_df[col] = df[col]
        
        vprint(f"Successfully loaded {len(standardized_df)} communities from {csv_file}")
        return standardized_df
        
    except Exception as e:
        raise ValueError(f"Error loading CSV file {csv_file}: {e}")

def process_single_town(community_name: str, lat: float, lon: float, distance: float, 
                       australia_links: pd.DataFrame, output_dir: str, zoom: int = 9) -> Dict[str, Any]:
    """Process a single community and return results"""
    global cache_manager
    
    vprint(f"🏘️  Processing {community_name}...")
    vprint(f"    📍 Location: ({lat:.6f}, {lon:.6f})")
    vprint(f"    📏 Search radius: {distance}km")
    
    # Check if we have cached results for this area
    if cache_manager:
        cached_buildings = cache_manager.get_building_area(lat, lon, distance)
        if cached_buildings is not None:
            vprint(f"    📦 Using cached building data ({len(cached_buildings)} buildings)")
            
            # Create community-specific output directory
            safe_name = sanitize_filename(community_name)
            community_output_dir = os.path.join(output_dir, safe_name)
            Path(community_output_dir).mkdir(exist_ok=True)
            
            # Save files directly from cache
            csv_file = os.path.join(community_output_dir, f"{safe_name}_buildings.csv")
            kml_file = os.path.join(community_output_dir, f"{safe_name}_buildings.kml")
            summary_file = os.path.join(community_output_dir, f"{safe_name}_summary.txt")
            
            stats = save_buildings_to_csv(cached_buildings, csv_file)
            save_buildings_to_kml(cached_buildings, kml_file, lat, lon, distance, community_name)
            generate_community_summary(cached_buildings, stats, summary_file, community_name, lat, lon, distance)
            
            vprint(f"    ✅ {community_name} completed from cache: {len(cached_buildings)} buildings")
            
            return {
                'community_name': community_name,
                'status': 'success',
                'buildings': cached_buildings,
                'stats': stats,
                'output_dir': community_output_dir,
                'files': {
                    'csv': csv_file,
                    'kml': kml_file,
                    'summary': summary_file
                },
                'from_cache': True
            }
    
    # Create community-specific output directory
    safe_name = sanitize_filename(community_name)
    community_output_dir = os.path.join(output_dir, safe_name)
    Path(community_output_dir).mkdir(exist_ok=True)
    
    try:
        # Step 1: Get QuadKeys
        target_quadkeys = get_quadkeys_for_area(lat, lon, distance, zoom)
        vprint(f"    🗺️  Generated {len(target_quadkeys)} QuadKeys")
        
        # Step 2: Find available data
        available_quadkeys = find_available_quadkeys(target_quadkeys, australia_links)
        if not available_quadkeys:
            vprint(f"    ❌ No building data available for {community_name}", "WARNING")
            return {
                'community_name': community_name,
                'status': 'no_data',
                'buildings': [],
                'stats': {'count': 0, 'types': {}, 'distance_range': (0, 0), 'height_stats': None}
            }
        
        vprint(f"    📦 Found {len(available_quadkeys)} data files")
        
        # Step 3: Download data
        download_dir = "temp_buildings"
        downloaded_files = download_building_files(available_quadkeys, australia_links, download_dir)
        
        if not downloaded_files:
            vprint(f"    ❌ Failed to download data for {community_name}", "ERROR")
            return {
                'community_name': community_name,
                'status': 'download_failed',
                'buildings': [],
                'stats': {'count': 0, 'types': {}, 'distance_range': (0, 0), 'height_stats': None}
            }
        
        # Step 4: Process buildings with OSM metadata
        vprint(f"    🏠 Processing buildings with OSM metadata...")
        all_buildings = []
        for filepath in downloaded_files:
            buildings = process_building_file(filepath, lat, lon, distance, cache_manager)
            all_buildings.extend(buildings)
        
        vprint(f"    🏠 Found {len(all_buildings)} buildings with metadata")
        
        # Cache the results for this area
        if cache_manager and all_buildings:
            cache_manager.save_building_area(lat, lon, distance, all_buildings)
        
        # Step 5: Save files in community directory
        csv_file = os.path.join(community_output_dir, f"{safe_name}_buildings.csv")
        kml_file = os.path.join(community_output_dir, f"{safe_name}_buildings.kml")
        summary_file = os.path.join(community_output_dir, f"{safe_name}_summary.txt")
        
        stats = save_buildings_to_csv(all_buildings, csv_file)
        save_buildings_to_kml(all_buildings, kml_file, lat, lon, distance, community_name)
        generate_community_summary(all_buildings, stats, summary_file, community_name, lat, lon, distance)
        
        vprint(f"    ✅ {community_name} completed: {len(all_buildings)} buildings")
        vprint(f"    📁 Files saved to: {community_output_dir}/")
        
        return {
            'community_name': community_name,
            'status': 'success',
            'buildings': all_buildings,
            'stats': stats,
            'output_dir': community_output_dir,
            'files': {
                'csv': csv_file,
                'kml': kml_file,
                'summary': summary_file
            },
            'from_cache': False
        }
        
    except Exception as e:
        vprint(f"    ❌ Error processing {community_name}: {e}", "ERROR")
        return {
            'community_name': community_name,
            'status': 'error',
            'error': str(e),
            'buildings': [],
            'stats': {'count': 0, 'types': {}, 'distance_range': (0, 0), 'height_stats': None}
        }

def main():
    global cache_manager
    
    # Update utils module VERBOSE setting based on arguments
    import app.utils
    
    parser = argparse.ArgumentParser(
        description="Extract building coordinates with OSM metadata and intelligent caching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Batch mode (processes TownData.csv with 2km radius)
  python batch_building_extractor.py --batch
  
  # Custom input file with different default distance
  python batch_building_extractor.py --batch --input my_towns.csv --default-distance 5
  
  # Single community mode
  python batch_building_extractor.py --lat -12.324 --lon 133.056 --distance 2 --name "Gunbalanya"
  
  # Cache management
  python batch_building_extractor.py --clear-cache
  python batch_building_extractor.py --batch --no-cache
  
Expected CSV format:
  Community Name,Latitude,Longitude[,distance_km]
  AGIL CODE,Community Name,Latitude,Longitude,State,LGA,ABS Remoteness
        """
    )
    
    # Mode selection
    parser.add_argument('--batch', action='store_true',
                        help='Batch mode: process communities from CSV file')
    
    # Batch mode arguments
    parser.add_argument('--input', type=str, default='TownData.csv',
                        help='Input CSV file for batch mode (default: TownData.csv)')
    parser.add_argument('--default-distance', type=float, default=DEFAULT_DISTANCE,
                        help=f'Default search radius in km if not specified in CSV (default: {DEFAULT_DISTANCE})')
    
    # Single community mode arguments
    parser.add_argument('--lat', type=float,
                        help='Latitude of center point (single community mode)')
    parser.add_argument('--lon', type=float,
                        help='Longitude of center point (single community mode)')
    parser.add_argument('--distance', type=float,
                        help='Search radius in kilometers (single community mode)')
    parser.add_argument('--name', type=str,
                        help='Community name (single community mode)')
    
    # Cache arguments
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable caching (fetch fresh data)')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Clear all cached data and exit')
    parser.add_argument('--cache-dir', type=str, default=CACHE_DIR,
                        help=f'Cache directory (default: {CACHE_DIR})')
    
    # Common arguments
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory (default: output)')
    parser.add_argument('--zoom', type=int, default=9,
                        help='QuadKey zoom level for tile selection (default: 9)')
    parser.add_argument('--quiet', action='store_true',
                        help='Disable verbose logging')
    parser.add_argument('--keep-temp', action='store_true',
                        help='Keep temporary downloaded files (for debugging)')
    
    args = parser.parse_args()
    
    # Set verbose mode in utils module
    if args.quiet:
        utils.VERBOSE = False
    
    # Handle cache clearing
    if args.clear_cache:
        temp_cache = CacheManager(args.cache_dir)
        temp_cache.clear_cache()
        print("✅ Cache cleared successfully!")
        sys.exit(0)
    
    # Initialize cache manager
    if not args.no_cache:
        cache_manager = CacheManager(args.cache_dir)
        vprint(f"📦 Cache enabled: {args.cache_dir}/")
    else:
        vprint("🚫 Cache disabled - fetching fresh data")
    
    # Validate arguments
    if not args.batch and not all([args.lat, args.lon, args.distance, args.name]):
        print("❌ Error: Either use --batch mode or provide --lat, --lon, --distance, and --name")
        sys.exit(1)
    
    # Create output directory
    output_dir = args.output_dir
    Path(output_dir).mkdir(exist_ok=True)
    
    print("🏗️  Enhanced Building Coordinate Extractor with OSM Metadata & Caching")
    print("=" * 75)
    
    # Check for OSM libraries
    try:
        import osmnx as ox
        import overpy
        print("✅ OSM libraries available for metadata enrichment")
    except ImportError:
        print("⚠️  OSM libraries not available - metadata collection will be limited")
        print("   Install with: pip install osmnx overpy")
    
    print()
    
    try:
        # Load Australia dataset once
        vprint("📡 Loading Microsoft Australia building dataset...")
        australia_links = load_australia_dataset_links()
        vprint(f"✅ Loaded {len(australia_links)} Australian dataset files")
        
        results = []
        
        if args.batch:
            # Batch mode: process communities from CSV
            vprint(f"📋 Reading communities from: {args.input}")
            
            if not os.path.exists(args.input):
                print(f"❌ Input file not found: {args.input}")
                print("💡 Create a CSV file with columns: Community Name,Latitude,Longitude[,distance_km]")
                sys.exit(1)
            
            try:
                communities_df = load_towns_from_csv(args.input, args.default_distance)
            except ValueError as e:
                print(f"❌ Error loading CSV: {e}")
                sys.exit(1)
            
            vprint(f"📊 Found {len(communities_df)} communities to process")
            
            for _, row in communities_df.iterrows():
                result = process_single_town(
                    row['community_name'], 
                    row['latitude'], 
                    row['longitude'], 
                    row['distance_km'],
                    australia_links,
                    output_dir,
                    args.zoom
                )
                results.append(result)
        
        else:
            # Single community mode
            result = process_single_town(
                args.name,
                args.lat,
                args.lon,
                args.distance,
                australia_links,
                output_dir,
                args.zoom
            )
            results.append(result)
        
        # Save cache before generating reports
        if cache_manager:
            cache_manager.save_all_caches()
        
        # Generate batch summary report
        summary_file = os.path.join(output_dir, 'batch_summary_report.txt')
        generate_summary_report(results, summary_file, cache_manager)
        
        # Cleanup
        if not args.keep_temp:
            cleanup_temp_files("temp_buildings")
        
        print(f"\n🎉 Processing complete!")
        print(f"📁 Output directory: {output_dir}/")
        print(f"📊 Batch summary: {summary_file}")
        
        # Show cache statistics
        if cache_manager:
            cache_stats = cache_manager.get_cache_stats()
            print(f"📦 Cache performance: OSM {cache_stats['osm_hit_rate']:.1f}%, "
                  f"Geocoding {cache_stats['geocoding_hit_rate']:.1f}%, "
                  f"Areas {cache_stats['building_hit_rate']:.1f}%")
        
        # Show successful extractions
        successful = [r for r in results if r['status'] == 'success']
        if successful:
            print(f"\n✅ Successfully processed {len(successful)} communities:")
            for result in successful:
                cache_indicator = " 📦" if result.get('from_cache', False) else ""
                print(f"   📁 {result['community_name']}{cache_indicator}: {result['output_dir']}/")
        
    except KeyboardInterrupt:
        if cache_manager:
            cache_manager.save_all_caches()
        vprint("\n❌ Process interrupted by user", "ERROR")
        sys.exit(1)
    except Exception as e:
        vprint(f"\n❌ Fatal error: {e}", "ERROR")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()