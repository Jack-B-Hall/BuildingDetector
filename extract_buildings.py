#!/usr/bin/env python3
"""
Building Extraction CLI
Main command-line interface for the community building extractor
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Import the main extractor
from app import CommunityBuildingExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('building_extraction.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description='Enhanced Community Building Extractor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_buildings.py                                    # Use default TownData.csv
  python extract_buildings.py --csv MyTowns.csv                  # Use custom CSV
  python extract_buildings.py --radius 2000 --no-geocoding      # 2km radius, no addresses
  python extract_buildings.py --force-refresh                    # Ignore cache, fresh data
  python extract_buildings.py --csv QuickTest.csv --radius 1000 # Quick test with small dataset
        """
    )
    
    parser.add_argument('--csv', '-c', default='TownData.csv',
                        help='CSV file with community data (default: TownData.csv)')
    parser.add_argument('--output', '-o', default='output',
                        help='Output directory (default: output)')
    parser.add_argument('--cache', default='cache',
                        help='Cache directory (default: cache)')
    parser.add_argument('--radius', '-r', type=int, default=1500,
                        help='Search radius in meters (default: 1500)')
    parser.add_argument('--delay', '-d', type=float, default=2.0,
                        help='Delay between API requests in seconds (default: 2.0)')
    parser.add_argument('--no-geocoding', action='store_true',
                        help='Disable address lookup (faster processing)')
    parser.add_argument('--cache-expiry', type=int, default=30,
                        help='Cache expiry in days (default: 30)')
    parser.add_argument('--force-refresh', action='store_true',
                        help='Force refresh of cached data')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: smaller radius, no geocoding, faster processing')
    
    return parser


def main():
    """Main function with command line interface"""
    parser = create_parser()
    args = parser.parse_args()
    
    print("🏘️ ENHANCED COMMUNITY BUILDING EXTRACTOR")
    print("=" * 60)
    print("✨ Features:")
    print("   🔧 Custom CSV input support")
    print("   💾 Smart caching system")
    print("   🏠 Address lookup with reverse geocoding")
    print("   ⚡ Modular architecture for better reliability")
    print("=" * 60)
    
    # Quick mode adjustments
    if args.quick:
        args.radius = min(args.radius, 1000)  # Max 1km in quick mode
        args.no_geocoding = True
        args.delay = 1.0
        print("⚡ QUICK MODE: Smaller radius, no geocoding, faster processing")
    
    # Check if CSV file exists
    if not os.path.exists(args.csv):
        print(f"❌ Error: CSV file not found: {args.csv}")
        print(f"💡 Please ensure {args.csv} exists in the current directory")
        print(f"📋 Expected columns: Community Name, Latitude, Longitude")
        return
    
    try:
        # Initialize extractor with command line arguments
        extractor = CommunityBuildingExtractor(
            csv_file=args.csv,
            output_dir=args.output,
            cache_dir=args.cache,
            default_radius=args.radius,
            delay_between_requests=args.delay,
            enable_geocoding=not args.no_geocoding,
            cache_expiry_days=args.cache_expiry,
            force_refresh=args.force_refresh
        )
        
        # Process all communities
        results = extractor.process_all_communities()
        
        # Print final statistics
        print(f"\n🎉 BATCH EXTRACTION COMPLETE!")
        print("=" * 40)
        stats = results['statistics']
        print(f"📊 Total communities processed: {stats['total_communities']}")
        print(f"✅ Successful extractions: {stats['successful_extractions']}")
        print(f"❌ Failed extractions: {stats['failed_extractions']}")
        print(f"🏠 Total buildings found: {stats['total_buildings_found']}")
        print(f"🏘️ Communities with buildings: {stats['communities_with_buildings']}")
        print(f"🔍 Communities without buildings: {stats['communities_without_buildings']}")
        
        # Cache statistics
        if 'building_cache_hit_rate' in stats:
            print(f"💾 Cache hit rate: {stats['building_cache_hit_rate']:.1f}%")
        
        # Classification statistics
        if 'total_classified' in stats:
            print(f"\n📊 Building Classification:")
            print(f"   Residential: {stats.get('residential', 0)}")
            print(f"   Community: {stats.get('community', 0)}")
            print(f"   Commercial: {stats.get('commercial', 0)}")
            if stats.get('unknown_defaulted', 0) > 0:
                print(f"   Unknown (defaulted to Residential): {stats['unknown_defaulted']}")
        
        # Geocoding statistics
        if not args.no_geocoding and 'success_rate' in stats:
            print(f"\n🏠 Address Lookup:")
            print(f"   Success rate: {stats['success_rate']:.1f}%")
            print(f"   Total geocoded: {stats.get('total_geocoded', 0)}")
        
        print(f"\n📁 Results saved in: {Path(args.output).absolute()}")
        print(f"💾 Cache saved in: {Path(args.cache).absolute()}")
        print("📋 Each community has its own folder with CSV, KML, and summary files")
        
    except Exception as e:
        logger.error(f"❌ Batch extraction failed: {e}")
        raise


if __name__ == "__main__":
    main()