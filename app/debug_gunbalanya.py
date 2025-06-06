#!/usr/bin/env python3
"""
Debug script to investigate why certain communities return few buildings
"""

import osmnx as ox
import pandas as pd
import requests
import json
from data_extractor import BuildingExtractor

def debug_community(name: str, lat: float, lon: float, radius: int = 1500):
    """Debug building extraction for a specific community"""
    
    print(f"\n🔍 DEBUGGING: {name}")
    print(f"📍 Location: {lat}, {lon}")
    print(f"📏 Radius: {radius}m")
    print("=" * 60)
    
    # Test 1: Direct OSMnx query
    print("\n1️⃣ Testing OSMnx direct query...")
    try:
        buildings = ox.features_from_point(
            (lat, lon),
            tags={'building': True},
            dist=radius
        )
        print(f"   ✅ OSMnx found {len(buildings)} buildings")
        if len(buildings) > 0:
            print(f"   Building types: {buildings['building'].value_counts().to_dict()}")
    except Exception as e:
        print(f"   ❌ OSMnx failed: {e}")
    
    # Test 2: Try with larger radius
    print(f"\n2️⃣ Testing with larger radius ({radius * 2}m)...")
    try:
        buildings_large = ox.features_from_point(
            (lat, lon),
            tags={'building': True},
            dist=radius * 2
        )
        print(f"   ✅ Found {len(buildings_large)} buildings with larger radius")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 3: Place name search
    print(f"\n3️⃣ Testing place name search...")
    place_queries = [
        f"{name}, Northern Territory, Australia",
        f"{name}, NT, Australia",
        f"{name}, Australia"
    ]
    
    for query in place_queries:
        try:
            place_buildings = ox.features_from_place(
                query,
                tags={'building': True}
            )
            print(f"   ✅ Query '{query}' found {len(place_buildings)} buildings")
            break
        except:
            print(f"   ❌ Query '{query}' failed")
    
    # Test 4: Raw Overpass API
    print(f"\n4️⃣ Testing raw Overpass API...")
    try:
        # Calculate bounding box
        lat_offset = radius / 111320
        lon_offset = radius / (111320 * abs(pd.np.cos(pd.np.radians(lat))))
        
        bbox = f"{lat - lat_offset},{lon - lon_offset},{lat + lat_offset},{lon + lon_offset}"
        
        overpass_query = f"""
        [out:json][timeout:30];
        (
          way["building"]({bbox});
          relation["building"]({bbox});
        );
        out body;
        >;
        out skel qt;
        """
        
        response = requests.post(
            "http://overpass-api.de/api/interpreter",
            data=overpass_query,
            timeout=35
        )
        
        if response.status_code == 200:
            data = response.json()
            elements = [e for e in data.get('elements', []) if e.get('type') == 'way']
            print(f"   ✅ Overpass API found {len(elements)} building ways")
        else:
            print(f"   ❌ Overpass API returned status {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Overpass API failed: {e}")
    
    # Test 5: Check what's actually in the area
    print(f"\n5️⃣ Checking all features in area...")
    try:
        all_features = ox.features_from_point(
            (lat, lon),
            tags={'building': True, 'landuse': True, 'amenity': True},
            dist=radius
        )
        print(f"   ✅ Total features found: {len(all_features)}")
        
        # Group by type
        if 'building' in all_features.columns:
            building_features = all_features[all_features['building'].notna()]
            print(f"   🏠 Features with building tag: {len(building_features)}")
        
        if 'landuse' in all_features.columns:
            landuse_features = all_features[all_features['landuse'].notna()]
            print(f"   🌍 Features with landuse tag: {len(landuse_features)}")
            
        if 'amenity' in all_features.columns:
            amenity_features = all_features[all_features['amenity'].notna()]
            print(f"   🏛️ Features with amenity tag: {len(amenity_features)}")
            
    except Exception as e:
        print(f"   ❌ Failed to get features: {e}")
    
    # Test 6: Using the BuildingExtractor class
    print(f"\n6️⃣ Testing BuildingExtractor class...")
    try:
        extractor = BuildingExtractor(default_radius=radius)
        buildings_df = extractor.extract_buildings(name, lat, lon, radius)
        
        if buildings_df is not None:
            print(f"   ✅ BuildingExtractor found {len(buildings_df)} buildings")
        else:
            print(f"   ❌ BuildingExtractor returned None")
            
    except Exception as e:
        print(f"   ❌ BuildingExtractor failed: {e}")
    
    print("\n" + "=" * 60)
    print("💡 RECOMMENDATIONS:")
    
    if 'buildings' in locals() and len(buildings) < 5:
        print("- Very few buildings found in OSM data")
        print("- Consider increasing search radius")
        print("- The area may lack detailed OSM mapping")
        print("- Try alternative data sources or manual verification")
    
    print("\n")


def main():
    """Main debug function"""
    print("🔧 BUILDING EXTRACTION DEBUGGER")
    print("=" * 60)
    
    # Test cases - communities with known issues
    test_communities = [
        ("Gunbalanya", -12.324, 133.056),
        # Add more problem communities here
    ]
    
    for name, lat, lon in test_communities:
        debug_community(name, lat, lon)
        
        # Also try with larger radius
        print(f"\n🔄 Retrying {name} with 3000m radius...")
        debug_community(name, lat, lon, radius=3000)


if __name__ == "__main__":
    # Configure OSMnx for debugging
    ox.settings.use_cache = False  # Disable cache for fresh data
    ox.settings.log_console = True  # Enable console logging
    
    main()