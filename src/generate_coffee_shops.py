#!/usr/bin/env python3
"""
London Coffee Shops Generator

This script generates coffee shop data for London with coffee types
extracted from the coffee_analysis.csv dataset.
"""

import pandas as pd
import json
import random
from typing import List, Dict, Any


def load_coffee_types():
    """Load coffee types from the dataset."""
    try:
        df = pd.read_csv("data/coffee_analysis.csv")
        print(f"📊 Loaded dataset with {len(df)} entries")

        # Extract coffee information
        coffee_types = []

        # Look for relevant columns
        name_cols = [col for col in df.columns if "name" in col.lower()]
        origin_cols = [
            col
            for col in df.columns
            if "origin" in col.lower() or "country" in col.lower()
        ]

        print(f"🔍 Found columns: name={name_cols}, origin={origin_cols}")

        for idx, row in df.iterrows():
            coffee_info = {}

            # Get coffee name
            for col in name_cols:
                if pd.notna(row[col]) and str(row[col]).strip():
                    coffee_info["name"] = str(row[col]).strip()
                    break

            # Get origin from origin_2 column if it exists, otherwise from any origin column
            origin_found = False
            if (
                "origin_2" in df.columns
                and pd.notna(row["origin_2"])
                and str(row["origin_2"]).strip()
            ):
                coffee_info["origin"] = str(row["origin_2"]).strip()
                origin_found = True
            else:
                for col in origin_cols:
                    if pd.notna(row[col]) and str(row[col]).strip():
                        coffee_info["origin"] = str(row[col]).strip()
                        origin_found = True
                        break

            # Only include if we have both name and origin
            if coffee_info.get("name") and origin_found:
                coffee_types.append(coffee_info)

        print(f"✅ Extracted {len(coffee_types)} coffee types")
        return coffee_types

    except FileNotFoundError:
        print("❌ coffee_analysis.csv not found")
        return []
    except Exception as e:
        print(f"❌ Error loading coffee types: {e}")
        return []


def generate_london_coffee_shops(coffee_types: List[Dict]) -> List[Dict]:
    """Generate London coffee shops with coffee types."""

    # London coffee shop locations (around 60, focusing on independent coffee shops)
    london_locations = [
        # Independent Coffee Shops
        {
            "name": "Monmouth Coffee Company",
            "address": "27 Monmouth Street, Covent Garden",
            "lat": 51.5129,
            "lng": -0.1270,
        },
        {
            "name": "Monmouth Coffee Company",
            "address": "2 Park Street, Borough Market",
            "lat": 51.5030,
            "lng": -0.0940,
        },
        {
            "name": "Monmouth Coffee Company",
            "address": "27 Maltby Street, Bermondsey",
            "lat": 51.4980,
            "lng": -0.0800,
        },
        {
            "name": "Prufrock Coffee",
            "address": "23-25 Leather Lane, Holborn",
            "lat": 51.5189,
            "lng": -0.1095,
        },
        {
            "name": "Workshop Coffee",
            "address": "27 Clerkenwell Road, Clerkenwell",
            "lat": 51.5220,
            "lng": -0.1060,
        },
        {
            "name": "Workshop Coffee",
            "address": "80A Mortimer Street, Fitzrovia",
            "lat": 51.5200,
            "lng": -0.1400,
        },
        {
            "name": "Workshop Coffee",
            "address": "75 Cowcross Street, Farringdon",
            "lat": 51.5220,
            "lng": -0.1060,
        },
        {
            "name": "Ozone Coffee Roasters",
            "address": "11 Leonard Street, Shoreditch",
            "lat": 51.5230,
            "lng": -0.0810,
        },
        {
            "name": "Caravan Coffee Roasters",
            "address": "11-13 Exmouth Market, Clerkenwell",
            "lat": 51.5230,
            "lng": -0.1060,
        },
        {
            "name": "Caravan Coffee Roasters",
            "address": "152-156 Clerkenwell Road, Clerkenwell",
            "lat": 51.5220,
            "lng": -0.1060,
        },
        {
            "name": "Allpress Espresso",
            "address": "58 Redchurch Street, Shoreditch",
            "lat": 51.5230,
            "lng": -0.0750,
        },
        {
            "name": "Allpress Espresso",
            "address": "55 Dalston Lane, Dalston",
            "lat": 51.5480,
            "lng": -0.0750,
        },
        {
            "name": "Kaffeine",
            "address": "66 Great Titchfield Street, Fitzrovia",
            "lat": 51.5200,
            "lng": -0.1400,
        },
        {
            "name": "Kaffeine",
            "address": "15 Eastcastle Street, Fitzrovia",
            "lat": 51.5200,
            "lng": -0.1400,
        },
        {
            "name": "Notes Coffee",
            "address": "31 St Martin's Lane, Covent Garden",
            "lat": 51.5120,
            "lng": -0.1280,
        },
        {
            "name": "Notes Coffee",
            "address": "36 Trafalgar Square, Charing Cross",
            "lat": 51.5070,
            "lng": -0.1280,
        },
        {
            "name": "Notes Coffee",
            "address": "1 New Street, Covent Garden",
            "lat": 51.5120,
            "lng": -0.1280,
        },
        {
            "name": "Flat White",
            "address": "17 Berwick Street, Soho",
            "lat": 51.5140,
            "lng": -0.1360,
        },
        {
            "name": "Flat White",
            "address": "25 Frith Street, Soho",
            "lat": 51.5140,
            "lng": -0.1360,
        },
        {
            "name": "The Coffee Works",
            "address": "40-42 Great Titchfield Street, Fitzrovia",
            "lat": 51.5200,
            "lng": -0.1400,
        },
        {
            "name": "Department of Coffee and Social Affairs",
            "address": "14-16 Leather Lane, Holborn",
            "lat": 51.5189,
            "lng": -0.1095,
        },
        {
            "name": "Department of Coffee and Social Affairs",
            "address": "15-17 Old Street, Shoreditch",
            "lat": 51.5230,
            "lng": -0.0810,
        },
        {
            "name": "TAP Coffee",
            "address": "193 Wardour Street, Soho",
            "lat": 51.5140,
            "lng": -0.1360,
        },
        {
            "name": "TAP Coffee",
            "address": "114 Tottenham Court Road, Fitzrovia",
            "lat": 51.5200,
            "lng": -0.1400,
        },
        {
            "name": "The Gentlemen Baristas",
            "address": "63 Union Street, Borough",
            "lat": 51.5030,
            "lng": -0.0940,
        },
        {
            "name": "The Gentlemen Baristas",
            "address": "44-46 Commercial Street, Spitalfields",
            "lat": 51.5200,
            "lng": -0.0750,
        },
        {
            "name": "WatchHouse Coffee",
            "address": "8-10 Bermondsey Street, Bermondsey",
            "lat": 51.4980,
            "lng": -0.0800,
        },
        {
            "name": "WatchHouse Coffee",
            "address": "125 Fenchurch Street, City",
            "lat": 51.5120,
            "lng": -0.0810,
        },
        {
            "name": "Origin Coffee Roasters",
            "address": "65 Shoreditch High Street, Shoreditch",
            "lat": 51.5230,
            "lng": -0.0810,
        },
        {
            "name": "Origin Coffee Roasters",
            "address": "40-42 Great Titchfield Street, Fitzrovia",
            "lat": 51.5200,
            "lng": -0.1400,
        },
        {
            "name": "Climpson & Sons",
            "address": "67 Broadway Market, Hackney",
            "lat": 51.5400,
            "lng": -0.0600,
        },
        {
            "name": "Climpson & Sons",
            "address": "Arch 374, Helmsley Place, Hackney",
            "lat": 51.5400,
            "lng": -0.0600,
        },
        {
            "name": "Nude Espresso",
            "address": "26 Hanbury Street, Spitalfields",
            "lat": 51.5200,
            "lng": -0.0750,
        },
        {
            "name": "Nude Espresso",
            "address": "Soho Square, Soho",
            "lat": 51.5140,
            "lng": -0.1360,
        },
        {
            "name": "The Roasting Party",
            "address": "1-3 Dray Walk, Brick Lane",
            "lat": 51.5200,
            "lng": -0.0750,
        },
        {
            "name": "The Roasting Party",
            "address": "146 Brick Lane, Shoreditch",
            "lat": 51.5200,
            "lng": -0.0750,
        },
        {
            "name": "Coffee Island",
            "address": "45-47 Old Street, Shoreditch",
            "lat": 51.5230,
            "lng": -0.0810,
        },
        {
            "name": "Coffee Island",
            "address": "123 Old Street, Shoreditch",
            "lat": 51.5230,
            "lng": -0.0810,
        },
        {
            "name": "Brew Lab",
            "address": "6-8 South College Street, Edinburgh",
            "lat": 55.9480,
            "lng": -3.1870,
        },
        {
            "name": "Artisan Coffee",
            "address": "123 Oxford Street, Fitzrovia",
            "lat": 51.5200,
            "lng": -0.1400,
        },
        {
            "name": "Brew & Bake",
            "address": "45 Camden High Street, Camden",
            "lat": 51.5400,
            "lng": -0.1400,
        },
        {
            "name": "Café Nero",
            "address": "78 Oxford Street, Fitzrovia",
            "lat": 51.5200,
            "lng": -0.1400,
        },
        {
            "name": "Café Nero",
            "address": "156 Regent Street, Soho",
            "lat": 51.5140,
            "lng": -0.1360,
        },
        {
            "name": "Café Nero",
            "address": "89 Tottenham Court Road, Fitzrovia",
            "lat": 51.5200,
            "lng": -0.1400,
        },
        {
            "name": "Costa Coffee",
            "address": "125 Oxford Street, Fitzrovia",
            "lat": 51.5200,
            "lng": -0.1400,
        },
        {
            "name": "Costa Coffee",
            "address": "67 Regent Street, Soho",
            "lat": 51.5140,
            "lng": -0.1360,
        },
        {
            "name": "Costa Coffee",
            "address": "234 Tottenham Court Road, Fitzrovia",
            "lat": 51.5200,
            "lng": -0.1400,
        },
        {
            "name": "Starbucks",
            "address": "156 Oxford Street, Fitzrovia",
            "lat": 51.5200,
            "lng": -0.1400,
        },
        {
            "name": "Starbucks",
            "address": "89 Regent Street, Soho",
            "lat": 51.5140,
            "lng": -0.1360,
        },
        {
            "name": "Starbucks",
            "address": "345 Tottenham Court Road, Fitzrovia",
            "lat": 51.5200,
            "lng": -0.1400,
        },
        {
            "name": "Pret A Manger",
            "address": "234 Oxford Street, Fitzrovia",
            "lat": 51.5200,
            "lng": -0.1400,
        },
        {
            "name": "Pret A Manger",
            "address": "123 Regent Street, Soho",
            "lat": 51.5140,
            "lng": -0.1360,
        },
        {
            "name": "Pret A Manger",
            "address": "456 Tottenham Court Road, Fitzrovia",
            "lat": 51.5200,
            "lng": -0.1400,
        },
        {
            "name": "Gail's Bakery",
            "address": "67 Oxford Street, Fitzrovia",
            "lat": 51.5200,
            "lng": -0.1400,
        },
        {
            "name": "Gail's Bakery",
            "address": "234 Regent Street, Soho",
            "lat": 51.5140,
            "lng": -0.1360,
        },
        {
            "name": "Paul",
            "address": "345 Oxford Street, Fitzrovia",
            "lat": 51.5200,
            "lng": -0.1400,
        },
        {
            "name": "Paul",
            "address": "456 Regent Street, Soho",
            "lat": 51.5140,
            "lng": -0.1360,
        },
        {
            "name": "Eat",
            "address": "567 Oxford Street, Fitzrovia",
            "lat": 51.5200,
            "lng": -0.1400,
        },
        {
            "name": "Eat",
            "address": "678 Regent Street, Soho",
            "lat": 51.5140,
            "lng": -0.1360,
        },
        {
            "name": "Itsu",
            "address": "789 Oxford Street, Fitzrovia",
            "lat": 51.5200,
            "lng": -0.1400,
        },
        {
            "name": "Itsu",
            "address": "890 Regent Street, Soho",
            "lat": 51.5140,
            "lng": -0.1360,
        },
        {
            "name": "Wasabi",
            "address": "901 Oxford Street, Fitzrovia",
            "lat": 51.5200,
            "lng": -0.1400,
        },
        {
            "name": "Wasabi",
            "address": "012 Regent Street, Soho",
            "lat": 51.5140,
            "lng": -0.1360,
        },
    ]

    coffee_shops = []

    for location in london_locations:
        # Randomly select 5-10 coffee types for each shop
        num_coffees = random.randint(5, 10)
        shop_coffees = random.sample(coffee_types, min(num_coffees, len(coffee_types)))

        coffee_shop = {
            "name": location["name"],
            "address": location["address"],
            "lat": location["lat"],
            "lng": location["lng"],
            "coffees": shop_coffees,
        }

        coffee_shops.append(coffee_shop)

    return coffee_shops


def create_web_app_data(coffee_shops: List[Dict]) -> Dict[str, Any]:
    """Create data structure for web application."""
    return {
        "coffeeShops": coffee_shops,
        "totalShops": len(coffee_shops),
        "totalCoffees": sum(len(shop["coffees"]) for shop in coffee_shops),
        "generatedAt": pd.Timestamp.now().isoformat(),
    }


def save_coffee_shops_data(
    data: Dict[str, Any], filename: str = "coffee_shops_data.json"
):
    """Save coffee shops data to file."""
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Coffee shops data saved to {filename}")


def create_html_snippet(coffee_shops: List[Dict]) -> str:
    """Create HTML snippet for the web app."""
    html_snippet = "const coffeeShops = [\n"

    for shop in coffee_shops:
        html_snippet += f"    {{\n"
        html_snippet += f"        name: \"{shop['name']}\",\n"
        html_snippet += f"        address: \"{shop['address']}\",\n"
        html_snippet += f"        lat: {shop['lat']},\n"
        html_snippet += f"        lng: {shop['lng']},\n"
        html_snippet += f"        coffees: [\n"

        for coffee in shop["coffees"]:
            html_snippet += f"            {{\n"
            html_snippet += (
                f"                name: \"{coffee.get('name', 'Unknown')}\",\n"
            )
            html_snippet += (
                f"                origin: \"{coffee.get('origin', 'Unknown')}\",\n"
            )
            html_snippet += f"            }},\n"

        html_snippet += f"        ]\n"
        html_snippet += f"    }},\n"

    html_snippet += "];"

    return html_snippet


def main():
    """Main function to generate coffee shops data."""
    print("☕ London Coffee Shops Generator")
    print("=" * 50)

    # Load coffee types from dataset
    coffee_types = load_coffee_types()

    if not coffee_types:
        print("❌ No coffee types found. Using sample data.")
        coffee_types = [
            {
                "name": "Ethiopian Yirgacheffe",
                "origin": "Ethiopia",
            },
            {
                "name": "Colombian Supremo",
                "origin": "Colombia",
            },
            {
                "name": "Brazilian Santos",
                "origin": "Brazil",
            },
            {
                "name": "Guatemalan Antigua",
                "origin": "Guatemala",
            },
            {
                "name": "Costa Rican Tarrazu",
                "origin": "Costa Rica",
            },
        ]

    # Generate coffee shops
    coffee_shops = generate_london_coffee_shops(coffee_types)

    # Create data structure
    data = create_web_app_data(coffee_shops)

    # Save data
    save_coffee_shops_data(data)

    # Create HTML snippet
    html_snippet = create_html_snippet(coffee_shops)

    # Save HTML snippet
    with open("coffee_shops_html.js", "w") as f:
        f.write(html_snippet)

    print("✅ HTML snippet saved to coffee_shops_html.js")

    # Print summary
    print("\n📊 Summary:")
    print(f"  - Coffee shops generated: {len(coffee_shops)}")
    print(f"  - Total coffees: {data['totalCoffees']}")
    print(
        f"  - Average coffees per shop: "
        f"{data['totalCoffees']/len(coffee_shops):.1f}"
    )

    # Show sample
    print("\n🏪 Sample coffee shop:")
    sample_shop = coffee_shops[0]
    print(f"  Name: {sample_shop['name']}")
    print(f"  Address: {sample_shop['address']}")
    print(f"  Coffees: {len(sample_shop['coffees'])} types")
    for coffee in sample_shop["coffees"][:3]:
        print(f"    - {coffee['name']} ({coffee['origin']})")


if __name__ == "__main__":
    main()
