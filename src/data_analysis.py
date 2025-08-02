#!/usr/bin/env python3
"""
Data Analysis Script for Coffee Dataset

This script analyzes the coffee_analysis.csv dataset to understand its structure
and extract coffee information for the web application.
"""

import pandas as pd
import json
from pathlib import Path


def analyze_coffee_dataset():
    """Analyze the coffee dataset structure."""
    try:
        # Load the dataset
        df = pd.read_csv("data/coffee_analysis.csv")

        print("📊 Coffee Dataset Analysis")
        print("=" * 50)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Display first few rows
        print("\nFirst 5 rows:")
        print(df.head())

        # Check for missing values
        print("\nMissing values:")
        print(df.isnull().sum())

        # Analyze unique values in key columns
        print("\nUnique values in key columns:")
        for col in df.columns:
            if df[col].dtype == "object":  # String columns
                unique_count = df[col].nunique()
                print(f"{col}: {unique_count} unique values")
                if unique_count < 20:  # Show if not too many
                    print(f"  Values: {df[col].unique()}")

        # Check the last column (taste/flavors)
        last_col = df.columns[-1]
        print(f"\nLast column '{last_col}' sample values:")
        print(df[last_col].head(10).tolist())

        return df

    except FileNotFoundError:
        print("❌ coffee_analysis.csv not found in data/ directory")
        return None
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")
        return None


def extract_coffee_types(df):
    """Extract coffee types from the dataset."""
    if df is None:
        return []

    # Get unique coffee names and origins
    coffee_data = []

    # Look for columns that might contain coffee names
    name_columns = [
        col for col in df.columns if "name" in col.lower() or "brand" in col.lower()
    ]
    origin_columns = [
        col for col in df.columns if "origin" in col.lower() or "country" in col.lower()
    ]

    print(f"\n🔍 Found name columns: {name_columns}")
    print(f"🔍 Found origin columns: {origin_columns}")

    # Extract coffee information
    for idx, row in df.iterrows():
        coffee_info = {}

        # Get coffee name
        for col in name_columns:
            if pd.notna(row[col]) and str(row[col]).strip():
                coffee_info["name"] = str(row[col]).strip()
                break

        # Get origin/country
        for col in origin_columns:
            if pd.notna(row[col]) and str(row[col]).strip():
                coffee_info["origin"] = str(row[col]).strip()
                break

        # Get rating if available
        rating_cols = [col for col in df.columns if "rating" in col.lower()]
        if rating_cols and pd.notna(row[rating_cols[0]]):
            coffee_info["rating"] = float(row[rating_cols[0]])

        # Get roast level if available
        roast_cols = [col for col in df.columns if "roast" in col.lower()]
        if roast_cols and pd.notna(row[roast_cols[0]]):
            coffee_info["roast_level"] = str(row[roast_cols[0]]).strip()

        # Get flavors from last column
        last_col = df.columns[-1]
        if pd.notna(row[last_col]) and str(row[last_col]).strip():
            coffee_info["flavors"] = str(row[last_col]).strip()

        if coffee_info:
            coffee_data.append(coffee_info)

    print(f"\n📈 Extracted {len(coffee_data)} coffee entries")

    # Show sample of extracted data
    print("\nSample extracted coffee data:")
    for i, coffee in enumerate(coffee_data[:5]):
        print(f"  {i+1}. {coffee}")

    return coffee_data


def main():
    """Main analysis function."""
    print("☕ Coffee Dataset Analysis")
    print("=" * 50)

    # Analyze the dataset
    df = analyze_coffee_dataset()

    if df is not None:
        # Extract coffee types
        coffee_types = extract_coffee_types(df)

        # Save extracted data
        with open("data/extracted_coffee_types.json", "w") as f:
            json.dump(coffee_types, f, indent=2)

        print(f"\n✅ Extracted coffee data saved to data/extracted_coffee_types.json")
        print(f"📊 Total coffee types extracted: {len(coffee_types)}")


if __name__ == "__main__":
    main()
