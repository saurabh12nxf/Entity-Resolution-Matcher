#!/usr/bin/env python3
"""
Generate synthetic clean and messy data for entity resolution matching.
Produces 150 clean records and 180 messy records (150 corrupted + 30 true negatives).
"""

import pandas as pd
import random
import re

# Set seed for reproducible results
random.seed(42)

# Nickname dictionary for realistic name corruptions
NICKNAMES = {
    'Robert': 'Bob', 'William': 'Bill', 'Richard': 'Rick', 'James': 'Jim',
    'John': 'Jack', 'Michael': 'Mike', 'David': 'Dave', 'Christopher': 'Chris',
    'Matthew': 'Matt', 'Anthony': 'Tony', 'Donald': 'Don', 'Steven': 'Steve',
    'Andrew': 'Andy', 'Joshua': 'Josh', 'Daniel': 'Dan', 'Elizabeth': 'Liz',
    'Jennifer': 'Jen', 'Jessica': 'Jess', 'Sarah': 'Sally', 'Susan': 'Sue',
    'Michelle': 'Shelly', 'Kimberly': 'Kim', 'Patricia': 'Pat', 'Linda': 'Lynn',
    'Barbara': 'Barb', 'Margaret': 'Maggie', 'Dorothy': 'Dot', 'Lisa': 'Lee'
}

# Common first and last names for generating realistic records
FIRST_NAMES = [
    'James', 'Robert', 'John', 'Michael', 'David', 'William', 'Richard', 'Thomas',
    'Christopher', 'Charles', 'Daniel', 'Matthew', 'Anthony', 'Mark', 'Donald',
    'Steven', 'Paul', 'Andrew', 'Joshua', 'Kenneth', 'Kevin', 'Brian', 'George',
    'Timothy', 'Ronald', 'Jason', 'Edward', 'Jeffrey', 'Ryan', 'Jacob', 'Gary',
    'Nicholas', 'Eric', 'Jonathan', 'Stephen', 'Larry', 'Justin', 'Scott', 'Brandon',
    'Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara', 'Susan',
    'Jessica', 'Sarah', 'Karen', 'Nancy', 'Lisa', 'Betty', 'Helen', 'Sandra',
    'Donna', 'Carol', 'Ruth', 'Sharon', 'Michelle', 'Laura', 'Sarah', 'Kimberly',
    'Deborah', 'Dorothy', 'Lisa', 'Nancy', 'Karen', 'Betty', 'Helen'
]

LAST_NAMES = [
    'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
    'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson',
    'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee', 'Perez', 'Thompson',
    'White', 'Harris', 'Sanchez', 'Clark', 'Ramirez', 'Lewis', 'Robinson', 'Walker',
    'Young', 'Allen', 'King', 'Wright', 'Scott', 'Torres', 'Nguyen', 'Hill',
    'Flores', 'Green', 'Adams', 'Nelson', 'Baker', 'Hall', 'Rivera', 'Campbell'
]

STREET_NAMES = [
    'Main Street', 'Oak Avenue', 'Park Lane', 'Maple Drive', 'Cedar Road',
    'Pine Street', 'Washington Avenue', 'Lincoln Boulevard', 'First Street',
    'Second Avenue', 'Third Street', 'Elm Street', 'Church Street', 'High Street',
    'Market Street', 'Center Street', 'Spring Street', 'Broad Street', 'Mill Road',
    'Valley Drive', 'Hill Street', 'Forest Avenue', 'Garden Lane', 'River Road'
]

CITIES = [
    'Springfield', 'Franklin', 'Georgetown', 'Arlington', 'Fairview', 'Madison',
    'Salem', 'Oxford', 'Clinton', 'Dover', 'Newport', 'Hudson', 'Bristol',
    'Clayton', 'Richmond', 'Auburn', 'Ashland', 'Troy', 'Chester', 'Greenville'
]

def generate_clean_records(num_records=150):
    """Generate clean base records."""
    records = []
    
    for i in range(num_records):
        record = {
            'id': f'clean_{i:03d}',
            'full_name': f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
            'address': f"{random.randint(1, 9999)} {random.choice(STREET_NAMES)}, {random.choice(CITIES)}, {random.randint(10000, 99999)}",
        }
        records.append(record)
    
    return pd.DataFrame(records)

def apply_nickname_corruption(name):
    """Apply nickname substitution to a name."""
    parts = name.split()
    for i, part in enumerate(parts):
        if part in NICKNAMES:
            parts[i] = NICKNAMES[part]
            break
    return ' '.join(parts)

def apply_typo_corruption(text):
    """Apply a random typo (character swap or deletion)."""
    if len(text) < 2:
        return text
    
    text = list(text)
    corruption_type = random.choice(['swap', 'delete'])
    
    if corruption_type == 'swap' and len(text) >= 2:
        # Swap two adjacent characters
        pos = random.randint(0, len(text) - 2)
        text[pos], text[pos + 1] = text[pos + 1], text[pos]
    elif corruption_type == 'delete':
        # Delete one character
        pos = random.randint(0, len(text) - 1)
        text.pop(pos)
    
    return ''.join(text)

def apply_abbreviation_corruption(address):
    """Apply common address abbreviations."""
    abbreviations = {
        'Street': 'St', 'Avenue': 'Ave', 'Boulevard': 'Blvd', 
        'Drive': 'Dr', 'Road': 'Rd', 'Lane': 'Ln', 'Apartment': 'Apt'
    }
    
    for full, abbrev in abbreviations.items():
        if full in address:
            address = address.replace(full, abbrev)
            break
    
    return address

def apply_word_reordering(record):
    """Apply word reordering to name or address."""
    corruption_target = random.choice(['name', 'address'])
    
    if corruption_target == 'name':
        # Swap first and last name
        name_parts = record['full_name'].split()
        if len(name_parts) >= 2:
            record['full_name'] = f"{name_parts[-1]}, {' '.join(name_parts[:-1])}"
    else:
        # Reorder address components (city, street)
        address_parts = record['address'].split(', ')
        if len(address_parts) >= 3:
            # Swap city and street
            street_part = address_parts[0]
            city_part = address_parts[1]
            zip_part = address_parts[2]
            record['address'] = f"{city_part}, {street_part}, {zip_part}"
    
    return record

def apply_missing_field_corruption(record):
    """Remove zip code or apartment number from address."""
    address_parts = record['address'].split(', ')
    
    if len(address_parts) >= 3:
        # Remove zip code
        record['address'] = ', '.join(address_parts[:-1])
    
    return record

def generate_messy_records(clean_df):
    """Generate messy records by corrupting clean records + true negatives."""
    messy_records = []
    
    # Generate corrupted versions of clean records
    for _, clean_record in clean_df.iterrows():
        # Convert to dict to avoid pandas Series issues
        messy_record = {
            'messy_id': f"messy_{len(messy_records):03d}",
            'full_name': clean_record['full_name'],
            'address': clean_record['address'],
            'true_match_id': clean_record['id']
        }
        
        # Apply one random corruption
        corruption_type = random.choice(['nickname', 'typo_name', 'typo_address', 'abbreviation', 'reordering', 'missing'])
        
        if corruption_type == 'nickname':
            messy_record['full_name'] = apply_nickname_corruption(messy_record['full_name'])
        elif corruption_type == 'typo_name':
            messy_record['full_name'] = apply_typo_corruption(messy_record['full_name'])
        elif corruption_type == 'typo_address':
            messy_record['address'] = apply_typo_corruption(messy_record['address'])
        elif corruption_type == 'abbreviation':
            messy_record['address'] = apply_abbreviation_corruption(messy_record['address'])
        elif corruption_type == 'reordering':
            messy_record = apply_word_reordering(messy_record)
        elif corruption_type == 'missing':
            messy_record = apply_missing_field_corruption(messy_record)
        
        messy_records.append(messy_record)
    
    # Add 30 true negatives (records with no match)
    for i in range(30):
        fake_record = {
            'messy_id': f"messy_{len(messy_records):03d}",
            'full_name': f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
            'address': f"{random.randint(1, 9999)} {random.choice(STREET_NAMES)}, {random.choice(CITIES)}, {random.randint(10000, 99999)}",
            'true_match_id': None  # True negative
        }
        messy_records.append(fake_record)
    
    return pd.DataFrame(messy_records)

def main():
    print("Generating clean records...")
    clean_df = generate_clean_records(150)
    clean_df.to_csv('clean_records.csv', index=False)
    print(f"Generated {len(clean_df)} clean records → clean_records.csv")
    
    print("Generating messy records...")
    messy_df = generate_messy_records(clean_df)
    messy_df.to_csv('messy_records.csv', index=False)
    print(f"Generated {len(messy_df)} messy records → messy_records.csv")
    
    # Print some statistics
    true_positives = messy_df['true_match_id'].notna().sum()
    true_negatives = messy_df['true_match_id'].isna().sum()
    print(f"\nDataset composition:")
    print(f"- Corrupted records (should match): {true_positives}")
    print(f"- True negatives (should not match): {true_negatives}")
    print(f"- Total messy records: {len(messy_df)}")

if __name__ == '__main__':
    main()