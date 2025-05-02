#!/usr/bin/env python3
import os
import glob
import re
import csv
import json
from pathlib import Path

# Define the paths
blog_dir = "website/src/content/blog"
output_csv = "data/summarized_paper.csv"

# Ensure output directory exists
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# List to store paper data
papers = []

# Function to extract frontmatter from markdown files
def extract_frontmatter(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract the frontmatter (content between --- and ---)
    frontmatter_match = re.search(r'^---\s+(.*?)\s+---', content, re.DOTALL)
    if not frontmatter_match:
        return None
    
    frontmatter = frontmatter_match.group(1)
    
    # Parse the frontmatter
    data = {}
    for line in frontmatter.strip().split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip().strip('"')
            
            # Special handling for lists (authors, tags)
            if value.startswith('[') and value.endswith(']'):
                try:
                    # Try to parse as JSON
                    value = json.loads(value)
                except json.JSONDecodeError:
                    # If not valid JSON, parse manually
                    value = [item.strip().strip('"\'') for item in value[1:-1].split(',')]
            
            data[key] = value
    
    return data

# Find all markdown files recursively
md_files = glob.glob(f"{blog_dir}/**/*.md", recursive=True)

# Process each markdown file
for md_file in md_files:
    frontmatter = extract_frontmatter(md_file)
    
    if frontmatter and 'type' in frontmatter and 'id' in frontmatter and 'score' in frontmatter and 'slug' in frontmatter:
        # Extract type, id, score and slug
        paper_data = {
            'type': str(frontmatter.get('type', '')),
            'id': str(frontmatter.get('id', '')),
            'score': frontmatter.get('score', ''),
            'slug': frontmatter.get('slug', '')
        }
        
        papers.append(paper_data)
        print(f"Found paper: {frontmatter.get('title', '')} ({paper_data['id']})")

# Save to CSV
if papers:
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['type', 'id', 'score', 'slug']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for paper in papers:
            writer.writerow(paper)
    
    print(f"\nSuccessfully saved {len(papers)} papers to {output_csv}")
else:
    print("No papers with required fields (type, id, score, slug) found.") 