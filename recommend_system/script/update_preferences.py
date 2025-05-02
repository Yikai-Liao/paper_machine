import sys
import csv
import json
import os
from pathlib import Path
from collections import defaultdict
import re
import polars as pl

def main():
    if len(sys.argv) != 3:
        print("Usage: python update_preferences.py <reactions_file> <paper_info_csv>")
        sys.exit(1)
        
    reactions_file = sys.argv[1]
    paper_info_csv = sys.argv[2]
    preference_base_dir = Path("data/preference")

    # 1. Load paper info (slug -> id)
    slug_to_id = {}
    print(f"Attempting to load paper info from: {paper_info_csv}")
    try:
        with open(paper_info_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or 'slug' not in reader.fieldnames or 'id' not in reader.fieldnames:
                 print(f"Error: CSV {paper_info_csv} is missing required columns ('slug', 'id') or is empty.")
                 sys.exit(1)
            for row in reader:
                if row.get('slug') and row.get('id'):
                    slug_to_id[row['slug']] = row['id']
        print(f"Loaded info for {len(slug_to_id)} papers from {paper_info_csv}")
    except FileNotFoundError:
        print(f"Error: Paper info file not found at {paper_info_csv}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading paper info CSV {paper_info_csv}: {e}")
        sys.exit(1)
        
    if not slug_to_id:
         print("Warning: No slug-to-id mapping loaded. Cannot process reactions.")
         sys.exit(0) 

    # 2. Load reactions
    reactions = []
    print(f"Attempting to load reactions from: {reactions_file}")
    try:
        with open(reactions_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t') 
                if len(parts) == 2:
                    slug, reaction_type = parts
                    if reaction_type in ["THUMBS_UP", "THUMBS_DOWN"]:
                        if slug in slug_to_id:
                            preference = "like" if reaction_type == "THUMBS_UP" else "dislike"
                            paper_id = slug_to_id[slug]
                            match = re.match(r"(\d{4}-\d{2})-.*", slug)
                            if match:
                                year_month = match.group(1)
                                reactions.append({"id": paper_id, "preference": preference, "year_month": year_month, "slug": slug})
                            else:
                                print(f"Warning: Could not extract YYYY-MM from slug '{slug}'. Skipping reaction.")
                        else:
                            print(f"Warning: Slug '{slug}' from reaction not found in paper info. Skipping reaction.")
                elif line.strip(): 
                    print(f"Warning: Malformed reaction line: {line.strip()}")
        print(f"Loaded {len(reactions)} valid reactions to process.")
    except FileNotFoundError:
        print(f"Error: Reactions file not found at {reactions_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading reactions file {reactions_file}: {e}")
        sys.exit(1)

    if not reactions:
        print("No valid reactions to process. Exiting.")
        sys.exit(0)

    # 3. Group reactions by target CSV file
    prefs_by_file = defaultdict(list)
    for r in reactions:
        target_csv = preference_base_dir / f"{r['year_month']}.csv"
        # Store as id -> preference dict for easier lookup within the target file group
        prefs_by_file[target_csv].append({"id": r['id'], "preference": r['preference']})

    # 4. Update CSV files using Polars, preserving order
    preference_base_dir.mkdir(parents=True, exist_ok=True)
    updated_files = []
    expected_schema = {"id": pl.Utf8, "preference": pl.Utf8}

    for target_csv, updates_list in prefs_by_file.items():
        print(f"Processing updates for {target_csv}...")
        file_updated = False
        df_existing = pl.DataFrame(schema=expected_schema) # Start with empty DF
        updates_dict = {up["id"]: up["preference"] for up in updates_list} # Convert list to dict for faster lookup
        processed_update_ids = set()
        
        try:
            # Read existing data with Polars
            if target_csv.is_file():
                try:
                    df_existing = pl.read_csv(target_csv, schema_overrides=expected_schema, raise_if_empty=False)
                    if df_existing.is_empty() and target_csv.stat().st_size > 0:
                         # Handle case where file exists but might be empty after header or corrupted
                         print(f"Warning: Read empty DataFrame from non-empty file {target_csv}. Will overwrite.")
                         df_existing = pl.DataFrame(schema=expected_schema)
                except (pl.SchemaError, pl.ComputeError) as e:
                    print(f"Warning: Error reading {target_csv} ({type(e).__name__}). Assuming file needs creation/overwrite.")
                    df_existing = pl.DataFrame(schema=expected_schema)
                except Exception as e:
                    print(f"Unexpected error reading {target_csv}: {e}. Assuming file needs creation/overwrite.")
                    df_existing = pl.DataFrame(schema=expected_schema)
            else:
                 print(f"Info: File {target_csv} does not exist. Will create new.")

            # Apply updates to existing rows
            updated_rows_count = 0
            preferences_to_update = []
            for i, row_id in enumerate(df_existing["id"]):
                if row_id in updates_dict:
                    new_preference = updates_dict[row_id]
                    current_preference = df_existing["preference"][i]
                    if current_preference != new_preference:
                        print(f"  Updating existing ID {row_id}: '{current_preference}' -> '{new_preference}'")
                        preferences_to_update.append(new_preference)
                        updated_rows_count += 1
                        file_updated = True
                    else:
                        preferences_to_update.append(current_preference) # Keep existing
                    processed_update_ids.add(row_id)
                else:
                    preferences_to_update.append(df_existing["preference"][i]) # Keep existing
            
            # Update the 'preference' column if changes were made
            if updated_rows_count > 0:
                df_existing = df_existing.with_columns(pl.Series("preference", preferences_to_update))

            # Identify and prepare new rows
            new_rows_list = []
            new_rows_count = 0
            for paper_id, new_preference in updates_dict.items():
                if paper_id not in processed_update_ids:
                    print(f"  Adding new ID {paper_id}: '{new_preference}'")
                    new_rows_list.append({"id": paper_id, "preference": new_preference})
                    new_rows_count += 1
                    file_updated = True
            
            df_new = pl.DataFrame(new_rows_list, schema=expected_schema)

            if updated_rows_count > 0 or new_rows_count > 0:
                print(f"  Summary for {target_csv}: {updated_rows_count} updated, {new_rows_count} added.")
            else:
                 print(f"  No changes needed for {target_csv}.")

            # Write back if updated
            if file_updated:
                updated_files.append(str(target_csv.relative_to(Path.cwd()))) 
                
                # Combine existing (potentially updated) and new rows
                df_final = pl.concat([df_existing, df_new], how="vertical_relaxed")
                                
                # Write using Polars
                df_final.write_csv(target_csv)
                print(f"Successfully updated {target_csv} using Polars (order preserved)")

        except Exception as e:
            print(f"Error processing file {target_csv} with Polars: {e}")
            import traceback
            traceback.print_exc()

    # Output updated file paths for git commit step
    print("\n<<<UPDATED_FILES_START>>>")
    for f_rel in updated_files:
        print(f_rel)
    print("<<<UPDATED_FILES_END>>>")


if __name__ == "__main__":
    main() 