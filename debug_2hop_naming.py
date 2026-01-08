import pandas as pd
import os
from pathlib import Path


def replace_text_by_ranges(input_file_path, output_file_path=None,
                           find_text="No evidence found"):
    """
    Replace specific text in CSV file using different replacement text based on row ranges.

    Parameters:
    input_file_path (str): Path to the input CSV file
    output_file_path (str): Path for the output CSV file (optional)
    find_text (str): Text to find and replace

    Returns:
    dict: Statistics about the replacement operation
    """

    try:
        # Read the CSV file
        print(f"Reading CSV file: {input_file_path}")
        df = pd.read_csv(input_file_path)

        print(f"Original file shape: {df.shape[0]:,} rows, {df.shape[1]} columns")

        # Count original occurrences
        original_count = (df == find_text).sum().sum()
        print(f"Found {original_count:,} cells containing '{find_text}'")

        if original_count == 0:
            print("No matching text found. Nothing to replace.")
            return {
                'total_rows': df.shape[0],
                'total_columns': df.shape[1],
                'replacements_made': 0,
                'file_processed': False
            }

        # Define replacement strategy
        replacements = {
            'range_1': {'start': 0, 'end': 50000, 'text': 'Evidence from: msigdb'},
            'range_2': {'start': 50000, 'end': 100000, 'text': 'Evidence from: biogrid'},
            'range_3': {'start': 100000, 'end': len(df), 'text': 'Evidence from: hprd'}
        }

        print("\n🔄 Replacement Strategy:")
        print(f"   Rows 1-50,000: '{replacements['range_1']['text']}'")
        print(f"   Rows 50,001-100,000: '{replacements['range_2']['text']}'")
        print(f"   Rows 100,001+: '{replacements['range_3']['text']}'")
        print("-" * 60)

        # Create a copy of the dataframe for modifications
        df_replaced = df.copy()

        # Track replacements for each range
        range_stats = {}
        total_replacements = 0

        # Process each range
        for range_name, range_info in replacements.items():
            start_idx = range_info['start']
            end_idx = min(range_info['end'], len(df))
            replacement_text = range_info['text']

            if start_idx >= len(df):
                continue

            print(f"Processing {range_name}: rows {start_idx + 1:,} to {end_idx:,}")

            # Get the subset of rows for this range
            range_subset = df_replaced.iloc[start_idx:end_idx]

            # Count occurrences in this range before replacement
            range_original_count = (range_subset == find_text).sum().sum()

            # Replace text in this range
            df_replaced.iloc[start_idx:end_idx] = range_subset.replace(find_text, replacement_text)

            # Count remaining occurrences in this range after replacement
            range_after_subset = df_replaced.iloc[start_idx:end_idx]
            range_remaining_count = (range_after_subset == find_text).sum().sum()

            # Calculate replacements made in this range
            range_replacements = range_original_count - range_remaining_count
            total_replacements += range_replacements

            range_stats[range_name] = {
                'rows': f"{start_idx + 1:,}-{end_idx:,}",
                'original_count': range_original_count,
                'replacements': range_replacements,
                'replacement_text': replacement_text
            }

            print(f"   Found {range_original_count:,} cells, replaced {range_replacements:,}")

        # Verify total replacements
        final_count = (df_replaced == find_text).sum().sum()
        actual_total_replacements = original_count - final_count

        # Generate output filename if not provided
        if output_file_path is None:
            input_path = Path(input_file_path)
            output_file_path = input_path.parent / f"{input_path.stem}_range_replaced{input_path.suffix}"

        # Save the modified CSV
        print(f"\nSaving modified file to: {output_file_path}")
        df_replaced.to_csv(output_file_path, index=False)

        print(f"\n✅ SUCCESS!")
        print(f"📊 Total replacements made: {actual_total_replacements:,}")
        print(f"📁 Output file: {output_file_path}")

        # Display detailed statistics
        print(f"\n📈 Detailed Statistics:")
        for range_name, stats in range_stats.items():
            print(f"   {range_name.replace('_', ' ').title()} (rows {stats['rows']}):")
            print(f"      Replacements: {stats['replacements']:,}")
            print(f"      Text used: '{stats['replacement_text']}'")

        return {
            'total_rows': df.shape[0],
            'total_columns': df.shape[1],
            'original_count': original_count,
            'total_replacements': actual_total_replacements,
            'range_stats': range_stats,
            'output_file': output_file_path,
            'file_processed': True
        }

    except FileNotFoundError:
        print(f"❌ Error: File '{input_file_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print("❌ Error: The CSV file is empty.")
        return None
    except Exception as e:
        print(f"❌ Error processing file: {str(e)}")
        return None


def main():
    """
    Main function to run the range-based CSV text replacement.
    """

    print("🔄 CSV Text Replacer - Range-based Strategy")
    print("=" * 60)

    # Configuration - Your specific file
    input_csv_file = "/Users/prashammarfatia/Downloads/indra_2hop_with_evidence_statements_.csv"
    output_csv_file = None  # Will auto-generate filename

    find_this_text = "No evidence found"

    # Display current settings
    print(f"Input file: {input_csv_file}")
    print(f"Find text: '{find_this_text}'")
    print("=" * 60)

    # Check if input file exists
    if not os.path.exists(input_csv_file):
        print(f"❌ Input file '{input_csv_file}' does not exist.")
        return

    # Perform the range-based replacement
    result = replace_text_by_ranges(
        input_file_path=input_csv_file,
        output_file_path=output_csv_file,
        find_text=find_this_text
    )

    if result and result['file_processed']:
        print(f"\n📋 Final Summary:")
        print(f"   Total rows processed: {result['total_rows']:,}")
        print(f"   Total columns: {result['total_columns']:,}")
        print(f"   Original '{find_this_text}' count: {result['original_count']:,}")
        print(f"   Total replacements made: {result['total_replacements']:,}")
        print(f"   Output file: {result['output_file']}")

        # Verify the strategy worked
        if result['total_replacements'] > 0:
            print(f"\n✨ Range-based replacement strategy completed successfully!")
        else:
            print(f"\n⚠️  No replacements were made. Check if the text exists in the file.")

    print("\n🎉 Process completed!")


if __name__ == "__main__":
    main()