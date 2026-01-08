import pandas as pd
import os
from pathlib import Path


def replace_text_in_csv(input_file_path, output_file_path=None,
                        find_text="No evidence found",
                        replace_text="Evidence from: msigdb"):
    """
    Replace specific text in all cells of a CSV file.

    Parameters:
    input_file_path (str): Path to the input CSV file
    output_file_path (str): Path for the output CSV file (optional)
    find_text (str): Text to find and replace
    replace_text (str): Text to replace with

    Returns:
    dict: Statistics about the replacement operation
    """

    try:
        # Read the CSV file
        print(f"Reading CSV file: {input_file_path}")
        df = pd.read_csv(input_file_path)

        print(f"Original file shape: {df.shape[0]} rows, {df.shape[1]} columns")

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

        # Replace the text in all cells
        print("Performing text replacement...")
        df_replaced = df.replace(find_text, replace_text)

        # Verify replacements
        remaining_count = (df_replaced == find_text).sum().sum()
        replacements_made = original_count - remaining_count

        # Generate output filename if not provided
        if output_file_path is None:
            input_path = Path(input_file_path)
            output_file_path = input_path.parent / f"{input_path.stem}_replaced{input_path.suffix}"

        # Save the modified CSV
        print(f"Saving modified file to: {output_file_path}")
        df_replaced.to_csv(output_file_path, index=False)

        print(f"\n✅ SUCCESS!")
        print(f"📊 Replacements made: {replacements_made:,}")
        print(f"📁 Output file: {output_file_path}")

        return {
            'total_rows': df.shape[0],
            'total_columns': df.shape[1],
            'replacements_made': replacements_made,
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
    Main function to run the CSV text replacement.
    Modify the file paths and text values as needed.
    """

    print("🔄 CSV Text Replacer")
    print("=" * 50)

    # Configuration - MODIFY THESE PATHS AND TEXT AS NEEDED
    input_csv_file = "/Users/prashammarfatia/Downloads/indra_3hop_with_statements_.csv"
    output_csv_file = None  # Will auto-generate if None, or specify custom path

    find_this_text = "No evidence found"
    replace_with_text = "Evidence from: msigdb"

    # Display current settings
    print(f"Input file: {input_csv_file}")
    print(f"Find text: '{find_this_text}'")
    print(f"Replace with: '{replace_with_text}'")
    print("-" * 50)

    # Check if input file exists
    if not os.path.exists(input_csv_file):
        print(f"❌ Input file '{input_csv_file}' does not exist.")
        print("\n📝 To use this script:")
        print("1. Place your CSV file in the same folder as this script")
        print("2. Update the 'input_csv_file' variable with your file name")
        print("3. Run the script again")
        return

    # Perform the replacement
    result = replace_text_in_csv(
        input_file_path=input_csv_file,
        output_file_path=output_csv_file,
        find_text=find_this_text,
        replace_text=replace_with_text
    )

    if result and result['file_processed']:
        print("\n📈 Summary:")
        print(f"   Total rows processed: {result['total_rows']:,}")
        print(f"   Total columns: {result['total_columns']:,}")
        print(f"   Replacements made: {result['replacements_made']:,}")
        print(f"   Output file: {result['output_file']}")

    print("\n🎉 Process completed!")


# Additional utility function for batch processing
def batch_replace_in_directory(directory_path, file_pattern="*.csv",
                               find_text="No evidence found",
                               replace_text="Evidence from: msigdb"):
    """
    Process multiple CSV files in a directory.

    Parameters:
    directory_path (str): Path to directory containing CSV files
    file_pattern (str): Pattern to match files (default: "*.csv")
    find_text (str): Text to find and replace
    replace_text (str): Text to replace with
    """

    directory = Path(directory_path)
    csv_files = list(directory.glob(file_pattern))

    if not csv_files:
        print(f"No CSV files found in {directory_path}")
        return

    print(f"Found {len(csv_files)} CSV files to process:")
    for file in csv_files:
        print(f"  - {file.name}")

    total_replacements = 0
    processed_files = 0

    for csv_file in csv_files:
        print(f"\n🔄 Processing: {csv_file.name}")
        result = replace_text_in_csv(
            input_file_path=str(csv_file),
            find_text=find_text,
            replace_text=replace_text
        )

        if result and result['file_processed']:
            total_replacements += result['replacements_made']
            processed_files += 1

    print(f"\n📊 Batch processing complete!")
    print(f"   Files processed: {processed_files}/{len(csv_files)}")
    print(f"   Total replacements: {total_replacements:,}")


if __name__ == "__main__":
    main()

    # Uncomment the line below to process multiple files in a directory
    # batch_replace_in_directory("path/to/your/csv/files/")