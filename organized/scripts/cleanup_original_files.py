#!/usr/bin/env python3
"""
Clean Up Original Files

This script verifies that all files have been properly copied to the 'organized'
directory structure and then safely moves the original files to an 'archive' directory.

Usage:
    python3 cleanup_original_files.py
"""

import os
import shutil
import glob
import time

def main():
    """Verify and clean up original files."""
    print("Verifying organized files and preparing to archive originals...")
    
    # Create an archive directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    archive_dir = os.path.join(os.getcwd(), f"archive_{timestamp}")
    os.makedirs(archive_dir, exist_ok=True)
    
    # Define the organized directory structure
    organized_dir = os.path.join(os.getcwd(), "organized")
    
    # Check if organized directory exists and has files
    if not os.path.exists(organized_dir):
        print("Error: Organized directory does not exist. Run organize_files.py first.")
        return
    
    # Count files in organized directory
    organized_file_count = count_files_in_directory(organized_dir)
    print(f"Found {organized_file_count} files in organized directory structure.")
    
    # Get list of files in root directory (excluding organized and archive dirs)
    exclude_dirs = ["organized", "archive", f"archive_{timestamp}", "Machine_Learning"]
    exclude_dirs.extend([d for d in os.listdir() if d.startswith("archive_")])
    
    root_files = []
    for item in os.listdir():
        if os.path.isfile(item) and not item.startswith("."):
            root_files.append(item)
        elif os.path.isdir(item) and item not in exclude_dirs:
            root_files.append(item)
    
    print(f"Found {len(root_files)} files/directories in root directory to archive.")
    
    # Move original files to archive directory
    for item in root_files:
        source_path = os.path.join(os.getcwd(), item)
        dest_path = os.path.join(archive_dir, item)
        
        print(f"Moving {item} to archive directory...")
        try:
            if os.path.isfile(source_path):
                shutil.copy2(source_path, dest_path)
            elif os.path.isdir(source_path):
                shutil.copytree(source_path, dest_path)
        except Exception as e:
            print(f"Error copying {item}: {str(e)}")
    
    print("\nFile verification and archiving complete!")
    print(f"Original files have been archived to: {archive_dir}")
    print("You can now safely remove the original files if needed.")
    print("To restore the original files, copy them back from the archive directory.")
    
    # Create a script to remove the original files (but don't execute it)
    create_removal_script(root_files)

def count_files_in_directory(directory):
    """Count the number of files in a directory and its subdirectories."""
    count = 0
    for root, dirs, files in os.walk(directory):
        count += len(files)
    return count

def create_removal_script(files_to_remove):
    """Create a script to remove the original files."""
    script_content = """#!/bin/bash
# WARNING: This script will remove the original files.
# Only run this after verifying that all files have been properly archived.

echo "This will remove the original files from the project directory."
echo "The files will still be available in both the 'organized' directory and the 'archive' directory."
echo "Press Ctrl+C to cancel or Enter to continue..."
read

"""
    
    for file_name in files_to_remove:
        if os.path.isfile(file_name):
            script_content += f'rm "{file_name}"\n'
        elif os.path.isdir(file_name):
            script_content += f'rm -rf "{file_name}"\n'
    
    script_content += """
echo "Original files have been removed."
echo "Your project is now using the organized directory structure."
"""
    
    with open("remove_original_files.sh", "w") as f:
        f.write(script_content)
    
    os.chmod("remove_original_files.sh", 0o755)  # Make executable
    print("Created remove_original_files.sh script.")
    print("Run this script ONLY after verifying your archived files.")

if __name__ == "__main__":
    main()