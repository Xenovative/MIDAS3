#!/bin/bash
# backup_db.sh - Script to backup and restore ChromaDB data

# Set paths
BACKUP_DIR="/MIDAS3/backups"
CHROMA_DIR="/MIDAS3/db/chroma_db"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="$BACKUP_DIR/chroma_backup_$TIMESTAMP.tar.gz"

# Function to display usage
usage() {
    echo "Usage: $0 [backup|restore] [backup_file_path]"
    echo "  backup  - Create a backup of the ChromaDB data"
    echo "  restore - Restore ChromaDB data from a backup file"
    echo "  backup_file_path - Path to backup file (only needed for restore)"
    exit 1
}

# Create backup directory if it doesn't exist
mkdir -p $BACKUP_DIR

# Backup function
backup() {
    echo "Creating backup of ChromaDB data..."
    
    # Check if ChromaDB directory exists
    if [ ! -d "$CHROMA_DIR" ]; then
        echo "Error: ChromaDB directory not found at $CHROMA_DIR"
        exit 1
    fi
    
    # Create backup
    tar -czf $BACKUP_FILE -C $(dirname $CHROMA_DIR) $(basename $CHROMA_DIR)
    
    if [ $? -eq 0 ]; then
        echo "Backup created successfully: $BACKUP_FILE"
        echo "Backup size: $(du -h $BACKUP_FILE | cut -f1)"
    else
        echo "Error: Backup failed"
        exit 1
    fi
}

# Restore function
restore() {
    local restore_file=$1
    
    # Check if restore file exists
    if [ ! -f "$restore_file" ]; then
        echo "Error: Backup file not found at $restore_file"
        exit 1
    fi
    
    echo "Restoring ChromaDB data from $restore_file..."
    
    # Create temporary directory
    local temp_dir=$(mktemp -d)
    
    # Extract backup to temporary directory
    tar -xzf $restore_file -C $temp_dir
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to extract backup file"
        rm -rf $temp_dir
        exit 1
    fi
    
    # Check if extracted directory contains ChromaDB data
    if [ ! -d "$temp_dir/chroma_db" ]; then
        echo "Error: Invalid backup file, ChromaDB data not found"
        rm -rf $temp_dir
        exit 1
    fi
    
    # Stop services that might be using ChromaDB
    echo "Stopping MIDAS3 services..."
    # Add commands to stop services here if needed
    
    # Backup current data before restoring
    local current_backup="$BACKUP_DIR/pre_restore_backup_$TIMESTAMP.tar.gz"
    echo "Creating backup of current data before restoring..."
    tar -czf $current_backup -C $(dirname $CHROMA_DIR) $(basename $CHROMA_DIR)
    echo "Current data backed up to: $current_backup"
    
    # Remove current data
    echo "Removing current ChromaDB data..."
    rm -rf $CHROMA_DIR/*
    
    # Copy restored data
    echo "Copying restored data..."
    mkdir -p $CHROMA_DIR
    cp -R $temp_dir/chroma_db/* $CHROMA_DIR/
    
    # Set permissions
    chmod -R 777 $CHROMA_DIR
    
    # Clean up
    rm -rf $temp_dir
    
    echo "Restore completed successfully"
    echo "You may need to restart MIDAS3 services for changes to take effect"
}

# Main script logic
case "$1" in
    backup)
        backup
        ;;
    restore)
        if [ -z "$2" ]; then
            echo "Error: Backup file path required for restore operation"
            usage
        fi
        restore "$2"
        ;;
    *)
        usage
        ;;
esac

exit 0
