import os
import shutil
from pathlib import Path
from typing import List, Optional
import json
import hashlib

class Utils:
    @staticmethod
    def ensure_directory(path: str):
        """Ensure a directory exists"""
        Path(path).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def cleanup_files(file_paths: List[str]):
        """Clean up temporary files"""
        for path in file_paths:
            try:
                if os.path.exists(path):
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
            except Exception as e:
                print(f"[ERROR] Error cleaning up {path}: {str(e)}")
    
    @staticmethod
    def get_file_hash(file_path: str) -> str:
        """Get SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    @staticmethod
    def save_metadata(metadata: dict, path: str):
        """Save metadata to JSON file"""
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @staticmethod
    def load_metadata(path: str) -> dict:
        """Load metadata from JSON file"""
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return {}