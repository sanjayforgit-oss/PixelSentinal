"""File management utilities."""

import os
from pathlib import Path
from typing import List


class FileManager:
    """Manage files and directories."""
    
    @staticmethod
    def create_directory(path: str) -> None:
        """Create directory.
        
        Args:
            path: Directory path
        """
        os.makedirs(path, exist_ok=True)
    
    @staticmethod
    def list_files(directory: str, extension: str = None) -> List[str]:
        """List files in directory.
        
        Args:
            directory: Directory path
            extension: File extension filter
            
        Returns:
            List of file paths
        """
        files = []
        for file in os.listdir(directory):
            if extension and not file.endswith(extension):
                continue
            files.append(os.path.join(directory, file))
        return files