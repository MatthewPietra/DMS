"""File Management Utilities.

Provides file operations, data validation, backup management,
and storage optimization for DMS.
"""

import hashlib
import shutil
import threading
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Union

from .logger import get_component_logger

try:
    import send2trash
except ImportError:
    send2trash: Optional[ModuleType] = None  # type: ignore

try:
    from PIL import Image
except ImportError:
    Image: Optional[ModuleType] = None  # type: ignore


class FileManager:
    """Comprehensive file management system for DMS.

    Handles file operations, data validation, backup management,
    and storage optimization.
    """

    def __init__(self) -> None:
        """Initialize the FileManager.

        Sets up logging, thread safety, and file caching.
        """
        self.logger = get_component_logger("file_manager")
        self._lock = threading.Lock()
        self._file_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timeout = 300  # 5 minutes

    def ensure_directory(self, path: Union[str, Path]) -> Path:
        """Ensure directory exists, create if necessary.

        Args:
            path: Directory path to ensure exists.

        Returns:
            Path object of the ensured directory.

        Raises:
            Exception: If directory creation fails.
        """
        path = Path(path)
        try:
            path.mkdir(parents=True, exist_ok=True)
            return path
        except Exception as _e:
            self.logger.error(f"Failed to create directory {path}: {_e}")
            raise

    def copy_file(
        self, src: Union[str, Path], dst: Union[str, Path], overwrite: bool = False
    ) -> bool:
        """Copy file with error handling.

        Args:
            src: Source file path.
            dst: Destination file path.
            overwrite: Whether to overwrite existing destination file.

        Returns:
            True if copy successful, False otherwise.
        """
        src, dst = Path(src), Path(dst)

        if not src.exists():
            self.logger.error(f"Source file does not exist: {src}")
            return False

        if dst.exists() and not overwrite:
            self.logger.warning(f"Destination file exists: {dst}")
            return False

        try:
            # Ensure destination directory exists
            dst.parent.mkdir(parents=True, exist_ok=True)

            # Copy file
            shutil.copy2(src, dst)
            self.logger.debug(f"Copied file: {src} -> {dst}")
            return True

        except Exception as _e:
            self.logger.error(f"Failed to copy file {src} -> {dst}: {_e}")
            return False

    def move_file(
        self, src: Union[str, Path], dst: Union[str, Path], overwrite: bool = False
    ) -> bool:
        """Move file with error handling.

        Args:
            src: Source file path.
            dst: Destination file path.
            overwrite: Whether to overwrite existing destination file.

        Returns:
            True if move successful, False otherwise.
        """
        src, dst = Path(src), Path(dst)

        if not src.exists():
            self.logger.error(f"Source file does not exist: {src}")
            return False

        if dst.exists() and not overwrite:
            self.logger.warning(f"Destination file exists: {dst}")
            return False

        try:
            # Ensure destination directory exists
            dst.parent.mkdir(parents=True, exist_ok=True)

            # Move file
            shutil.move(str(src), str(dst))
            self.logger.debug(f"Moved file: {src} -> {dst}")
            return True

        except Exception as _e:
            self.logger.error(f"Failed to move file {src} -> {dst}: {_e}")
            return False

    def delete_file(self, path: Union[str, Path], safe: bool = True) -> bool:
        """Delete file with optional safe mode (move to trash).

        Args:
            path: File path to delete.
            safe: If True, move to trash instead of permanent deletion.

        Returns:
            True if deletion successful, False otherwise.
        """
        path = Path(path)

        if not path.exists():
            self.logger.warning(f"File does not exist: {path}")
            return True

        try:
            if safe and send2trash is not None:
                # Move to trash using send2trash if available
                send2trash.send2trash(str(path))
                self.logger.debug(f"Moved to trash: {path}")
            else:
                path.unlink()
                self.logger.debug(f"Deleted file: {path}")
            return True
        except Exception as _e:
            self.logger.error(f"Failed to delete file {path}: {_e}")
            return False

    def get_file_hash(
        self, path: Union[str, Path], algorithm: str = "md5"
    ) -> Optional[str]:
        """Calculate file hash.

        Args:
            path: File path to hash.
            algorithm: Hash algorithm to use (default: md5).

        Returns:
            Hex digest of file hash, or None if failed.
        """
        path = Path(path)

        if not path.exists():
            return None

        try:
            hash_obj = hashlib.new(algorithm)
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as _e:
            self.logger.error(f"Failed to calculate hash for {path}: {_e}")
            return None

    def find_duplicates(
        self, directory: Union[str, Path], extensions: Optional[List[str]] = None
    ) -> Dict[str, List[Path]]:
        """Find duplicate files in directory.

        Args:
            directory: Directory to search for duplicates.
            extensions: Optional list of file extensions to include.

        Returns:
            Dictionary mapping file hashes to lists of duplicate file paths.
        """
        directory = Path(directory)
        duplicates: Dict[str, List[Path]] = {}

        if not directory.exists():
            return duplicates

        # Get all files
        files: List[Path] = []
        if extensions:
            for ext in extensions:
                files.extend(directory.rglob(f"*.{ext}"))
        else:
            files = [f for f in directory.rglob("*") if f.is_file()]

        # Group by hash
        hash_groups: Dict[str, List[Path]] = {}

        for file_path in files:
            file_hash = self.get_file_hash(file_path)
            if file_hash:
                if file_hash not in hash_groups:
                    hash_groups[file_hash] = []
                hash_groups[file_hash].append(file_path)

        # Find duplicates
        for file_hash, paths in hash_groups.items():
            if len(paths) > 1:
                duplicates[file_hash] = paths

        self.logger.info(f"Found {len(duplicates)} groups of duplicate files")
        return duplicates

    def get_directory_size(self, directory: Union[str, Path]) -> int:
        """Get total size of directory in bytes.

        Args:
            directory: Directory path to calculate size for.

        Returns:
            Total size in bytes.
        """
        directory = Path(directory)

        if not directory.exists():
            return 0

        total_size = 0
        try:
            for path in directory.rglob("*"):
                if path.is_file():
                    total_size += path.stat().st_size
        except Exception as _e:
            self.logger.error(f"Failed to calculate directory size: {_e}")
        return total_size

    def cleanup_old_files(
        self,
        directory: Union[str, Path],
        max_age_days: int = 30,
        extensions: Optional[List[str]] = None,
    ) -> int:
        """Clean up old files in directory.

        Args:
            directory: Directory to clean up.
            max_age_days: Maximum age of files to keep.
            extensions: Optional list of file extensions to include.

        Returns:
            Number of files deleted.
        """
        directory = Path(directory)

        if not directory.exists():
            return 0

        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        deleted_count = 0

        # Get files to check
        files: List[Path] = []
        if extensions:
            for ext in extensions:
                files.extend(directory.rglob(f"*.{ext}"))
        else:
            files = [f for f in directory.rglob("*") if f.is_file()]

        for file_path in files:
            try:
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_time:
                    if self.delete_file(file_path):
                        deleted_count += 1
            except Exception as _e:
                self.logger.error(f"Error processing file {file_path}: {_e}")

        self.logger.info(f"Cleaned up {deleted_count} old files from {directory}")
        return deleted_count

    def backup_directory(
        self,
        source: Union[str, Path],
        backup_dir: Union[str, Path],
        compress: bool = True,
    ) -> Optional[Path]:
        """Create backup of directory.

        Args:
            source: Source directory to backup.
            backup_dir: Directory to store backup in.
            compress: Whether to compress the backup.

        Returns:
            Path to backup file/directory, or None if failed.
        """
        source, backup_dir = Path(source), Path(backup_dir)

        if not source.exists():
            self.logger.error(f"Source directory does not exist: {source}")
            return None

        # Create backup directory
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Generate backup name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{source.name}_backup_{timestamp}"

        try:
            if compress:
                # Create compressed archive
                backup_path = backup_dir / f"{backup_name}.tar.gz"
                shutil.make_archive(
                    str(backup_path.with_suffix("")),
                    "gztar",
                    str(source.parent),
                    str(source.name),
                )
            else:
                # Copy directory
                backup_path = backup_dir / backup_name
                shutil.copytree(source, backup_path)

            self.logger.info(f"Created backup: {backup_path}")
            return backup_path

        except Exception as _e:
            self.logger.error(f"Failed to create backup of {source}: {_e}")
            return None

    def validate_image_file(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Validate image file and return metadata.

        Args:
            path: Path to image file to validate.

        Returns:
            Dictionary containing validation results and metadata.
        """
        path = Path(path)

        result: Dict[str, Any] = {
            "valid": False,
            "error": None,
            "format": None,
            "size": None,
            "mode": None,
            "file_size": 0,
        }

        path_exists = path.exists()
        pil_available = Image is not None

        if not path_exists:
            result["error"] = "File does not exist"
        elif not pil_available:
            result["error"] = "PIL not available"
        else:
            try:
                # Get file size
                result["file_size"] = path.stat().st_size

                # Open and validate image
                with Image.open(path) as img:
                    result["valid"] = True
                    result["format"] = img.format
                    result["size"] = img.size
                    result["mode"] = img.mode
            except Exception as _e:
                result["error"] = str(_e)
        
        return result

    def _convert_image_internal(
        self, src: Path, dst: Path, format: str, quality: int
    ) -> bool:
        """Convert image to different format internally.

        Args:
            src: Source image path.
            dst: Destination image path.
            format: Target format (PNG, JPEG, etc.).
            quality: Quality setting for JPEG (1-100).

        Returns:
            True if conversion successful, False otherwise.
        """
        try:
            with Image.open(src) as img:
                # Convert to RGB if necessary
                if format.upper() == "JPEG" and img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")

                # Save with appropriate parameters
                save_kwargs: Dict[str, Any] = {}
                if format.upper() == "JPEG":
                    save_kwargs["quality"] = quality
                    save_kwargs["optimize"] = True

                img.save(dst, format=format.upper(), **save_kwargs)

            self.logger.debug(f"Converted image: {src} -> {dst} ({format})")
            return True
        except Exception as _e:
            self.logger.error(f"Failed to convert image {src}: {_e}")
            return False

    def convert_image_format(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
        format: str = "PNG",
        quality: int = 95,
    ) -> bool:
        """Convert image to different format.

        Args:
            src: Source image path.
            dst: Destination image path.
            format: Target format (PNG, JPEG, etc.).
            quality: Quality setting for JPEG (1-100).

        Returns:
            True if conversion successful, False otherwise.
        """
        src, dst = Path(src), Path(dst)

        if Image is None:
            self.logger.error("PIL not available for image conversion")
            return False

        # Convert the image
        return self._convert_image_internal(src, dst, format, quality)

    def organize_files_by_date(
        self,
        source_dir: Union[str, Path],
        target_dir: Union[str, Path],
        date_format: str = "%Y/%m/%d",
    ) -> int:
        """Organize files by date into subdirectories.

        Args:
            source_dir: Source directory containing files to organize.
            target_dir: Target directory for organized files.
            date_format: Date format for subdirectory names.

        Returns:
            Number of files moved.
        """
        source_dir, target_dir = Path(source_dir), Path(target_dir)

        moved_count = 0
        source_exists = source_dir.exists()
        
        if source_exists:
            moved_count = self._process_files_for_organization(
                source_dir, target_dir, date_format
            )
        
        self.logger.info(f"Organized {moved_count} files by date")
        return moved_count

    def _process_files_for_organization(
        self, source_dir: Path, target_dir: Path, date_format: str
    ) -> int:
        """Process files for organization by date.

        Args:
            source_dir: Source directory.
            target_dir: Target directory.
            date_format: Date format for subdirectory names.

        Returns:
            Number of files moved.
        """
        moved_count = 0
        files_to_process = [f for f in source_dir.rglob("*") if f.is_file()]
        
        for file_path in files_to_process:
            try:
                # Get file modification date
                file_date = datetime.fromtimestamp(file_path.stat().st_mtime)
                date_subdir = file_date.strftime(date_format)

                # Create target directory
                target_subdir = target_dir / date_subdir
                target_subdir.mkdir(parents=True, exist_ok=True)

                # Move file
                target_path = target_subdir / file_path.name
                move_success = self.move_file(file_path, target_path)
                if move_success:
                    moved_count += 1
            except Exception as _e:
                self.logger.error(f"Error organizing file {file_path}: {_e}")
        pass
        return moved_count

    def get_storage_stats(self, directory: Union[str, Path]) -> Dict[str, Any]:
        """Get storage statistics for directory.

        Args:
            directory: Directory to analyze.

        Returns:
            Dictionary containing storage statistics.
        """
        directory = Path(directory)

        stats: Dict[str, Any] = {
            "total_files": 0,
            "total_size": 0,
            "file_types": {},
            "largest_files": [],
            "oldest_file": None,
            "newest_file": None,
        }

        if not directory.exists():
            return stats

        files_info: List[Dict[str, Any]] = []

        for file_path in directory.rglob("*"):
            if not file_path.is_file():
                continue

            try:
                file_stat = file_path.stat()
                file_info: Dict[str, Any] = {
                    "path": file_path,
                    "size": file_stat.st_size,
                    "mtime": file_stat.st_mtime,
                    "extension": file_path.suffix.lower(),
                }
                files_info.append(file_info)

                # Update stats
                stats["total_files"] += 1
                stats["total_size"] += file_info["size"]

                # File type statistics
                ext = file_info["extension"] or "no_extension"
                if ext not in stats["file_types"]:
                    stats["file_types"][ext] = {"count": 0, "size": 0}
                stats["file_types"][ext]["count"] += 1
                stats["file_types"][ext]["size"] += file_info["size"]

            except Exception as _e:
                self.logger.error(f"Error processing file {file_path}: {_e}")

        if files_info:
            # Sort by size for largest files
            files_info.sort(key=lambda x: x["size"], reverse=True)
            stats["largest_files"] = [
                {"path": str(f["path"]), "size": f["size"]} for f in files_info[:10]
            ]

            # Find oldest and newest files
            files_info.sort(key=lambda x: x["mtime"])
            stats["oldest_file"] = {
                "path": str(files_info[0]["path"]),
                "mtime": files_info[0]["mtime"],
            }
            stats["newest_file"] = {
                "path": str(files_info[-1]["path"]),
                "mtime": files_info[-1]["mtime"],
            }

        return stats
