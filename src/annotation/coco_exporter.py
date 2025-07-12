"""
YOLO Vision Studio - COCO Format Exporter

Export annotations in COCO format with support for multiple output formats.
Handles conversion between annotation formats and dataset preparation.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import shutil
from PIL import Image
import numpy as np

try:
    from defusedxml.ElementTree import Element
except ImportError:
    Element = None

from ..utils.logger import get_logger


class COCOExporter:
    """Export annotations in COCO format."""

    def __init__(self):
        self.logger = get_logger(__name__)

    def export_dataset(
        self,
        project_path: Path,
        output_path: Path,
        export_format: str = "COCO",
        include_images: bool = True,
    ) -> bool:
        """Export complete dataset in specified format."""
        try:
            project_path = Path(project_path)
            output_path = Path(output_path)

            # Create output directory
            output_path.mkdir(parents=True, exist_ok=True)

            # Load project configuration
            config_file = project_path / "project_config.yaml"
            if config_file.exists():
                import yaml

                with open(config_file, "r") as f:
                    project_config = yaml.safe_load(f)
            else:
                project_config = {"classes": ["object"]}

            classes = project_config.get("classes", ["object"])

            if export_format.upper() == "COCO":
                return self._export_coco_format(
                    project_path, output_path, classes, include_images
                )
            elif export_format.upper() == "YOLO":
                return self._export_yolo_format(
                    project_path, output_path, classes, include_images
                )
            elif export_format.upper() == "PASCAL_VOC":
                return self._export_pascal_voc_format(
                    project_path, output_path, classes, include_images
                )
            else:
                raise ValueError(f"Unsupported export format: {export_format}")

        except Exception as e:
            self.logger.error(f"Dataset export failed: {e}")
            return False

    def _export_coco_format(
        self,
        project_path: Path,
        output_path: Path,
        classes: List[str],
        include_images: bool,
    ) -> bool:
        """Export in COCO format."""
        images_dir = project_path / "images"
        annotations_dir = project_path / "annotations"

        # Initialize COCO structure
        coco_data = {
            "info": {
                "description": f"YOLO Vision Studio Export - {project_path.name}",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "YOLO Vision Studio",
                "date_created": datetime.now().isoformat(),
            },
            "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
            "categories": [],
            "images": [],
            "annotations": [],
        }

        # Add categories
        for i, class_name in enumerate(classes):
            coco_data["categories"].append(
                {"id": i, "name": class_name, "supercategory": "object"}
            )

        # Process images and annotations
        annotation_id = 1

        for image_file in images_dir.glob("*.jpg"):
            if not image_file.exists():
                continue

            # Also check for .png files
            if not image_file.exists():
                png_file = image_file.with_suffix(".png")
                if png_file.exists():
                    image_file = png_file
                else:
                    continue

            # Get image info
            try:
                with Image.open(image_file) as img:
                    width, height = img.size
            except Exception as e:
                self.logger.warning(f"Failed to read image {image_file}: {e}")
                continue

            image_id = len(coco_data["images"])
            image_info = {
                "id": image_id,
                "file_name": image_file.name,
                "width": width,
                "height": height,
                "license": 1,
                "date_captured": datetime.now().isoformat(),
            }
            coco_data["images"].append(image_info)

            # Copy image if requested
            if include_images:
                dest_images_dir = output_path / "images"
                dest_images_dir.mkdir(exist_ok=True)
                shutil.copy2(image_file, dest_images_dir / image_file.name)

            # Process annotations
            annotation_file = annotations_dir / f"{image_file.stem}.json"
            if annotation_file.exists():
                try:
                    with open(annotation_file, "r") as f:
                        annotation_data = json.load(f)

                    for ann in annotation_data.get("annotations", []):
                        coco_annotation = self._convert_to_coco_annotation(
                            ann, annotation_id, image_id, width, height
                        )
                        if coco_annotation:
                            coco_data["annotations"].append(coco_annotation)
                            annotation_id += 1

                except Exception as e:
                    self.logger.warning(
                        f"Failed to process annotation {annotation_file}: {e}"
                    )

        # Save COCO annotations
        annotations_file = output_path / "annotations.json"
        with open(annotations_file, "w") as f:
            json.dump(coco_data, f, indent=2)

        self.logger.info(
            f"COCO export completed: {len(coco_data['images'])} images, "
            f"{len(coco_data['annotations'])} annotations"
        )

        return True

    def _convert_to_coco_annotation(
        self,
        annotation: Dict[str, Any],
        annotation_id: int,
        image_id: int,
        image_width: int,
        image_height: int,
    ) -> Optional[Dict[str, Any]]:
        """Convert annotation to COCO format."""
        try:
            annotation_type = annotation.get("annotation_type", "bbox")
            coordinates = annotation.get("coordinates", [])

            if annotation_type == "bbox" and len(coordinates) >= 4:
                # Convert from center format to COCO format if needed
                if annotation.get("format") == "center":
                    cx, cy, w, h = coordinates[:4]
                    x = cx - w / 2
                    y = cy - h / 2
                    bbox = [x, y, w, h]
                else:
                    # Assume already in x, y, w, h format
                    bbox = coordinates[:4]

                area = bbox[2] * bbox[3]

                return {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": annotation.get("class_id", 0),
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": [],
                }

            elif annotation_type == "polygon" and len(coordinates) >= 6:
                # Polygon annotation
                segmentation = [coordinates]  # COCO expects list of polygons

                # Calculate bounding box from polygon
                x_coords = coordinates[::2]
                y_coords = coordinates[1::2]

                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                area = bbox[2] * bbox[3]

                return {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": annotation.get("class_id", 0),
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": segmentation,
                }

        except Exception as e:
            self.logger.warning(f"Failed to convert annotation: {e}")

        return None

    def _export_yolo_format(
        self,
        project_path: Path,
        output_path: Path,
        classes: List[str],
        include_images: bool,
    ) -> bool:
        """Export in YOLO format."""
        images_dir = project_path / "images"
        annotations_dir = project_path / "annotations"

        # Create YOLO directory structure
        yolo_images_dir = output_path / "images"
        yolo_labels_dir = output_path / "labels"

        yolo_images_dir.mkdir(parents=True, exist_ok=True)
        yolo_labels_dir.mkdir(parents=True, exist_ok=True)

        # Create classes.txt
        with open(output_path / "classes.txt", "w") as f:
            for class_name in classes:
                f.write(f"{class_name}\n")

        # Create data.yaml
        data_yaml = {
            "path": str(output_path),
            "train": "images",
            "val": "images",
            "nc": len(classes),
            "names": classes,
        }

        import yaml

        with open(output_path / "data.yaml", "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        # Process each image
        processed_count = 0

        for image_file in images_dir.glob("*"):
            if image_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                continue

            # Get image dimensions
            try:
                with Image.open(image_file) as img:
                    width, height = img.size
            except Exception as e:
                self.logger.warning(f"Failed to read image {image_file}: {e}")
                continue

            # Copy image
            if include_images:
                shutil.copy2(image_file, yolo_images_dir / image_file.name)

            # Convert annotations
            annotation_file = annotations_dir / f"{image_file.stem}.json"
            label_file = yolo_labels_dir / f"{image_file.stem}.txt"

            yolo_annotations = []

            if annotation_file.exists():
                try:
                    with open(annotation_file, "r") as f:
                        annotation_data = json.load(f)

                    for ann in annotation_data.get("annotations", []):
                        yolo_line = self._convert_to_yolo_annotation(ann, width, height)
                        if yolo_line:
                            yolo_annotations.append(yolo_line)

                except Exception as e:
                    self.logger.warning(
                        f"Failed to process annotation {annotation_file}: {e}"
                    )

            # Save YOLO label file
            with open(label_file, "w") as f:
                for line in yolo_annotations:
                    f.write(f"{line}\n")

            processed_count += 1

        self.logger.info(f"YOLO export completed: {processed_count} images processed")
        return True

    def _convert_to_yolo_annotation(
        self, annotation: Dict[str, Any], image_width: int, image_height: int
    ) -> Optional[str]:
        """Convert annotation to YOLO format."""
        try:
            annotation_type = annotation.get("annotation_type", "bbox")
            coordinates = annotation.get("coordinates", [])
            class_id = annotation.get("class_id", 0)

            if annotation_type == "bbox" and len(coordinates) >= 4:
                # Convert to normalized center format
                if annotation.get("format") == "center":
                    # Already in center format
                    cx, cy, w, h = coordinates[:4]
                else:
                    # Convert from x, y, w, h to center format
                    x, y, w, h = coordinates[:4]
                    cx = x + w / 2
                    cy = y + h / 2

                # Normalize
                cx_norm = cx / image_width
                cy_norm = cy / image_height
                w_norm = w / image_width
                h_norm = h / image_height

                return (
                    f"{class_id} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
                )

        except Exception as e:
            self.logger.warning(f"Failed to convert annotation to YOLO: {e}")

        return None

    def _export_pascal_voc_format(
        self,
        project_path: Path,
        output_path: Path,
        classes: List[str],
        include_images: bool,
    ) -> bool:
        """Export in Pascal VOC format."""
        try:
            from defusedxml.ElementTree import Element, SubElement, tostring
            from defusedxml.minidom import parseString
        except ImportError:
            self.logger.error(
                "defusedxml not available for Pascal VOC export. Please install: pip install defusedxml"
            )
            return False

        images_dir = project_path / "images"
        annotations_dir = project_path / "annotations"

        # Create Pascal VOC structure
        voc_images_dir = output_path / "JPEGImages"
        voc_annotations_dir = output_path / "Annotations"

        voc_images_dir.mkdir(parents=True, exist_ok=True)
        voc_annotations_dir.mkdir(parents=True, exist_ok=True)

        processed_count = 0

        for image_file in images_dir.glob("*"):
            if image_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                continue

            # Get image info
            try:
                with Image.open(image_file) as img:
                    width, height = img.size
                    depth = len(img.getbands())
            except Exception as e:
                self.logger.warning(f"Failed to read image {image_file}: {e}")
                continue

            # Copy image
            if include_images:
                shutil.copy2(image_file, voc_images_dir / image_file.name)

            # Create XML annotation
            annotation_file = annotations_dir / f"{image_file.stem}.json"

            # Create XML structure
            root = Element("annotation")

            # Add basic info
            SubElement(root, "folder").text = "JPEGImages"
            SubElement(root, "filename").text = image_file.name
            SubElement(root, "path").text = str(voc_images_dir / image_file.name)

            # Add source
            source = SubElement(root, "source")
            SubElement(source, "database").text = "YOLO Vision Studio"

            # Add size
            size = SubElement(root, "size")
            SubElement(size, "width").text = str(width)
            SubElement(size, "height").text = str(height)
            SubElement(size, "depth").text = str(depth)

            SubElement(root, "segmented").text = "0"

            # Add objects
            if annotation_file.exists():
                try:
                    with open(annotation_file, "r") as f:
                        annotation_data = json.load(f)

                    for ann in annotation_data.get("annotations", []):
                        obj_element = self._convert_to_pascal_voc_object(
                            ann, classes, width, height
                        )
                        if obj_element is not None:
                            root.append(obj_element)

                except Exception as e:
                    self.logger.warning(
                        f"Failed to process annotation {annotation_file}: {e}"
                    )

            # Save XML file
            xml_str = parseString(tostring(root)).toprettyxml(indent="  ")
            xml_file = voc_annotations_dir / f"{image_file.stem}.xml"

            with open(xml_file, "w") as f:
                f.write(xml_str)

            processed_count += 1

        self.logger.info(
            f"Pascal VOC export completed: {processed_count} images processed"
        )
        return True

    def _convert_to_pascal_voc_object(
        self,
        annotation: Dict[str, Any],
        classes: List[str],
        image_width: int,
        image_height: int,
    ) -> Optional["Element"]:
        """Convert annotation to Pascal VOC object element."""
        try:
            from defusedxml.ElementTree import Element, SubElement

            annotation_type = annotation.get("annotation_type", "bbox")
            coordinates = annotation.get("coordinates", [])
            class_id = annotation.get("class_id", 0)

            if annotation_type == "bbox" and len(coordinates) >= 4:
                class_name = classes[class_id] if class_id < len(classes) else "object"

                # Convert coordinates to corner format
                if annotation.get("format") == "center":
                    cx, cy, w, h = coordinates[:4]
                    xmin = cx - w / 2
                    ymin = cy - h / 2
                    xmax = cx + w / 2
                    ymax = cy + h / 2
                else:
                    x, y, w, h = coordinates[:4]
                    xmin = x
                    ymin = y
                    xmax = x + w
                    ymax = y + h

                # Create object element
                obj = Element("object")
                SubElement(obj, "name").text = class_name
                SubElement(obj, "pose").text = "Unspecified"
                SubElement(obj, "truncated").text = "0"
                SubElement(obj, "difficult").text = "0"

                # Add bounding box
                bndbox = SubElement(obj, "bndbox")
                SubElement(bndbox, "xmin").text = str(int(xmin))
                SubElement(bndbox, "ymin").text = str(int(ymin))
                SubElement(bndbox, "xmax").text = str(int(xmax))
                SubElement(bndbox, "ymax").text = str(int(ymax))

                return obj

        except Exception as e:
            self.logger.warning(f"Failed to convert annotation to Pascal VOC: {e}")

        return None
