"""
Export Validation Tests

Test suite for dataset export validation and format verification.
"""

import json
import tempfile
import unittest

try:
    import defusedxml.ElementTree as ET
except ImportError:
    # Fallback to standard library with warning
    import warnings
    import xml.etree.ElementTree as ET

    warnings.warn(
        "defusedxml not available, using potentially unsafe XML parsing",
        SecurityWarning,
    )
import shutil
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.annotation.coco_exporter import COCOExporter
from src.utils.metrics import BoundingBox


class TestCOCOExport(unittest.TestCase):
    """Test COCO format export validation"""

    def setUp(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = COCOExporter()

        # Create sample annotations
        self.sample_annotations = {
            "image1.jpg": [
                BoundingBox(0.1, 0.1, 0.3, 0.3, class_id=0, confidence=0.9),
                BoundingBox(0.6, 0.6, 0.8, 0.8, class_id=1, confidence=0.8),
            ],
            "image2.jpg": [BoundingBox(0.2, 0.2, 0.4, 0.4, class_id=0, confidence=0.7)],
        }

        self.classes = {"0": "person", "1": "car"}

    def tearDown(self):
        """Cleanup test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_coco_format_structure(self):
        """Test COCO format structure validation"""
        output_path = Path(self.temp_dir) / "test_coco.json"

        # Export to COCO format
        self.exporter.export_coco(self.sample_annotations, self.classes, output_path)

        # Validate file exists
        self.assertTrue(output_path.exists())

        # Load and validate JSON structure
        with open(output_path, "r") as f:
            coco_data = json.load(f)

        # Check required COCO fields
        required_fields = ["info", "licenses", "images", "annotations", "categories"]
        for field in required_fields:
            self.assertIn(field, coco_data)

        # Validate categories
        self.assertEqual(len(coco_data["categories"]), len(self.classes))
        for category in coco_data["categories"]:
            self.assertIn("id", category)
            self.assertIn("name", category)
            self.assertIn("supercategory", category)

        # Validate images
        self.assertEqual(len(coco_data["images"]), len(self.sample_annotations))
        for image in coco_data["images"]:
            self.assertIn("id", image)
            self.assertIn("file_name", image)
            self.assertIn("width", image)
            self.assertIn("height", image)

        # Validate annotations
        total_annotations = sum(len(anns) for anns in self.sample_annotations.values())
        self.assertEqual(len(coco_data["annotations"]), total_annotations)

        for annotation in coco_data["annotations"]:
            self.assertIn("id", annotation)
            self.assertIn("image_id", annotation)
            self.assertIn("category_id", annotation)
            self.assertIn("bbox", annotation)
            self.assertIn("area", annotation)
            self.assertIn("iscrowd", annotation)

    def test_coco_bbox_format(self):
        """Test COCO bounding box format validation"""
        output_path = Path(self.temp_dir) / "test_bbox.json"

        self.exporter.export_coco(self.sample_annotations, self.classes, output_path)

        with open(output_path, "r") as f:
            coco_data = json.load(f)

        # COCO bbox format: [x, y, width, height]
        for annotation in coco_data["annotations"]:
            bbox = annotation["bbox"]
            self.assertEqual(len(bbox), 4)

            # All values should be positive
            for value in bbox:
                self.assertGreaterEqual(value, 0)

            # Width and height should be positive
            self.assertGreater(bbox[2], 0)  # width
            self.assertGreater(bbox[3], 0)  # height

    def test_yolo_format_export(self):
        """Test YOLO format export validation"""
        output_dir = Path(self.temp_dir) / "yolo_export"
        output_dir.mkdir(exist_ok=True)

        # Export to YOLO format
        self.exporter.export_yolo(self.sample_annotations, self.classes, output_dir)

        # Check that annotation files are created
        for image_name in self.sample_annotations.keys():
            annotation_file = output_dir / f"{Path(image_name).stem}.txt"
            self.assertTrue(annotation_file.exists())

            # Validate YOLO format content
            with open(annotation_file, "r") as f:
                lines = f.readlines()

            expected_annotations = len(self.sample_annotations[image_name])
            self.assertEqual(len(lines), expected_annotations)

            for line in lines:
                parts = line.strip().split()
                self.assertEqual(
                    len(parts), 5
                )  # class_id, center_x, center_y, width, height

                # Validate format
                class_id = int(parts[0])
                center_x, center_y, width, height = map(float, parts[1:])

                # Check ranges
                self.assertGreaterEqual(class_id, 0)
                self.assertLessEqual(class_id, len(self.classes) - 1)

                # Normalized coordinates should be in [0, 1]
                for coord in [center_x, center_y, width, height]:
                    self.assertGreaterEqual(coord, 0.0)
                    self.assertLessEqual(coord, 1.0)

        # Check classes file
        classes_file = output_dir / "classes.txt"
        self.assertTrue(classes_file.exists())

        with open(classes_file, "r") as f:
            class_lines = f.readlines()

        self.assertEqual(len(class_lines), len(self.classes))

    def test_pascal_voc_export(self):
        """Test Pascal VOC format export validation"""
        output_dir = Path(self.temp_dir) / "voc_export"
        output_dir.mkdir(exist_ok=True)

        # Export to Pascal VOC format
        self.exporter.export_pascal_voc(
            self.sample_annotations,
            self.classes,
            output_dir,
            image_width=640,
            image_height=480,
        )

        # Check that XML files are created
        for image_name in self.sample_annotations.keys():
            xml_file = output_dir / f"{Path(image_name).stem}.xml"
            self.assertTrue(xml_file.exists())

            # Validate XML structure
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Check required elements
            self.assertEqual(root.tag, "annotation")
            self.assertIsNotNone(root.find("filename"))
            self.assertIsNotNone(root.find("size"))

            # Check size element
            size_elem = root.find("size")
            self.assertIsNotNone(size_elem.find("width"))
            self.assertIsNotNone(size_elem.find("height"))
            self.assertIsNotNone(size_elem.find("depth"))

            # Check objects
            objects = root.findall("object")
            expected_objects = len(self.sample_annotations[image_name])
            self.assertEqual(len(objects), expected_objects)

            for obj in objects:
                self.assertIsNotNone(obj.find("name"))
                self.assertIsNotNone(obj.find("bndbox"))

                # Check bounding box
                bndbox = obj.find("bndbox")
                self.assertIsNotNone(bndbox.find("xmin"))
                self.assertIsNotNone(bndbox.find("ymin"))
                self.assertIsNotNone(bndbox.find("xmax"))
                self.assertIsNotNone(bndbox.find("ymax"))

                # Validate coordinates
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)

                self.assertGreaterEqual(xmin, 0)
                self.assertGreaterEqual(ymin, 0)
                self.assertLess(xmin, xmax)
                self.assertLess(ymin, ymax)

    def test_tensorflow_export(self):
        """Test TensorFlow format export validation"""
        output_dir = Path(self.temp_dir) / "tf_export"
        output_dir.mkdir(exist_ok=True)

        # Export to TensorFlow format
        self.exporter.export_tensorflow(
            self.sample_annotations, self.classes, output_dir
        )

        # Check required files
        train_record = output_dir / "train.tfrecord"
        val_record = output_dir / "val.tfrecord"
        label_map = output_dir / "label_map.pbtxt"

        # At least one of train/val should exist
        self.assertTrue(train_record.exists() or val_record.exists())
        self.assertTrue(label_map.exists())

        # Validate label map format
        with open(label_map, "r") as f:
            content = f.read()

        # Should contain item definitions
        self.assertIn("item", content)
        self.assertIn("id:", content)
        self.assertIn("name:", content)


class TestExportIntegrity(unittest.TestCase):
    """Test export data integrity and consistency"""

    def setUp(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = COCOExporter()

    def tearDown(self):
        """Cleanup test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_round_trip_consistency(self):
        """Test that export-import maintains data consistency"""
        # Create original data
        original_annotations = {
            "test.jpg": [
                BoundingBox(0.1, 0.2, 0.3, 0.4, class_id=0, confidence=0.9),
                BoundingBox(0.5, 0.6, 0.7, 0.8, class_id=1, confidence=0.8),
            ]
        }
        classes = {"0": "person", "1": "car"}

        # Export to YOLO format
        yolo_dir = Path(self.temp_dir) / "yolo"
        yolo_dir.mkdir(exist_ok=True)

        self.exporter.export_yolo(original_annotations, classes, yolo_dir)

        # Read back the exported data
        annotation_file = yolo_dir / "test.txt"
        self.assertTrue(annotation_file.exists())

        with open(annotation_file, "r") as f:
            lines = f.readlines()

        # Verify data integrity
        self.assertEqual(len(lines), 2)

        for i, line in enumerate(lines):
            parts = line.strip().split()
            class_id = int(parts[0])
            center_x, center_y, width, height = map(float, parts[1:])

            # Convert back to corner coordinates
            x1 = center_x - width / 2
            y1 = center_y - height / 2
            x2 = center_x + width / 2
            y2 = center_y + height / 2

            original_bbox = original_annotations["test.jpg"][i]

            # Check class ID
            self.assertEqual(class_id, original_bbox.class_id)

            # Check coordinates (with small tolerance for floating point)
            self.assertAlmostEqual(x1, original_bbox.x1, places=5)
            self.assertAlmostEqual(y1, original_bbox.y1, places=5)
            self.assertAlmostEqual(x2, original_bbox.x2, places=5)
            self.assertAlmostEqual(y2, original_bbox.y2, places=5)

    def test_empty_annotations_handling(self):
        """Test handling of empty annotations"""
        empty_annotations = {"empty.jpg": []}
        classes = {"0": "person"}

        # Should not crash with empty annotations
        output_path = Path(self.temp_dir) / "empty_coco.json"

        try:
            self.exporter.export_coco(empty_annotations, classes, output_path)

            # Validate structure
            with open(output_path, "r") as f:
                coco_data = json.load(f)

            self.assertIn("annotations", coco_data)
            self.assertEqual(len(coco_data["annotations"]), 0)

        except Exception as e:
            self.fail(f"Empty annotations should be handled gracefully: {e}")

    def test_large_dataset_export(self):
        """Test export performance with larger datasets"""
        # Create larger dataset
        large_annotations = {}
        classes = {str(i): f"class_{i}" for i in range(10)}

        for i in range(100):  # 100 images
            image_name = f"image_{i:03d}.jpg"
            annotations = []

            # Add 5-10 annotations per image
            import random

            random.seed(42)  # Deterministic for testing

            for j in range(random.randint(5, 10)):
                x1 = random.uniform(0, 0.8)
                y1 = random.uniform(0, 0.8)
                x2 = x1 + random.uniform(0.1, 0.2)
                y2 = y1 + random.uniform(0.1, 0.2)
                class_id = random.randint(0, 9)

                annotations.append(BoundingBox(x1, y1, x2, y2, class_id=class_id))

            large_annotations[image_name] = annotations

        # Test COCO export performance
        import time

        start_time = time.time()

        output_path = Path(self.temp_dir) / "large_coco.json"
        self.exporter.export_coco(large_annotations, classes, output_path)

        export_time = time.time() - start_time

        # Should complete within reasonable time (adjust as needed)
        self.assertLess(export_time, 10.0)  # 10 seconds max

        # Validate output
        self.assertTrue(output_path.exists())

        with open(output_path, "r") as f:
            coco_data = json.load(f)

        self.assertEqual(len(coco_data["images"]), 100)
        self.assertEqual(len(coco_data["categories"]), 10)

        # Count total annotations
        total_annotations = sum(len(anns) for anns in large_annotations.values())
        self.assertEqual(len(coco_data["annotations"]), total_annotations)


class TestExportValidation(unittest.TestCase):
    """Test export validation utilities"""

    def setUp(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Cleanup test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_validate_coco_format(self):
        """Test COCO format validation"""
        # Create valid COCO file
        valid_coco = {
            "info": {"description": "Test dataset"},
            "licenses": [],
            "images": [{"id": 1, "file_name": "test.jpg", "width": 640, "height": 480}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [100, 100, 50, 50],
                    "area": 2500,
                    "iscrowd": 0,
                }
            ],
            "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
        }

        valid_file = Path(self.temp_dir) / "valid.json"
        with open(valid_file, "w") as f:
            json.dump(valid_coco, f)

        # Validation should pass
        self.assertTrue(self._validate_coco_file(valid_file))

        # Test invalid COCO file
        invalid_coco = {"invalid": "structure"}
        invalid_file = Path(self.temp_dir) / "invalid.json"
        with open(invalid_file, "w") as f:
            json.dump(invalid_coco, f)

        # Validation should fail
        self.assertFalse(self._validate_coco_file(invalid_file))

    def _validate_coco_file(self, file_path: Path) -> bool:
        """Validate COCO format file"""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            required_fields = [
                "info",
                "licenses",
                "images",
                "annotations",
                "categories",
            ]
            for field in required_fields:
                if field not in data:
                    return False

            # Validate structure
            if not isinstance(data["images"], list):
                return False
            if not isinstance(data["annotations"], list):
                return False
            if not isinstance(data["categories"], list):
                return False

            return True

        except Exception:
            return False

    def test_validate_yolo_format(self):
        """Test YOLO format validation"""
        # Create valid YOLO annotation
        valid_yolo_dir = Path(self.temp_dir) / "valid_yolo"
        valid_yolo_dir.mkdir(exist_ok=True)

        annotation_file = valid_yolo_dir / "test.txt"
        with open(annotation_file, "w") as f:
            f.write("0 0.5 0.5 0.2 0.3\n")
            f.write("1 0.7 0.3 0.1 0.2\n")

        classes_file = valid_yolo_dir / "classes.txt"
        with open(classes_file, "w") as f:
            f.write("person\ncar\n")

        # Validation should pass
        self.assertTrue(self._validate_yolo_directory(valid_yolo_dir))

        # Test invalid YOLO format
        invalid_yolo_dir = Path(self.temp_dir) / "invalid_yolo"
        invalid_yolo_dir.mkdir(exist_ok=True)

        invalid_annotation = invalid_yolo_dir / "test.txt"
        with open(invalid_annotation, "w") as f:
            f.write("invalid format\n")

        # Validation should fail
        self.assertFalse(self._validate_yolo_directory(invalid_yolo_dir))

    def _validate_yolo_directory(self, directory: Path) -> bool:
        """Validate YOLO format directory"""
        try:
            # Check for annotation files
            annotation_files = list(directory.glob("*.txt"))
            if not annotation_files:
                return False

            # Validate annotation format
            for file_path in annotation_files:
                if file_path.name == "classes.txt":
                    continue

                with open(file_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split()
                        if len(parts) != 5:
                            return False

                        try:
                            class_id = int(parts[0])
                            coords = [float(x) for x in parts[1:]]

                            # Check coordinate ranges
                            for coord in coords:
                                if not (0.0 <= coord <= 1.0):
                                    return False

                        except ValueError:
                            return False

            return True

        except Exception:
            return False


if __name__ == "__main__":
    unittest.main()
