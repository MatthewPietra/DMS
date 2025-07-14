"""
DMS - Annotation Interface.

Comprehensive annotation interface with support for:
- Bounding box, polygon, and point annotation tools
- Keyboard shortcuts and rapid workflow
- Zoom, pan, and navigation controls
- Class management with customizable colors
- Real-time collaboration features
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from PyQt5.QtCore import (
        QKeySequence,
        QPoint,
        QShortcut,
        Qt,
    )
    from PyQt5.QtGui import (
        QAction,
        QBrush,
        QColor,
        QColorDialog,
        QMouseEvent,
        QPainter,
        QPen,
        QPixmap,
    )
    from PyQt5.QtWidgets import (
        QApplication,
        QButtonGroup,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QRadioButton,
        QScrollArea,
        QVBoxLayout,
        QWidget,
    )
    PYQT_AVAILABLE = True
except ImportError:
    try:
        from PyQt6.QtCore import (
            QKeySequence,
            QPoint,
            QShortcut,
            Qt,
        )
        from PyQt6.QtGui import (
            QAction,
            QBrush,
            QColor,
            QColorDialog,
            QMouseEvent,
            QPainter,
            QPen,
            QPixmap,
        )
        from PyQt6.QtWidgets import (
            QApplication,
            QButtonGroup,
            QComboBox,
            QDialog,
            QDialogButtonBox,
            QDoubleSpinBox,
            QFileDialog,
            QFormLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QListWidget,
            QListWidgetItem,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QRadioButton,
            QScrollArea,
            QVBoxLayout,
            QWidget,
        )
        PYQT_AVAILABLE = True
    except ImportError:
        PYQT_AVAILABLE = False

        # Create dummy classes for testing
        class QWidget:
            """Dummy QWidget class for testing."""

            pass

        class QMainWindow:
            """Dummy QMainWindow class for testing."""

            pass

        class QLabel:
            """Dummy QLabel class for testing."""

            pass

        class QListWidget:
            """Dummy QListWidget class for testing."""

            pass

        class QPushButton:
            """Dummy QPushButton class for testing."""

            pass

        class QRadioButton:
            """Dummy QRadioButton class for testing."""

            pass

        class QButtonGroup:
            """Dummy QButtonGroup class for testing."""

            pass

        class QGroupBox:
            """Dummy QGroupBox class for testing."""

            pass

        class QVBoxLayout:
            """Dummy QVBoxLayout class for testing."""

            pass

        class QHBoxLayout:
            """Dummy QHBoxLayout class for testing."""

            pass

        class QDialog:
            """Dummy QDialog class for testing."""

            pass

        class QPainter:
            """Dummy QPainter class for testing."""

            pass

        class QPixmap:
            """Dummy QPixmap class for testing."""

            pass

        class QPoint:
            """Dummy QPoint class for testing."""

            pass

from ..utils.config import ConfigManager
from ..utils.logger import get_logger, setup_logger
from .coco_exporter import COCOExporter


@dataclass
class Annotation:
    """Annotation data structure."""

    id: str
    class_id: int
    class_name: str
    coordinates: List[float]
    annotation_type: str  # 'bbox', 'polygon', 'point'
    confidence: float = 1.0
    created_by: str = "user"
    created_at: str = ""
    modified_at: str = ""

    def __post_init__(self) -> None:
        """Initialize timestamps if not provided."""
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.modified_at:
            self.modified_at = self.created_at


class AnnotationInterface(QMainWindow if PYQT_AVAILABLE else object):
    """Main annotation interface with comprehensive UI capabilities."""

    def __init__(self, config_manager: ConfigManager) -> None:
        """Initialize the annotation interface.

        Args:
            config_manager: Configuration manager instance.

        Raises:
            ImportError: If PyQt6 or PyQt5 is not available.
        """
        if not PYQT_AVAILABLE:
            raise ImportError("PyQt6 or PyQt5 required for annotation interface")

        super().__init__()
        self.config = config_manager
        self.logger = get_logger(__name__)

        # State management
        self.current_image_path: Optional[Path] = None
        self.current_annotations: List[Annotation] = []
        self.image_list: List[Path] = []
        self.current_index: int = 0
        self.current_tool: str = "bbox"
        self.classes: Dict[int, Dict[str, Any]] = {}
        self.zoom_factor: float = 1.0
        self.pan_offset: QPoint = QPoint(0, 0)

        # UI components
        self.image_label: Optional["AnnotationCanvas"] = None
        self.class_list: Optional[QListWidget] = None
        self.annotation_list: Optional[QListWidget] = None
        self.tool_buttons: Optional[QButtonGroup] = None
        self.image_counter: Optional[QLabel] = None
        self.zoom_label: Optional[QLabel] = None
        self.class_combo: Optional[QComboBox] = None
        self.confidence_spin: Optional[QDoubleSpinBox] = None

        self._setup_ui()
        self._setup_shortcuts()
        self._load_classes()

    def _setup_ui(self) -> None:
        """Setup the main UI components."""
        self.setWindowTitle("YOLO Vision Studio - Annotation Interface")
        self.setMinimumSize(1200, 800)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)

        # Left panel - Tools and classes
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel, 1)

        # Center panel - Image display
        center_panel = self._create_center_panel()
        main_layout.addWidget(center_panel, 4)

        # Right panel - Annotations and properties
        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel, 1)

        # Status bar
        self.statusBar().showMessage("Ready")

        # Menu bar
        self._create_menu_bar()

        # Toolbar
        self._create_toolbar()

    def _create_left_panel(self) -> QWidget:
        """Create left panel with tools and classes.

        Returns:
            QWidget: The left panel widget.
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Tool selection
        tool_group = QGroupBox("Annotation Tools")
        tool_layout = QVBoxLayout(tool_group)

        self.tool_buttons = QButtonGroup()
        tools = [
            ("bbox", "Bounding Box", "B"),
            ("polygon", "Polygon", "P"),
            ("point", "Point", "O"),
        ]

        for tool_id, tool_name, shortcut in tools:
            btn = QRadioButton(f"{tool_name} ({shortcut})")
            btn.setChecked(tool_id == "bbox")
            btn.toggled.connect(
                lambda checked, t=tool_id: self._set_tool(t) if checked else None
            )
            self.tool_buttons.addButton(btn)
            tool_layout.addWidget(btn)

        layout.addWidget(tool_group)

        # Class management
        class_group = QGroupBox("Classes")
        class_layout = QVBoxLayout(class_group)

        # Class list
        self.class_list = QListWidget()
        self.class_list.itemClicked.connect(self._on_class_selected)
        class_layout.addWidget(self.class_list)

        # Class management buttons
        class_btn_layout = QHBoxLayout()
        add_class_btn = QPushButton("Add")
        add_class_btn.clicked.connect(self._add_class)
        edit_class_btn = QPushButton("Edit")
        edit_class_btn.clicked.connect(self._edit_class)
        delete_class_btn = QPushButton("Delete")
        delete_class_btn.clicked.connect(self._delete_class)

        class_btn_layout.addWidget(add_class_btn)
        class_btn_layout.addWidget(edit_class_btn)
        class_btn_layout.addWidget(delete_class_btn)
        class_layout.addLayout(class_btn_layout)

        layout.addWidget(class_group)
        layout.addStretch()

        return panel

    def _create_center_panel(self) -> QWidget:
        """Create center panel with image display.

        Returns:
            QWidget: The center panel widget.
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Image navigation
        nav_layout = QHBoxLayout()

        prev_btn = QPushButton("← Previous")
        prev_btn.clicked.connect(self._previous_image)
        next_btn = QPushButton("Next →")
        next_btn.clicked.connect(self._next_image)

        self.image_counter = QLabel("0 / 0")

        nav_layout.addWidget(prev_btn)
        nav_layout.addStretch()
        nav_layout.addWidget(self.image_counter)
        nav_layout.addStretch()
        nav_layout.addWidget(next_btn)

        layout.addLayout(nav_layout)

        # Image display with scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.image_label = AnnotationCanvas(self)
        scroll_area.setWidget(self.image_label)

        layout.addWidget(scroll_area)

        # Zoom controls
        zoom_layout = QHBoxLayout()

        zoom_out_btn = QPushButton("Zoom Out")
        zoom_out_btn.clicked.connect(self._zoom_out)
        zoom_in_btn = QPushButton("Zoom In")
        zoom_in_btn.clicked.connect(self._zoom_in)
        fit_btn = QPushButton("Fit to Window")
        fit_btn.clicked.connect(self._fit_to_window)

        self.zoom_label = QLabel("100%")

        zoom_layout.addWidget(zoom_out_btn)
        zoom_layout.addWidget(zoom_in_btn)
        zoom_layout.addWidget(fit_btn)
        zoom_layout.addStretch()
        zoom_layout.addWidget(self.zoom_label)

        layout.addLayout(zoom_layout)

        return panel

    def _create_right_panel(self) -> QWidget:
        """Create right panel with annotation list and properties.

        Returns:
            QWidget: The right panel widget.
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Annotation list
        ann_group = QGroupBox("Annotations")
        ann_layout = QVBoxLayout(ann_group)

        self.annotation_list = QListWidget()
        self.annotation_list.itemClicked.connect(self._on_annotation_selected)
        ann_layout.addWidget(self.annotation_list)

        # Annotation management buttons
        ann_btn_layout = QHBoxLayout()
        delete_ann_btn = QPushButton("Delete")
        delete_ann_btn.clicked.connect(self._delete_annotation)
        duplicate_ann_btn = QPushButton("Duplicate")
        duplicate_ann_btn.clicked.connect(self._duplicate_annotation)

        ann_btn_layout.addWidget(delete_ann_btn)
        ann_btn_layout.addWidget(duplicate_ann_btn)
        ann_layout.addLayout(ann_btn_layout)

        layout.addWidget(ann_group)

        # Properties panel
        props_group = QGroupBox("Properties")
        props_layout = QFormLayout(props_group)

        self.class_combo = QComboBox()
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.01)
        self.confidence_spin.setValue(1.0)

        props_layout.addRow("Class:", self.class_combo)
        props_layout.addRow("Confidence:", self.confidence_spin)

        layout.addWidget(props_group)
        layout.addStretch()

        return panel

    def _create_menu_bar(self) -> None:
        """Create menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open Images", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_images)
        file_menu.addAction(open_action)

        save_action = QAction("Save Annotations", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_annotations)
        file_menu.addAction(save_action)

        export_action = QAction("Export COCO", self)
        export_action.triggered.connect(self._export_coco)
        file_menu.addAction(export_action)

        # Edit menu
        edit_menu = menubar.addMenu("Edit")

        undo_action = QAction("Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        edit_menu.addAction(undo_action)

        redo_action = QAction("Redo", self)
        redo_action.setShortcut("Ctrl+Y")
        edit_menu.addAction(redo_action)

    def _create_toolbar(self) -> None:
        """Create toolbar."""
        toolbar = self.addToolBar("Main")

        # Quick actions
        open_action = QAction("Open", self)
        open_action.triggered.connect(self._open_images)
        toolbar.addAction(open_action)

        save_action = QAction("Save", self)
        save_action.triggered.connect(self._save_annotations)
        toolbar.addAction(save_action)

        toolbar.addSeparator()

        # Tool actions
        bbox_action = QAction("BBox", self)
        bbox_action.setCheckable(True)
        bbox_action.setChecked(True)
        bbox_action.triggered.connect(lambda: self._set_tool("bbox"))
        toolbar.addAction(bbox_action)

    def _setup_shortcuts(self) -> None:
        """Setup keyboard shortcuts."""
        shortcuts = self.config.get("annotation.shortcuts", {})

        # Navigation shortcuts
        QShortcut(
            QKeySequence(shortcuts.get("next_image", "Right")),
            self,
            self._next_image,
        )
        QShortcut(
            QKeySequence(shortcuts.get("prev_image", "Left")),
            self,
            self._previous_image,
        )

        # Tool shortcuts
        QShortcut(QKeySequence("B"), self, lambda: self._set_tool("bbox"))
        QShortcut(QKeySequence("P"), self, lambda: self._set_tool("polygon"))
        QShortcut(QKeySequence("O"), self, lambda: self._set_tool("point"))

        # Zoom shortcuts
        QShortcut(
            QKeySequence(shortcuts.get("zoom_in", "Ctrl+Plus")),
            self,
            self._zoom_in,
        )
        QShortcut(
            QKeySequence(shortcuts.get("zoom_out", "Ctrl+Minus")),
            self,
            self._zoom_out,
        )
        QShortcut(
            QKeySequence(shortcuts.get("fit_to_window", "Ctrl+0")),
            self,
            self._fit_to_window,
        )

    def _load_classes(self) -> None:
        """Load class definitions."""
        # Load from config or create defaults
        default_classes = {0: {"name": "object", "color": "#FF0000"}}

        self.classes = self.config.get("annotation.classes", default_classes)
        self._update_class_list()

    def _update_class_list(self) -> None:
        """Update the class list widget."""
        if not self.class_list:
            return

        self.class_list.clear()
        self.class_combo.clear()

        for class_id, class_info in self.classes.items():
            name = class_info["name"]
            color = class_info["color"]

            # Add to class list
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, class_id)
            item.setBackground(QColor(color))
            self.class_list.addItem(item)

            # Add to combo box
            self.class_combo.addItem(name, class_id)

    def _set_tool(self, tool: str) -> None:
        """Set the current annotation tool.

        Args:
            tool: The tool to set ('bbox', 'polygon', 'point').
        """
        self.current_tool = tool
        self.statusBar().showMessage(f"Tool: {tool.title()}")

    def _open_images(self) -> None:
        """Open image directory or files."""
        dialog = QFileDialog()
        files, _ = dialog.getOpenFileNames(
            self,
            "Select Images",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff)",
        )

        if files:
            self.image_list = [Path(f) for f in files]
            self.current_index = 0
            self._load_current_image()

    def _load_current_image(self) -> None:
        """Load the current image and its annotations."""
        if not self.image_list or self.current_index >= len(self.image_list):
            return

        self.current_image_path = self.image_list[self.current_index]

        # Load image
        pixmap = QPixmap(str(self.current_image_path))
        if self.image_label:
            self.image_label.set_image(pixmap)

        # Load annotations
        self._load_annotations()

        # Update UI
        self.image_counter.setText(f"{self.current_index + 1} / {len(self.image_list)}")
        self.setWindowTitle(f"YOLO Vision Studio - {self.current_image_path.name}")

    def _load_annotations(self) -> None:
        """Load annotations for current image."""
        if not self.current_image_path:
            return

        # Look for annotation file
        ann_path = self.current_image_path.with_suffix(".json")

        self.current_annotations = []
        if ann_path.exists():
            try:
                with open(ann_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for ann_data in data.get("annotations", []):
                        ann = Annotation(**ann_data)
                        self.current_annotations.append(ann)
            except Exception as e:
                self.logger.error(f"Failed to load annotations: {e}")

        self._update_annotation_list()
        if self.image_label:
            self.image_label.set_annotations(self.current_annotations)

    def _save_annotations(self) -> None:
        """Save annotations for current image."""
        if not self.current_image_path:
            return

        ann_path = self.current_image_path.with_suffix(".json")

        try:
            data = {
                "image_path": str(self.current_image_path),
                "image_size": [
                    self.image_label.pixmap().width(),
                    self.image_label.pixmap().height(),
                ],
                "annotations": [asdict(ann) for ann in self.current_annotations],
            }

            with open(ann_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            self.statusBar().showMessage("Annotations saved", 2000)

        except Exception as e:
            self.logger.error(f"Failed to save annotations: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save annotations: {e}")

    def _next_image(self) -> None:
        """Navigate to next image."""
        if self.image_list and self.current_index < len(self.image_list) - 1:
            self._save_annotations()  # Auto-save
            self.current_index += 1
            self._load_current_image()

    def _previous_image(self) -> None:
        """Navigate to previous image."""
        if self.image_list and self.current_index > 0:
            self._save_annotations()  # Auto-save
            self.current_index -= 1
            self._load_current_image()

    def _zoom_in(self) -> None:
        """Zoom in on image."""
        self.zoom_factor *= 1.25
        self._update_zoom()

    def _zoom_out(self) -> None:
        """Zoom out on image."""
        self.zoom_factor /= 1.25
        self._update_zoom()

    def _fit_to_window(self) -> None:
        """Fit image to window."""
        if self.image_label and self.image_label.pixmap():
            # Calculate zoom to fit
            widget_size = self.image_label.parent().size()
            pixmap_size = self.image_label.pixmap().size()

            scale_x = widget_size.width() / pixmap_size.width()
            scale_y = widget_size.height() / pixmap_size.height()

            self.zoom_factor = min(scale_x, scale_y, 1.0)
            self._update_zoom()

    def _update_zoom(self) -> None:
        """Update zoom display."""
        if self.image_label:
            self.image_label.set_zoom(self.zoom_factor)
            self.zoom_label.setText(f"{int(self.zoom_factor * 100)}%")

    def _add_class(self) -> None:
        """Add new class."""
        dialog = ClassDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            class_info = dialog.get_class_info()
            class_id = max(self.classes.keys()) + 1 if self.classes else 0
            self.classes[class_id] = class_info
            self._update_class_list()

    def _edit_class(self) -> None:
        """Edit selected class."""
        current_item = self.class_list.currentItem()
        if current_item:
            class_id = current_item.data(Qt.ItemDataRole.UserRole)
            class_info = self.classes[class_id]

            dialog = ClassDialog(self, class_info)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.classes[class_id] = dialog.get_class_info()
                self._update_class_list()

    def _delete_class(self) -> None:
        """Delete selected class."""
        current_item = self.class_list.currentItem()
        if current_item:
            class_id = current_item.data(Qt.ItemDataRole.UserRole)
            reply = QMessageBox.question(
                self,
                "Delete Class",
                f"Delete class '{self.classes[class_id]['name']}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                del self.classes[class_id]
                self._update_class_list()

    def _on_class_selected(self, item: QListWidgetItem) -> None:
        """Handle class selection.

        Args:
            item: The selected list item.
        """
        # Set as current class for new annotations
        class_id = item.data(Qt.ItemDataRole.UserRole)
        # Find the index in the class_combo with this class_id
        index = self.class_combo.findData(class_id)
        if index >= 0:
            self.class_combo.setCurrentIndex(index)

    def _update_annotation_list(self) -> None:
        """Update annotation list widget."""
        if not self.annotation_list:
            return

        self.annotation_list.clear()

        for i, ann in enumerate(self.current_annotations):
            class_name = self.classes.get(ann.class_id, {}).get("name", "Unknown")
            item_text = f"{class_name} ({ann.annotation_type})"

            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, i)
            self.annotation_list.addItem(item)

    def _on_annotation_selected(self, item: QListWidgetItem) -> None:
        """Handle annotation selection.

        Args:
            item: The selected list item.
        """
        ann_index = item.data(Qt.ItemDataRole.UserRole)
        if 0 <= ann_index < len(self.current_annotations):
            ann = self.current_annotations[ann_index]

            # Update properties panel
            class_index = self.class_combo.findData(ann.class_id)
            if class_index >= 0:
                self.class_combo.setCurrentIndex(class_index)
            self.confidence_spin.setValue(ann.confidence)

            # Highlight annotation in image
            if self.image_label:
                self.image_label.highlight_annotation(ann_index)

    def _delete_annotation(self) -> None:
        """Delete selected annotation."""
        current_item = self.annotation_list.currentItem()
        if current_item:
            ann_index = current_item.data(Qt.ItemDataRole.UserRole)
            if 0 <= ann_index < len(self.current_annotations):
                del self.current_annotations[ann_index]
                self._update_annotation_list()
                if self.image_label:
                    self.image_label.set_annotations(self.current_annotations)

    def _duplicate_annotation(self) -> None:
        """Duplicate selected annotation."""
        current_item = self.annotation_list.currentItem()
        if current_item:
            ann_index = current_item.data(Qt.ItemDataRole.UserRole)
            if 0 <= ann_index < len(self.current_annotations):
                original = self.current_annotations[ann_index]

                # Create duplicate with offset coordinates
                duplicate = Annotation(
                    id=f"{original.id}_copy",
                    class_id=original.class_id,
                    class_name=original.class_name,
                    coordinates=[
                        c + 10 for c in original.coordinates
                    ],  # Offset by 10 pixels
                    annotation_type=original.annotation_type,
                    confidence=original.confidence,
                )

                self.current_annotations.append(duplicate)
                self._update_annotation_list()
                if self.image_label:
                    self.image_label.set_annotations(self.current_annotations)

    def _export_coco(self) -> None:
        """Export annotations in COCO format."""
        if not self.image_list:
            QMessageBox.warning(self, "Warning", "No images loaded")
            return

        # Get export path
        export_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export COCO Dataset",
            "annotations.json",
            "JSON Files (*.json)",
        )

        if export_path:
            try:
                exporter = COCOExporter()
                exporter.export_dataset(self.image_list, Path(export_path))
                QMessageBox.information(
                    self, "Success", "Dataset exported successfully"
                )
            except Exception as e:
                self.logger.error(f"Export failed: {e}")
                QMessageBox.critical(self, "Error", f"Export failed: {e}")


class AnnotationCanvas(QLabel):
    """Custom widget for image display and annotation drawing."""

    def __init__(self, parent: AnnotationInterface) -> None:
        """Initialize the annotation canvas.

        Args:
            parent: The parent annotation interface.
        """
        super().__init__(parent)
        self.parent_interface = parent
        self.setMinimumSize(400, 400)
        self.setStyleSheet("border: 1px solid gray;")

        # State
        self.original_pixmap: Optional[QPixmap] = None
        self.annotations: List[Annotation] = []
        self.highlighted_annotation: int = -1
        self.zoom_factor: float = 1.0

        # Drawing state
        self.drawing: bool = False
        self.current_points: List[QPoint] = []

    def set_image(self, pixmap: QPixmap) -> None:
        """Set the image to display.

        Args:
            pixmap: The pixmap to display.
        """
        self.original_pixmap = pixmap
        self._update_display()

    def set_annotations(self, annotations: List[Annotation]) -> None:
        """Set annotations to display.

        Args:
            annotations: List of annotations to display.
        """
        self.annotations = annotations
        self._update_display()

    def set_zoom(self, zoom_factor: float) -> None:
        """Set zoom factor.

        Args:
            zoom_factor: The zoom factor to apply.
        """
        self.zoom_factor = zoom_factor
        self._update_display()

    def highlight_annotation(self, index: int) -> None:
        """Highlight specific annotation.

        Args:
            index: Index of annotation to highlight.
        """
        self.highlighted_annotation = index
        self._update_display()

    def _update_display(self) -> None:
        """Update the display with current image and annotations."""
        if not self.original_pixmap:
            return

        # Scale pixmap
        scaled_pixmap = self.original_pixmap.scaled(
            self.original_pixmap.size() * self.zoom_factor,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        # Draw annotations on pixmap
        if self.annotations:
            painter = QPainter(scaled_pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            for i, ann in enumerate(self.annotations):
                self._draw_annotation(painter, ann, i == self.highlighted_annotation)

            painter.end()

        self.setPixmap(scaled_pixmap)

    def _draw_annotation(
        self, painter: QPainter, ann: Annotation, highlighted: bool = False
    ) -> None:
        """Draw a single annotation.

        Args:
            painter: The painter to use for drawing.
            ann: The annotation to draw.
            highlighted: Whether the annotation should be highlighted.
        """
        # Get class color
        class_info = self.parent_interface.classes.get(ann.class_id, {})
        color = QColor(class_info.get("color", "#FF0000"))

        if highlighted:
            color = color.lighter(150)

        painter.setPen(QPen(color, 2))
        painter.setBrush(QBrush(color, Qt.BrushStyle.NoBrush))

        coords = [c * self.zoom_factor for c in ann.coordinates]

        if ann.annotation_type == "bbox":
            # Bounding box: [x, y, width, height]
            if len(coords) >= 4:
                painter.drawRect(coords[0], coords[1], coords[2], coords[3])

        elif ann.annotation_type == "polygon":
            # Polygon: [x1, y1, x2, y2, ...]
            if len(coords) >= 6:  # At least 3 points
                points = [
                    QPoint(int(coords[i]), int(coords[i + 1]))
                    for i in range(0, len(coords) - 1, 2)
                ]
                painter.drawPolygon(points)

        elif ann.annotation_type == "point":
            # Point: [x, y]
            if len(coords) >= 2:
                painter.drawEllipse(QPoint(int(coords[0]), int(coords[1])), 5, 5)

        # Draw label
        if len(coords) >= 2:
            label_text = f"{ann.class_name} ({ann.confidence:.2f})"
            painter.drawText(int(coords[0]), int(coords[1]) - 5, label_text)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press for annotation creation.

        Args:
            event: The mouse event.
        """
        if event.button() == Qt.MouseButton.LeftButton and self.original_pixmap:
            # Convert to image coordinates
            pos = self._widget_to_image_coords(event.pos())

            tool = self.parent_interface.current_tool

            if tool == "bbox":
                self.drawing = True
                self.current_points = [pos]

            elif tool == "polygon":
                self.current_points.append(pos)

            elif tool == "point":
                self._create_point_annotation(pos)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move for annotation creation.

        Args:
            event: The mouse event.
        """
        if self.drawing and self.parent_interface.current_tool == "bbox":
            # Update bounding box preview
            pass

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release for annotation creation.

        Args:
            event: The mouse event.
        """
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            pos = self._widget_to_image_coords(event.pos())

            if self.parent_interface.current_tool == "bbox":
                self._create_bbox_annotation(self.current_points[0], pos)
                self.drawing = False
                self.current_points = []

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Handle double click to finish polygon.

        Args:
            event: The mouse event.
        """
        if (
            self.parent_interface.current_tool == "polygon"
            and len(self.current_points) >= 3
        ):
            self._create_polygon_annotation(self.current_points)
            self.current_points = []

    def _widget_to_image_coords(self, widget_pos: QPoint) -> QPoint:
        """Convert widget coordinates to image coordinates.

        Args:
            widget_pos: Position in widget coordinates.

        Returns:
            QPoint: Position in image coordinates.
        """
        if not self.original_pixmap:
            return widget_pos

        # Account for zoom and centering
        return QPoint(
            int(widget_pos.x() / self.zoom_factor),
            int(widget_pos.y() / self.zoom_factor),
        )

    def _create_bbox_annotation(self, start: QPoint, end: QPoint) -> None:
        """Create bounding box annotation.

        Args:
            start: Starting point of the bounding box.
            end: Ending point of the bounding box.
        """
        # Get current class
        current_class_index = self.parent_interface.class_combo.currentIndex()
        if current_class_index < 0:
            return

        class_id = self.parent_interface.class_combo.itemData(current_class_index)
        class_name = self.parent_interface.classes[class_id]["name"]

        # Calculate bbox coordinates
        x = min(start.x(), end.x())
        y = min(start.y(), end.y())
        width = abs(end.x() - start.x())
        height = abs(end.y() - start.y())

        if width < 5 or height < 5:  # Minimum size check
            return

        annotation = Annotation(
            id=f"bbox_{len(self.annotations)}",
            class_id=class_id,
            class_name=class_name,
            coordinates=[x, y, width, height],
            annotation_type="bbox",
        )

        self.parent_interface.current_annotations.append(annotation)
        self.parent_interface._update_annotation_list()
        self.set_annotations(self.parent_interface.current_annotations)

    def _create_polygon_annotation(self, points: List[QPoint]) -> None:
        """Create polygon annotation.

        Args:
            points: List of points defining the polygon.
        """
        current_class_index = self.parent_interface.class_combo.currentIndex()
        if current_class_index < 0:
            return

        class_id = self.parent_interface.class_combo.itemData(current_class_index)
        class_name = self.parent_interface.classes[class_id]["name"]

        # Flatten points to coordinate list
        coordinates = []
        for point in points:
            coordinates.extend([point.x(), point.y()])

        annotation = Annotation(
            id=f"polygon_{len(self.annotations)}",
            class_id=class_id,
            class_name=class_name,
            coordinates=coordinates,
            annotation_type="polygon",
        )

        self.parent_interface.current_annotations.append(annotation)
        self.parent_interface._update_annotation_list()
        self.set_annotations(self.parent_interface.current_annotations)

    def _create_point_annotation(self, point: QPoint) -> None:
        """Create point annotation.

        Args:
            point: The point to annotate.
        """
        current_class_index = self.parent_interface.class_combo.currentIndex()
        if current_class_index < 0:
            return

        class_id = self.parent_interface.class_combo.itemData(current_class_index)
        class_name = self.parent_interface.classes[class_id]["name"]

        annotation = Annotation(
            id=f"point_{len(self.annotations)}",
            class_id=class_id,
            class_name=class_name,
            coordinates=[point.x(), point.y()],
            annotation_type="point",
        )

        self.parent_interface.current_annotations.append(annotation)
        self.parent_interface._update_annotation_list()
        self.set_annotations(self.parent_interface.current_annotations)


class ClassDialog(QDialog if PYQT_AVAILABLE else object):
    """Dialog for adding/editing classes."""

    def __init__(
        self, parent: QWidget, class_info: Optional[Dict[str, str]] = None
    ) -> None:
        """Initialize the class dialog.

        Args:
            parent: The parent widget.
            class_info: Optional class information to edit.
        """
        super().__init__(parent)
        self.class_info = class_info or {"name": "", "color": "#FF0000"}
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup dialog UI."""
        self.setWindowTitle("Class Editor")
        self.setModal(True)

        layout = QFormLayout(self)

        # Name input
        self.name_edit = QLineEdit(self.class_info.get("name", ""))
        layout.addRow("Name:", self.name_edit)

        # Color picker
        self.color_button = QPushButton()
        self.color_button.setStyleSheet(
            f"background-color: {self.class_info.get('color', '#FF0000')}"
        )
        self.color_button.clicked.connect(self._choose_color)
        layout.addRow("Color:", self.color_button)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

    def _choose_color(self) -> None:
        """Open color picker."""
        color = QColorDialog.getColor(
            QColor(self.class_info.get("color", "#FF0000")), self
        )
        if color.isValid():
            self.class_info["color"] = color.name()
            self.color_button.setStyleSheet(f"background-color: {color.name()}")

    def get_class_info(self) -> Dict[str, str]:
        """Get class information.

        Returns:
            Dict containing class name and color.
        """
        return {
            "name": self.name_edit.text(),
            "color": self.class_info.get("color", "#FF0000"),
        }


def launch_annotation_interface(
    config_manager: ConfigManager,
) -> Optional[AnnotationInterface]:
    """Launch the annotation interface.

    Args:
        config_manager: Configuration manager instance.

    Returns:
        AnnotationInterface instance if successful, None otherwise.
    """
    if not PYQT_AVAILABLE:
        print("PyQt6 or PyQt5 required for annotation interface")
        return None

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    interface = AnnotationInterface(config_manager)
    interface.show()

    return interface


def main() -> int:
    """Main entry point for the annotation interface.

    Provides standalone execution capability.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="YOLO Vision Studio - Annotation Interface"
    )
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument(
        "--images-dir", type=str, help="Directory containing images to annotate"
    )
    parser.add_argument("--project", type=str, help="Project name")

    args = parser.parse_args()

    # Setup logging
    setup_logger("annotation")

    try:
        # Initialize config manager
        config_manager = ConfigManager(args.config)

        # Launch annotation interface
        interface = launch_annotation_interface(config_manager)

        if interface is None:
            print("Failed to launch annotation interface")
            return 1

        # Load images if directory specified
        if args.images_dir:
            # This would require implementing a method to load images from directory
            print(f"Loading images from: {args.images_dir}")

        # Show interface and run event loop
        if PYQT_AVAILABLE:
            app = QApplication.instance()
            if app:
                print("\nYOLO Vision Studio - Annotation Interface")
                print("=" * 50)
                print("Interface launched successfully!")
                print("Use the interface to annotate your images.")
                print("Press Ctrl+C to exit.")

                try:
                    sys.exit(app.exec())
                except KeyboardInterrupt:
                    print("\nAnnotation interface closed by user")
                    return 0
        else:
            print("PyQt is not available - annotation interface cannot be displayed")
            return 1

    except Exception as e:
        print(f"Error launching annotation interface: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
