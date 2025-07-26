#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Annotation Widget.

GUI component for data annotation interface.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from PySide6.QtCore import (
    QPoint,
    QRect,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...annotation.annotation_interface import Annotation
from ...utils.config import ConfigManager
from ...utils.logger import get_logger


class AnnotationCanvas(QWidget):
    """Custom widget for image display and annotation drawing."""

    annotation_created = Signal(Annotation)
    annotation_selected = Signal(int)
    annotation_deleted = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize the annotation canvas."""
        super().__init__(parent)
        self.logger = get_logger(__name__)

        # State management
        self.original_pixmap: Optional[QPixmap] = None
        self.display_pixmap: Optional[QPixmap] = None
        self.annotations: List[Annotation] = []
        self.selected_annotation: int = -1
        self.zoom_factor: float = 1.0
        self.pan_offset: QPoint = QPoint(0, 0)

        # Drawing state
        self.drawing: bool = False
        self.current_tool: str = "bbox"
        self.current_points: List[QPoint] = []
        self.drag_start: Optional[QPoint] = None

        # UI setup
        self.setMinimumSize(400, 400)
        self.setStyleSheet("border: 1px solid #666; background-color: #2b2b2b;")
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_image(self, image_path: Path) -> bool:
        """Set the image to display.

        Args:
            image_path: Path to the image file.

        Returns:
            True if image loaded successfully, False otherwise.
        """
        try:
            self.original_pixmap = QPixmap(str(image_path))
            if self.original_pixmap.isNull():
                self.logger.error(f"Failed to load image: {image_path}")
                return False

            self._update_display()
            return True
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {e}")
            return False

    def set_annotations(self, annotations: List[Annotation]) -> None:
        """Set annotations to display.

        Args:
            annotations: List of annotations to display.
        """
        self.annotations = annotations
        self._update_display()

    def set_tool(self, tool: str) -> None:
        """Set the current annotation tool.

        Args:
            tool: Tool type ('bbox', 'polygon', 'point').
        """
        self.current_tool = tool
        self.drawing = False
        self.current_points = []
        self.update()

    def set_zoom(self, zoom_factor: float) -> None:
        """Set zoom factor.

        Args:
            zoom_factor: Zoom factor (0.1 to 5.0).
        """
        self.zoom_factor = max(0.1, min(5.0, zoom_factor))
        self._update_display()

    def fit_to_window(self) -> None:
        """Fit image to window."""
        if not self.original_pixmap:
            return

        widget_size = self.size()
        image_size = self.original_pixmap.size()

        scale_x = widget_size.width() / image_size.width()
        scale_y = widget_size.height() / image_size.height()
        self.zoom_factor = min(scale_x, scale_y) * 0.9

        self._update_display()

    def _update_display(self) -> None:
        """Update the display pixmap."""
        if not self.original_pixmap:
            return

        # Create scaled pixmap
        scaled_size = self.original_pixmap.size() * self.zoom_factor
        self.display_pixmap = self.original_pixmap.scaled(
            scaled_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.update()

    def _widget_to_image_coords(self, widget_pos: QPoint) -> QPoint:
        """Convert widget coordinates to image coordinates.

        Args:
            widget_pos: Position in widget coordinates.

        Returns:
            Position in image coordinates.
        """
        if not self.display_pixmap:
            return widget_pos

        # Account for centering and zoom
        widget_center = QPoint(self.width() // 2, self.height() // 2)
        image_center = QPoint(
            self.display_pixmap.width() // 2,
            self.display_pixmap.height() // 2,
        )

        # Calculate offset from widget center
        offset = widget_pos - widget_center

        # Convert to image coordinates
        image_pos = image_center + offset
        return QPoint(
            int(image_pos.x() / self.zoom_factor),
            int(image_pos.y() / self.zoom_factor),
        )

    def _image_to_widget_coords(self, image_pos: QPoint) -> QPoint:
        """Convert image coordinates to widget coordinates.

        Args:
            image_pos: Position in image coordinates.

        Returns:
            Position in widget coordinates.
        """
        if not self.display_pixmap:
            return image_pos

        # Convert to display coordinates
        display_pos = QPoint(
            int(image_pos.x() * self.zoom_factor),
            int(image_pos.y() * self.zoom_factor),
        )

        # Account for centering
        widget_center = QPoint(self.width() // 2, self.height() // 2)
        image_center = QPoint(
            self.display_pixmap.width() // 2,
            self.display_pixmap.height() // 2,
        )

        return widget_center + (display_pos - image_center)

    def paintEvent(self, event: Any) -> None:
        """Paint the canvas.

        Args:
            event: Paint event.
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw background
        painter.fillRect(self.rect(), QColor("#2b2b2b"))

        if not self.display_pixmap:
            # Draw placeholder
            painter.setPen(QPen(QColor("#666"), 2))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "No image loaded"
            )
            return

        # Draw image
        image_rect = self.display_pixmap.rect()
        image_rect.moveCenter(self.rect().center())
        painter.drawPixmap(image_rect, self.display_pixmap, self.display_pixmap.rect())

        # Draw annotations
        self._draw_annotations(painter)

        # Draw current drawing
        self._draw_current_drawing(painter)

    def _draw_annotations(self, painter: QPainter) -> None:
        """Draw all annotations.

        Args:
            painter: QPainter instance.
        """
        for i, annotation in enumerate(self.annotations):
            highlighted = i == self.selected_annotation
            self._draw_annotation(painter, annotation, highlighted)

    def _draw_annotation(
        self, painter: QPainter, annotation: Annotation, highlighted: bool = False
    ) -> None:
        """Draw a single annotation.

        Args:
            painter: QPainter instance.
            annotation: Annotation to draw.
            highlighted: Whether to highlight the annotation.
        """
        # Get class color (default to red if not found)
        color = QColor("#FF0000")
        parent_widget = cast(AnnotationWidget, self.parent())
        if (
            hasattr(parent_widget, "classes")
            and annotation.class_id in parent_widget.classes
        ):
            color = QColor(
                parent_widget.classes[annotation.class_id].get("color", "#FF0000")
            )

        if highlighted:
            color = color.lighter(150)

        painter.setPen(QPen(color, 2))
        painter.setBrush(QBrush(color, Qt.BrushStyle.NoBrush))

        # Convert coordinates to widget space
        coords = annotation.coordinates

        if annotation.annotation_type == "bbox" and len(coords) >= 4:
            # Bounding box: [x, y, width, height]
            x, y, w, h = coords[:4]
            rect = QRect(int(x), int(y), int(w), int(h))
            widget_rect = self._image_to_widget_coords(rect.topLeft())
            widget_size = QSize(int(w * self.zoom_factor), int(h * self.zoom_factor))
            display_rect = QRect(widget_rect, widget_size)
            painter.drawRect(display_rect)

            # Draw label
            label_text = f"{annotation.class_name}"
            painter.drawText(display_rect.topLeft() + QPoint(2, -5), label_text)

        elif annotation.annotation_type == "polygon" and len(coords) >= 6:
            # Polygon: [x1, y1, x2, y2, ...]
            points = []
            for i in range(0, len(coords) - 1, 2):
                if i + 1 < len(coords):
                    point = QPoint(int(coords[i]), int(coords[i + 1]))
                    widget_point = self._image_to_widget_coords(point)
                    points.append(widget_point)

            if len(points) >= 3:
                painter.drawPolygon(points)

        elif annotation.annotation_type == "point" and len(coords) >= 2:
            # Point: [x, y]
            point = QPoint(int(coords[0]), int(coords[1]))
            widget_point = self._image_to_widget_coords(point)
            painter.drawEllipse(widget_point, 5, 5)

    def _draw_current_drawing(self, painter: QPainter) -> None:
        """Draw the current drawing in progress.

        Args:
            painter: QPainter instance.
        """
        if not self.drawing or not self.current_points:
            return

        painter.setPen(QPen(QColor("#00FF00"), 2, Qt.PenStyle.DashLine))
        painter.setBrush(QBrush(QColor("#00FF00"), Qt.BrushStyle.NoBrush))

        if self.current_tool == "bbox" and len(self.current_points) >= 1:
            # Draw bounding box preview
            start = self.current_points[0]
            if self.drag_start:
                rect = QRect(start, self.drag_start)
                painter.drawRect(rect)

        elif self.current_tool == "polygon" and len(self.current_points) >= 2:
            # Draw polygon preview
            painter.drawPolyline(self.current_points)

            # Draw points
            for point in self.current_points:
                painter.drawEllipse(point, 3, 3)

    def mousePressEvent(self, event: Any) -> None:
        """Handle mouse press events.

        Args:
            event: Mouse press event.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self._widget_to_image_coords(event.pos())

            if self.current_tool == "bbox":
                self.drawing = True
                self.current_points = [pos]
                self.drag_start = event.pos()

            elif self.current_tool == "polygon":
                self.current_points.append(event.pos())

            elif self.current_tool == "point":
                self._create_point_annotation(pos)

        self.update()

    def mouseMoveEvent(self, event: Any) -> None:
        """Handle mouse move events.

        Args:
            event: Mouse move event.
        """
        if self.drawing and self.current_tool == "bbox":
            self.drag_start = event.pos()
            self.update()

    def mouseReleaseEvent(self, event: Any) -> None:
        """Handle mouse release events.

        Args:
            event: Mouse release event.
        """
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            if self.current_tool == "bbox":
                end_pos = self._widget_to_image_coords(event.pos())
                if self.current_points:
                    self._create_bbox_annotation(self.current_points[0], end_pos)

            self.drawing = False
            self.current_points = []
            self.drag_start = None
            self.update()

    def mouseDoubleClickEvent(self, event: Any) -> None:
        """Handle double click events.

        Args:
            event: Double click event.
        """
        if self.current_tool == "polygon" and len(self.current_points) >= 3:
            self._create_polygon_annotation(self.current_points)
            self.current_points = []
            self.update()

    def _create_bbox_annotation(self, start: QPoint, end: QPoint) -> None:
        """Create bounding box annotation.

        Args:
            start: Start point of bounding box.
            end: End point of bounding box.
        """
        parent_widget = cast(AnnotationWidget, self.parent())
        if not hasattr(parent_widget, "classes") or not parent_widget.classes:
            return

        # Get current class
        current_class_id = 0  # Default to first class
        current_class_name = "object"

        if hasattr(parent_widget, "class_combo") and parent_widget.class_combo:
            current_index = parent_widget.class_combo.currentIndex()
            if current_index >= 0:
                current_class_id = parent_widget.class_combo.itemData(current_index)
                current_class_name = parent_widget.classes[current_class_id]["name"]

        # Calculate bbox coordinates
        x = min(start.x(), end.x())
        y = min(start.y(), end.y())
        width = abs(end.x() - start.x())
        height = abs(end.y() - start.y())

        if width < 5 or height < 5:  # Minimum size check
            return

        annotation = Annotation(
            id=f"bbox_{len(self.annotations)}",
            class_id=current_class_id,
            class_name=current_class_name,
            coordinates=[float(x), float(y), float(width), float(height)],
            annotation_type="bbox",
        )

        self.annotations.append(annotation)
        self.annotation_created.emit(annotation)
        self.update()

    def _create_polygon_annotation(self, points: List[QPoint]) -> None:
        """Create polygon annotation.

        Args:
            points: List of points for the polygon.
        """
        parent_widget = cast(AnnotationWidget, self.parent())
        if not hasattr(parent_widget, "classes") or not parent_widget.classes:
            return

        current_class_id = 0
        current_class_name = "object"

        if hasattr(parent_widget, "class_combo") and parent_widget.class_combo:
            current_index = parent_widget.class_combo.currentIndex()
            if current_index >= 0:
                current_class_id = parent_widget.class_combo.itemData(current_index)
                current_class_name = parent_widget.classes[current_class_id]["name"]

        # Convert points to image coordinates
        image_points = []
        for point in points:
            image_point = self._widget_to_image_coords(point)
            image_points.extend([image_point.x(), image_point.y()])

        annotation = Annotation(
            id=f"polygon_{len(self.annotations)}",
            class_id=current_class_id,
            class_name=current_class_name,
            coordinates=[float(p) for p in image_points],
            annotation_type="polygon",
        )

        self.annotations.append(annotation)
        self.annotation_created.emit(annotation)
        self.update()

    def _create_point_annotation(self, point: QPoint) -> None:
        """Create point annotation.

        Args:
            point: Point coordinates.
        """
        parent_widget = cast(AnnotationWidget, self.parent())
        if not hasattr(parent_widget, "classes") or not parent_widget.classes:
            return

        current_class_id = 0
        current_class_name = "object"

        if hasattr(parent_widget, "class_combo") and parent_widget.class_combo:
            current_index = parent_widget.class_combo.currentIndex()
            if current_index >= 0:
                current_class_id = parent_widget.class_combo.itemData(current_index)
                current_class_name = parent_widget.classes[current_class_id]["name"]

        annotation = Annotation(
            id=f"point_{len(self.annotations)}",
            class_id=current_class_id,
            class_name=current_class_name,
            coordinates=[float(point.x()), float(point.y())],
            annotation_type="point",
        )

        self.annotations.append(annotation)
        self.annotation_created.emit(annotation)
        self.update()


class AnnotationWidget(QWidget):
    """Data annotation interface."""

    def __init__(self, main_window: Any) -> None:
        """Initialize the annotation widget.

        Args:
            main_window: Reference to the main window.
        """
        super().__init__()
        self.main_window = main_window
        self.logger = get_logger(__name__)

        # State management
        self.current_image_path: Optional[Path] = None
        self.image_list: List[Path] = []
        self.current_index: int = 0
        self.classes: Dict[int, Dict[str, Any]] = {}
        self.config_manager = ConfigManager()

        # UI components
        self.canvas: Optional[AnnotationCanvas] = None
        self.image_list_widget: Optional[QListWidget] = None
        self.annotation_list: Optional[QListWidget] = None
        self.class_combo: Optional[QComboBox] = None
        self.tool_group: Optional[QButtonGroup] = None
        self.zoom_slider: Optional[QSlider] = None
        self.image_counter: Optional[QLabel] = None

        self.init_ui()
        self.setup_shortcuts()
        self.load_classes()

    def init_ui(self) -> None:
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Create main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - Tools and classes
        left_panel = self.create_left_panel()
        main_splitter.addWidget(left_panel)

        # Center panel - Image viewer
        center_panel = self.create_center_panel()
        main_splitter.addWidget(center_panel)

        # Right panel - Annotations and properties
        right_panel = self.create_right_panel()
        main_splitter.addWidget(right_panel)

        # Set splitter proportions
        main_splitter.setSizes([200, 600, 250])

        layout.addWidget(main_splitter)

        # Bottom toolbar
        toolbar = self.create_toolbar()
        layout.addWidget(toolbar)

    def create_left_panel(self) -> QWidget:
        """Create left panel with tools and classes.

        Returns:
            Left panel widget.
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)

        # Tool selection
        tool_group = QGroupBox("Annotation Tools")
        tool_layout = QVBoxLayout(tool_group)

        self.tool_group = QButtonGroup()
        tools = [
            ("bbox", "Bounding Box", "B"),
            ("polygon", "Polygon", "P"),
            ("point", "Point", "O"),
        ]

        for tool_id, tool_name, shortcut in tools:
            btn = QRadioButton(f"{tool_name} ({shortcut})")
            btn.setChecked(tool_id == "bbox")
            btn.toggled.connect(
                lambda checked, t=tool_id: self.set_tool(t) if checked else None
            )
            self.tool_group.addButton(btn)
            tool_layout.addWidget(btn)

        layout.addWidget(tool_group)

        # Class management
        class_group = QGroupBox("Classes")
        class_layout = QVBoxLayout(class_group)

        # Class combo box
        self.class_combo = QComboBox()
        self.class_combo.currentIndexChanged.connect(self.on_class_changed)
        class_layout.addWidget(self.class_combo)

        # Class management buttons
        class_btn_layout = QHBoxLayout()
        add_class_btn = QPushButton("Add")
        add_class_btn.clicked.connect(self.add_class)
        edit_class_btn = QPushButton("Edit")
        edit_class_btn.clicked.connect(self.edit_class)
        delete_class_btn = QPushButton("Delete")
        delete_class_btn.clicked.connect(self.delete_class)

        class_btn_layout.addWidget(add_class_btn)
        class_btn_layout.addWidget(edit_class_btn)
        class_btn_layout.addWidget(delete_class_btn)
        class_layout.addLayout(class_btn_layout)

        layout.addWidget(class_group)

        # Zoom controls
        zoom_group = QGroupBox("Zoom & Navigation")
        zoom_layout = QVBoxLayout(zoom_group)

        zoom_btn_layout = QHBoxLayout()
        zoom_out_btn = QPushButton("Zoom Out")
        zoom_out_btn.clicked.connect(self.zoom_out)
        zoom_in_btn = QPushButton("Zoom In")
        zoom_in_btn.clicked.connect(self.zoom_in)
        fit_btn = QPushButton("Fit to Window")
        fit_btn.clicked.connect(self.fit_to_window)

        zoom_btn_layout.addWidget(zoom_out_btn)
        zoom_btn_layout.addWidget(zoom_in_btn)
        zoom_btn_layout.addWidget(fit_btn)
        zoom_layout.addLayout(zoom_btn_layout)

        # Zoom slider
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 500)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        zoom_layout.addWidget(self.zoom_slider)

        layout.addWidget(zoom_group)
        layout.addStretch()

        return panel

    def create_center_panel(self) -> QWidget:
        """Create center panel with image viewer.

        Returns:
            Center panel widget.
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        # Image counter
        self.image_counter = QLabel("No images loaded")
        self.image_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_counter)

        # Canvas with scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.canvas = AnnotationCanvas(self)
        self.canvas.annotation_created.connect(self.on_annotation_created)
        self.canvas.annotation_selected.connect(self.on_annotation_selected)

        scroll_area.setWidget(self.canvas)
        layout.addWidget(scroll_area)

        return panel

    def create_right_panel(self) -> QWidget:
        """Create right panel with annotations and properties.

        Returns:
            Right panel widget.
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)

        # Image list
        image_group = QGroupBox("Images")
        image_layout = QVBoxLayout(image_group)

        self.image_list_widget = QListWidget()
        self.image_list_widget.itemClicked.connect(self.on_image_selected)
        image_layout.addWidget(self.image_list_widget)

        # Image navigation buttons
        img_btn_layout = QHBoxLayout()
        prev_btn = QPushButton("Previous")
        prev_btn.clicked.connect(self.previous_image)
        next_btn = QPushButton("Next")
        next_btn.clicked.connect(self.next_image)

        img_btn_layout.addWidget(prev_btn)
        img_btn_layout.addWidget(next_btn)
        image_layout.addLayout(img_btn_layout)

        layout.addWidget(image_group)

        # Annotations list
        ann_group = QGroupBox("Annotations")
        ann_layout = QVBoxLayout(ann_group)

        self.annotation_list = QListWidget()
        self.annotation_list.itemClicked.connect(self.on_annotation_list_selected)
        ann_layout.addWidget(self.annotation_list)

        # Annotation management buttons
        ann_btn_layout = QHBoxLayout()
        delete_ann_btn = QPushButton("Delete")
        delete_ann_btn.clicked.connect(self.delete_annotation)
        duplicate_ann_btn = QPushButton("Duplicate")
        duplicate_ann_btn.clicked.connect(self.duplicate_annotation)

        ann_btn_layout.addWidget(delete_ann_btn)
        ann_btn_layout.addWidget(duplicate_ann_btn)
        ann_layout.addLayout(ann_btn_layout)

        layout.addWidget(ann_group)

        return panel

    def create_toolbar(self) -> QWidget:
        """Create bottom toolbar.

        Returns:
            Toolbar widget.
        """
        toolbar = QWidget()
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(0, 5, 0, 5)

        # File actions
        open_btn = QPushButton("Open Images")
        open_btn.clicked.connect(self.open_images)
        save_btn = QPushButton("Save Annotations")
        save_btn.clicked.connect(self.save_annotations)
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self.export_annotations)

        layout.addWidget(open_btn)
        layout.addWidget(save_btn)
        layout.addWidget(export_btn)
        layout.addStretch()

        # Status
        status_label = QLabel("Ready")
        layout.addWidget(status_label)

        return toolbar

    def setup_shortcuts(self) -> None:
        """Set up keyboard shortcuts."""
        # Tool shortcuts
        bbox_shortcut = QShortcut(QKeySequence("B"), self)
        bbox_shortcut.activated.connect(lambda: self.set_tool("bbox"))

        polygon_shortcut = QShortcut(QKeySequence("P"), self)
        polygon_shortcut.activated.connect(lambda: self.set_tool("polygon"))

        point_shortcut = QShortcut(QKeySequence("O"), self)
        point_shortcut.activated.connect(lambda: self.set_tool("point"))

        # Navigation shortcuts
        next_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        next_shortcut.activated.connect(self.next_image)

        prev_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        prev_shortcut.activated.connect(self.previous_image)

        # Save shortcut
        save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        save_shortcut.activated.connect(self.save_annotations)

    def load_classes(self) -> None:
        """Load class definitions."""
        # Load from config or create defaults
        default_classes = {
            0: {"name": "object", "color": "#FF0000"},
            1: {"name": "person", "color": "#00FF00"},
            2: {"name": "car", "color": "#0000FF"},
        }

        loaded_classes = self.config_manager.get("annotation.classes", default_classes)

        # Check if loaded_classes is actually class definitions or config data
        if isinstance(loaded_classes, dict):
            # Check if this looks like class definitions (has numeric keys with dict values)
            has_class_definitions = any(
                isinstance(k, int) and isinstance(v, dict) and "name" in v
                for k, v in loaded_classes.items()
            )

            if has_class_definitions:
                self.classes = loaded_classes
            else:
                # This is config data, not class definitions, use defaults
                self.classes = default_classes
        elif isinstance(loaded_classes, list):
            # Convert list to dictionary format
            self.classes = {}
            for i, class_info in enumerate(loaded_classes):
                if isinstance(class_info, dict):
                    self.classes[i] = class_info
                else:
                    # Handle string format
                    self.classes[i] = {"name": str(class_info), "color": "#FF0000"}
        else:
            self.classes = default_classes

        self.update_class_combo()

    def update_class_combo(self) -> None:
        """Update the class combo box."""
        if not self.class_combo:
            return

        self.class_combo.clear()

        # Handle both list and dictionary formats
        if isinstance(self.classes, list):
            for i, class_info in enumerate(self.classes):
                if isinstance(class_info, dict):
                    self.class_combo.addItem(class_info["name"], i)
                else:
                    self.class_combo.addItem(str(class_info), i)
        else:
            for class_id, class_info in self.classes.items():
                self.class_combo.addItem(class_info["name"], class_id)

    def set_tool(self, tool: str) -> None:
        """Set the current annotation tool.

        Args:
            tool: Tool type ('bbox', 'polygon', 'point').
        """
        if self.canvas:
            self.canvas.set_tool(tool)

    def open_images(self) -> None:
        """Open image files."""
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
            self.load_current_image()
            self.update_image_list()

    def load_current_image(self) -> None:
        """Load the current image."""
        if not self.image_list or self.current_index >= len(self.image_list):
            return

        image_path = self.image_list[self.current_index]
        if self.canvas and self.canvas.set_image(image_path):
            self.current_image_path = image_path
            self.update_image_counter()
            self.load_annotations()

    def update_image_counter(self) -> None:
        """Update the image counter label."""
        if self.image_counter and self.image_list:
            self.image_counter.setText(
                f"Image {self.current_index + 1} of {len(self.image_list)}"
            )

    def update_image_list(self) -> None:
        """Update the image list widget."""
        if not self.image_list_widget:
            return

        self.image_list_widget.clear()
        for i, image_path in enumerate(self.image_list):
            item = QListWidgetItem(image_path.name)
            item.setData(Qt.ItemDataRole.UserRole, i)
            if i == self.current_index:
                item.setBackground(QColor("#4CAF50"))
            self.image_list_widget.addItem(item)

    def next_image(self) -> None:
        """Go to next image."""
        if self.image_list and self.current_index < len(self.image_list) - 1:
            self.save_current_annotations()
            self.current_index += 1
            self.load_current_image()
            self.update_image_list()

    def previous_image(self) -> None:
        """Go to previous image."""
        if self.current_index > 0:
            self.save_current_annotations()
            self.current_index -= 1
            self.load_current_image()
            self.update_image_list()

    def on_image_selected(self, item: QListWidgetItem) -> None:
        """Handle image selection.

        Args:
            item: Selected list item.
        """
        index = item.data(Qt.ItemDataRole.UserRole)
        if index != self.current_index:
            self.save_current_annotations()
            self.current_index = index
            self.load_current_image()
            self.update_image_list()

    def on_class_changed(self) -> None:
        """Handle class selection change."""
        # Update canvas if needed
        pass

    def on_annotation_created(self, annotation: Annotation) -> None:
        """Handle new annotation creation.

        Args:
            annotation: Created annotation.
        """
        self.update_annotation_list()

    def on_annotation_selected(self, index: int) -> None:
        """Handle annotation selection.

        Args:
            index: Selected annotation index.
        """
        if self.annotation_list and 0 <= index < self.annotation_list.count():
            self.annotation_list.setCurrentRow(index)

    def on_annotation_list_selected(self, item: QListWidgetItem) -> None:
        """Handle annotation list selection.

        Args:
            item: Selected list item.
        """
        if self.canvas and self.annotation_list:
            index = self.annotation_list.row(item)
            self.canvas.selected_annotation = index
            self.canvas.update()

    def update_annotation_list(self) -> None:
        """Update the annotation list widget."""
        if not self.annotation_list or not self.canvas:
            return

        self.annotation_list.clear()
        for i, annotation in enumerate(self.canvas.annotations):
            item = QListWidgetItem(
                f"{annotation.class_name} ({annotation.annotation_type})"
            )
            item.setData(Qt.ItemDataRole.UserRole, i)
            self.annotation_list.addItem(item)

    def delete_annotation(self) -> None:
        """Delete selected annotation."""
        if not self.canvas or not self.annotation_list:
            return

        current_row = self.annotation_list.currentRow()
        if 0 <= current_row < len(self.canvas.annotations):
            del self.canvas.annotations[current_row]
            self.canvas.selected_annotation = -1
            self.canvas.update()
            self.update_annotation_list()

    def duplicate_annotation(self) -> None:
        """Duplicate selected annotation."""
        if not self.canvas or not self.annotation_list:
            return

        current_row = self.annotation_list.currentRow()
        if 0 <= current_row < len(self.canvas.annotations):
            original = self.canvas.annotations[current_row]
            # Create duplicate with slight offset
            duplicate = Annotation(
                id=f"{original.id}_copy",
                class_id=original.class_id,
                class_name=original.class_name,
                coordinates=original.coordinates.copy(),
                annotation_type=original.annotation_type,
            )
            self.canvas.annotations.append(duplicate)
            self.canvas.update()
            self.update_annotation_list()

    def add_class(self) -> None:
        """Add a new class."""
        # Simple implementation - could be enhanced with dialog
        new_id = max(self.classes.keys()) + 1 if self.classes else 0
        self.classes[new_id] = {"name": f"class_{new_id}", "color": "#FF0000"}
        self.update_class_combo()

    def edit_class(self) -> None:
        """Edit selected class."""
        # Placeholder - could be enhanced with dialog
        pass

    def delete_class(self) -> None:
        """Delete selected class."""
        if not self.class_combo:
            return

        current_index = self.class_combo.currentIndex()
        if current_index >= 0:
            class_id = self.class_combo.itemData(current_index)
            if class_id in self.classes:
                del self.classes[class_id]
                self.update_class_combo()

    def zoom_in(self) -> None:
        """Zoom in."""
        if self.canvas and self.zoom_slider:
            self.canvas.set_zoom(self.canvas.zoom_factor * 1.2)
            self.zoom_slider.setValue(int(self.canvas.zoom_factor * 100))

    def zoom_out(self) -> None:
        """Zoom out."""
        if self.canvas and self.zoom_slider:
            self.canvas.set_zoom(self.canvas.zoom_factor / 1.2)
            self.zoom_slider.setValue(int(self.canvas.zoom_factor * 100))

    def fit_to_window(self) -> None:
        """Fit image to window."""
        if self.canvas and self.zoom_slider:
            self.canvas.fit_to_window()
            self.zoom_slider.setValue(int(self.canvas.zoom_factor * 100))

    def on_zoom_changed(self, value: int) -> None:
        """Handle zoom slider change.

        Args:
            value: New zoom value.
        """
        if self.canvas:
            self.canvas.set_zoom(value / 100.0)

    def save_current_annotations(self) -> None:
        """Save annotations for current image."""
        if not self.current_image_path or not self.canvas:
            return

        # Save annotations to JSON file
        annotations_file = self.current_image_path.with_suffix(".json")
        annotations_data = {
            "image_path": str(self.current_image_path),
            "annotations": [
                self._annotation_to_dict(ann) for ann in self.canvas.annotations
            ],
        }

        try:
            with open(annotations_file, "w", encoding="utf-8") as f:
                json.dump(annotations_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save annotations: {e}")

    def load_annotations(self) -> None:
        """Load annotations for current image."""
        if not self.current_image_path or not self.canvas:
            return

        annotations_file = self.current_image_path.with_suffix(".json")
        if annotations_file.exists():
            try:
                with open(annotations_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.canvas.annotations = [
                        self._dict_to_annotation(ann)
                        for ann in data.get("annotations", [])
                    ]
                    self.canvas.update()
                    self.update_annotation_list()
            except Exception as e:
                self.logger.error(f"Failed to load annotations: {e}")

    def _annotation_to_dict(self, annotation: Annotation) -> Dict[str, Any]:
        """Convert annotation to dictionary.

        Args:
            annotation: Annotation to convert.

        Returns:
            Dictionary representation of annotation.
        """
        return {
            "id": annotation.id,
            "class_id": annotation.class_id,
            "class_name": annotation.class_name,
            "coordinates": annotation.coordinates,
            "annotation_type": annotation.annotation_type,
            "confidence": annotation.confidence,
            "created_by": annotation.created_by,
            "created_at": annotation.created_at,
            "modified_at": annotation.modified_at,
        }

    def _dict_to_annotation(self, data: Dict[str, Any]) -> Annotation:
        """Convert dictionary to annotation.

        Args:
            data: Dictionary data.

        Returns:
            Annotation object.
        """
        return Annotation(
            id=data.get("id", ""),
            class_id=data.get("class_id", 0),
            class_name=data.get("class_name", "object"),
            coordinates=data.get("coordinates", []),
            annotation_type=data.get("annotation_type", "bbox"),
            confidence=data.get("confidence", 1.0),
            created_by=data.get("created_by", "user"),
            created_at=data.get("created_at", ""),
            modified_at=data.get("modified_at", ""),
        )

    def save_annotations(self) -> None:
        """Save all annotations."""
        self.save_current_annotations()
        QMessageBox.information(self, "Save", "Annotations saved successfully!")

    def export_annotations(self) -> None:
        """Export annotations to COCO format."""
        if not self.image_list:
            QMessageBox.warning(self, "Export", "No images loaded!")
            return

        dialog = QFileDialog()
        export_path, _ = dialog.getSaveFileName(
            self,
            "Export Annotations",
            "",
            "COCO JSON (*.json)",
        )

        if export_path:
            try:
                # Simple COCO export - could be enhanced
                coco_data: Dict[str, Any] = {
                    "images": [],
                    "annotations": [],
                    "categories": [],
                }

                # Add categories
                for class_id, class_info in self.classes.items():
                    coco_data["categories"].append(
                        {
                            "id": class_id,
                            "name": class_info["name"],
                        }
                    )

                # Add images and annotations
                annotation_id = 1
                for i, image_path in enumerate(self.image_list):
                    # Load annotations for this image
                    annotations_file = image_path.with_suffix(".json")
                    if annotations_file.exists():
                        with open(annotations_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            annotations = [
                                self._dict_to_annotation(ann)
                                for ann in data.get("annotations", [])
                            ]

                            # Add image
                            coco_data["images"].append(
                                {
                                    "id": i + 1,
                                    "file_name": image_path.name,
                                    # Could be enhanced to get actual dimensions
                                    "width": 800,
                                    "height": 600,
                                }
                            )

                            # Add annotations
                            for ann in annotations:
                                if (
                                    ann.annotation_type == "bbox"
                                    and len(ann.coordinates) >= 4
                                ):
                                    coco_data["annotations"].append(
                                        {
                                            "id": annotation_id,
                                            "image_id": i + 1,
                                            "category_id": ann.class_id,
                                            "bbox": ann.coordinates[:4],
                                            "area": ann.coordinates[2]
                                            * ann.coordinates[3],
                                            "iscrowd": 0,
                                        }
                                    )
                                    annotation_id += 1

                with open(export_path, "w", encoding="utf-8") as f:
                    json.dump(coco_data, f, indent=2, ensure_ascii=False)

                QMessageBox.information(
                    self, "Export", "Annotations exported successfully!"
                )

            except Exception as e:
                self.logger.error(f"Failed to export annotations: {e}")
                QMessageBox.critical(self, "Export Error", f"Failed to export: {e}")

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.canvas:
            self.save_current_annotations()
