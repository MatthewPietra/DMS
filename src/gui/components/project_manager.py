#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project Manager Widget.

GUI component for managing DMS projects.
"""

import json
import shutil
from pathlib import Path
from typing import Any, List, Optional

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...annotation.coco_exporter import COCOExporter


class ProjectManagerWidget(QWidget):
    """Project management interface."""

    def __init__(self, main_window: Any) -> None:
        """Initialize the project manager widget.

        Args:
            main_window: The main window instance.
        """
        super().__init__()
        self.main_window = main_window
        self.projects_dir = Path("data/projects")
        self.init_ui()
        self.refresh_project_list()

    def init_ui(self) -> None:
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        header = QLabel("Project Manager")
        header.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(header)

        # Project list
        self.project_list = QListWidget()
        self.project_list.itemDoubleClicked.connect(self.open_project)
        layout.addWidget(self.project_list)

        # Action buttons
        btn_layout = QHBoxLayout()
        new_btn = QPushButton("New Project")
        new_btn.clicked.connect(self.new_project)
        btn_layout.addWidget(new_btn)

        open_btn = QPushButton("Open Project")
        open_btn.clicked.connect(self.open_project)
        btn_layout.addWidget(open_btn)

        settings_btn = QPushButton("Project Settings")
        settings_btn.clicked.connect(self.project_settings)
        btn_layout.addWidget(settings_btn)

        import_btn = QPushButton("Import Project")
        import_btn.clicked.connect(self.import_project)
        btn_layout.addWidget(import_btn)

        export_btn = QPushButton("Export Project")
        export_btn.clicked.connect(self.export_project)
        btn_layout.addWidget(export_btn)

        layout.addLayout(btn_layout)
        layout.addStretch()

    def refresh_project_list(self) -> None:
        """Refresh the project list display."""
        self.project_list.clear()
        if not self.projects_dir.exists():
            self.projects_dir.mkdir(parents=True, exist_ok=True)
        projects_found = False
        for project_dir in sorted(self.projects_dir.iterdir()):
            if project_dir.is_dir():
                projects_found = True
                config_file = project_dir / "config.json"
                if config_file.exists():
                    try:
                        with open(config_file, "r", encoding="utf-8") as f:
                            config = json.load(f)
                        name = config.get("name", project_dir.name)
                        self.project_list.addItem(f"{name} [{project_dir.name}]")
                    except Exception:
                        # If config file is corrupted, just show the directory name
                        self.project_list.addItem(project_dir.name)
                else:
                    self.project_list.addItem(project_dir.name)
        if not projects_found:
            self.project_list.addItem(
                "No projects found. Create a new project to get started."
            )

    def showEvent(self, event: Any) -> None:
        """Handle show event to refresh project list when widget becomes visible.

        Args:
            event: The show event.
        """
        super().showEvent(event)
        self.refresh_project_list()

    def new_project(self) -> None:
        """Create a new project."""
        dialog = ProjectDialog(self, title="Create New Project")
        if dialog.exec() == QDialog.DialogCode.Accepted:
            name, description, classes = dialog.get_data()

            # Validate input
            if not name.strip():
                QMessageBox.warning(
                    self, "Validation Error", "Project name cannot be empty."
                )
                return

            if not classes:
                QMessageBox.warning(
                    self, "Validation Error", "At least one class must be specified."
                )
                return

            project_path = self.projects_dir / name
            if project_path.exists():
                QMessageBox.warning(self, "Error", f"Project '{name}' already exists.")
                return

            try:
                # Create project directory structure
                project_path.mkdir(parents=True)
                (project_path / "images").mkdir()
                (project_path / "annotations").mkdir()
                (project_path / "models").mkdir()
                (project_path / "exports").mkdir()
                (project_path / "logs").mkdir()

                # Create project configuration
                config = {
                    "name": name,
                    "description": description,
                    "classes": classes,
                    "created_at": str(Path.cwd()),
                    "version": "1.0.0",
                }

                # Save config file
                with open(project_path / "config.json", "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2)

                # Save classes file
                with open(project_path / "classes.txt", "w", encoding="utf-8") as f:
                    for c in classes:
                        f.write(f"{c}\n")

                self.refresh_project_list()
                QMessageBox.information(
                    self, "Success", f"Project '{name}' created successfully."
                )

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to create project: {e}")
                # Clean up partial creation
                if project_path.exists():
                    try:
                        shutil.rmtree(project_path)
                    except Exception:  # nosec B110
                        pass

    def open_project(self) -> None:
        """Open an existing project."""
        item = self.project_list.currentItem()
        if not item:
            QMessageBox.information(
                self, "Select Project", "Please select a project to open."
            )
            return
        project_name = item.text().split("[")[-1].rstrip("]")
        project_path = self.projects_dir / project_name
        if not project_path.exists():
            QMessageBox.warning(self, "Error", f"Project '{project_name}' not found.")
            return
        # Set as current project in main window using the new method
        self.main_window.set_current_project(str(project_path))
        QMessageBox.information(
            self, "Project Opened", f"Opened project: {project_name}"
        )

    def get_project_info(self, project_name: str) -> dict[str, Any]:
        """Get information about a project.

        Args:
            project_name: Name of the project directory.

        Returns:
            Dictionary containing project information.
        """
        project_path = self.projects_dir / project_name
        config_file = project_path / "config.json"

        if not config_file.exists():
            return {
                "name": project_name,
                "description": "",
                "classes": [],
                "error": "Config not found",
            }

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
            return dict(config)
        except Exception as e:
            return {
                "name": project_name,
                "description": "",
                "classes": [],
                "error": f"Config error: {e}",
            }

    def project_settings(self) -> None:
        """Edit project settings."""
        item = self.project_list.currentItem()
        if not item:
            QMessageBox.information(
                self, "Select Project", "Please select a project to edit settings."
            )
            return

        project_name = item.text().split("[")[-1].rstrip("]")
        if project_name == "No projects found. Create a new project to get started.":
            return

        project_path = self.projects_dir / project_name
        config_file = project_path / "config.json"

        if not config_file.exists():
            QMessageBox.warning(self, "Error", "Project config not found.")
            return

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read project config: {e}")
            return

        dialog = ProjectDialog(
            self,
            title="Project Settings",
            name=config.get("name", project_name),
            description=config.get("description", ""),
            classes=config.get("classes", []),
        )

        if dialog.exec() == QDialog.DialogCode.Accepted:
            name, description, classes = dialog.get_data()
            config["name"] = name
            config["description"] = description
            config["classes"] = classes

            try:
                with open(config_file, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2)
                with open(project_path / "classes.txt", "w", encoding="utf-8") as f:
                    for c in classes:
                        f.write(f"{c}\n")
                self.refresh_project_list()
                QMessageBox.information(
                    self, "Success", "Project settings updated successfully."
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to save project settings: {e}"
                )

    def import_project(self) -> None:
        """Import a project from a config file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Project (config.json)",
            str(self.projects_dir),
            "JSON Files (*.json)",
        )
        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as e:
            QMessageBox.critical(
                self, "Import Error", f"Failed to read config file: {e}"
            )
            return

        name = config.get("name")
        if not name:
            QMessageBox.warning(
                self, "Import Error", "Invalid project config: missing project name."
            )
            return

        project_path = self.projects_dir / name
        if project_path.exists():
            QMessageBox.warning(
                self, "Import Error", f"Project '{name}' already exists."
            )
            return

        try:
            project_path.mkdir(parents=True)
            with open(project_path / "config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            self.refresh_project_list()
            QMessageBox.information(
                self, "Import Success", f"Project '{name}' imported successfully."
            )
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import project: {e}")

    def export_project(self) -> None:
        """Export a project in COCO format."""
        item = self.project_list.currentItem()
        if not item:
            QMessageBox.information(
                self, "Select Project", "Please select a project to export."
            )
            return

        project_name = item.text().split("[")[-1].rstrip("]")
        if project_name == "No projects found. Create a new project to get started.":
            return

        project_path = self.projects_dir / project_name
        if not project_path.exists():
            QMessageBox.warning(
                self, "Export Error", f"Project '{project_name}' not found."
            )
            return

        export_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Export Directory",
            str(Path.cwd()),
        )
        if not export_dir:
            return

        try:
            exporter = COCOExporter()
            ok = exporter.export_dataset(
                project_path,
                Path(export_dir),
                export_format="COCO",
                include_images=True,
            )
            if ok:
                QMessageBox.information(
                    self,
                    "Export Complete",
                    f"Project exported to {export_dir}",
                )
            else:
                QMessageBox.warning(self, "Export Failed", "Project export failed.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export project: {e}")

    def cleanup(self) -> None:
        """Cleanup resources."""
        pass


class ProjectDialog(QDialog):
    """Dialog for creating and editing projects."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        title: str = "",
        name: str = "",
        description: str = "",
        classes: Optional[List[str]] = None,
    ) -> None:
        """Initialize the project dialog.

        Args:
            parent: Parent widget.
            title: Dialog title.
            name: Project name.
            description: Project description.
            classes: List of class names.
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(400)

        # Create form fields
        self.name_edit = QLineEdit(name)
        self.name_edit.setPlaceholderText("Enter project name")
        self.desc_edit = QLineEdit(description)
        self.desc_edit.setPlaceholderText("Enter project description")
        self.classes_edit = QLineEdit(", ".join(classes) if classes else "")
        self.classes_edit.setPlaceholderText(
            "Enter classes separated by commas (e.g., person, car, bike)"
        )

        # Create layout
        layout = QFormLayout(self)
        layout.addRow("Project Name:", self.name_edit)
        layout.addRow("Description:", self.desc_edit)
        layout.addRow("Classes:", self.classes_edit)

        # Add help text
        help_label = QLabel(
            "Classes are the object types you want to detect in your project."
        )
        help_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addRow("", help_label)

        # Add buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Set focus to name field
        self.name_edit.setFocus()

    def validate_and_accept(self) -> None:
        """Validate input before accepting the dialog."""
        name = self.name_edit.text().strip()
        classes_text = self.classes_edit.text().strip()

        if not name:
            QMessageBox.warning(
                self, "Validation Error", "Project name cannot be empty."
            )
            self.name_edit.setFocus()
            return

        if not classes_text:
            QMessageBox.warning(
                self, "Validation Error", "At least one class must be specified."
            )
            self.classes_edit.setFocus()
            return

        # Validate class names
        classes = [c.strip() for c in classes_text.split(",") if c.strip()]
        if not classes:
            QMessageBox.warning(
                self, "Validation Error", "At least one valid class must be specified."
            )
            self.classes_edit.setFocus()
            return

        self.accept()

    def get_data(self) -> tuple[str, str, List[str]]:
        """Get the dialog data.

        Returns:
            Tuple of (name, description, classes).
        """
        name = self.name_edit.text().strip()
        description = self.desc_edit.text().strip()
        classes = [c.strip() for c in self.classes_edit.text().split(",") if c.strip()]
        return name, description, classes
