#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Widget.

GUI component for model training interface.
"""

import os
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class TrainingThread(QThread):
    """
    Thread for running training in the background.

    Attributes:
        progress_update: Signal for epoch progress.
        status_update: Signal for status messages.
        training_finished: Signal for training completion.
        training_failed: Signal for training failure.
    """

    progress_update = Signal(int, int)
    status_update = Signal(str)
    training_finished = Signal(object)
    training_failed = Signal(str)

    def __init__(
        self,
        trainer: Any,
        data_yaml_path: str,
        training_config: Any,
    ) -> None:
        """
        Initialize the training thread.

        Args:
            trainer: YOLOTrainer instance.
            data_yaml_path: Path to data.yaml.
            training_config: Training configuration object.
        """
        super().__init__()
        self.trainer = trainer
        self.data_yaml_path = data_yaml_path
        self.training_config = training_config

    def run(self) -> None:
        """
        Run the training process in a separate thread.
        """
        try:
            self.status_update.emit("Training started...")
            results = self.trainer.train_model(
                self.data_yaml_path,
                self.training_config,
                callbacks={"on_epoch_end": self.on_epoch_end},
            )
            self.training_finished.emit(results)
        except Exception as e:
            self.training_failed.emit(str(e))

    def on_epoch_end(self, epoch: int, total_epochs: int) -> None:
        """
        Emit progress update at the end of each epoch.

        Args:
            epoch: Current epoch.
            total_epochs: Total number of epochs.
        """
        self.progress_update.emit(epoch, total_epochs)


class TrainingWidget(QWidget):
    """
    Model training interface.

    Provides model selection, parameter input, dataset selection,
    training progress, and results visualization.
    """

    def __init__(self, main_window: Any) -> None:
        """
        Initialize the TrainingWidget.

        Args:
            main_window: The main window instance.
        """
        super().__init__()
        self.main_window = main_window
        self.trainer = getattr(main_window, "trainer", None)
        self.training_thread: Optional[TrainingThread] = None
        self.init_ui()

    def init_ui(self) -> None:
        """
        Initialize the user interface.
        """
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        header = QLabel("Model Training")
        header.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(header)

        param_group = QGroupBox("Model & Training Parameters")
        param_group.setObjectName("dashboard-group")
        param_layout = QFormLayout(param_group)
        param_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(False)
        self.model_combo.setMinimumWidth(180)
        self.model_combo.addItems(self.get_supported_models())
        param_layout.addRow("Model:", self.model_combo)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(100)
        param_layout.addRow("Epochs:", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 256)
        self.batch_spin.setValue(16)
        param_layout.addRow("Batch Size:", self.batch_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(5)
        self.lr_spin.setRange(0.00001, 1.0)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.01)
        param_layout.addRow("Learning Rate:", self.lr_spin)

        self.img_size_spin = QSpinBox()
        self.img_size_spin.setRange(128, 2048)
        self.img_size_spin.setSingleStep(32)
        self.img_size_spin.setValue(640)
        param_layout.addRow("Image Size:", self.img_size_spin)

        layout.addWidget(param_group)

        dataset_group = QGroupBox("Dataset Selection")
        dataset_group.setObjectName("dashboard-group")
        dataset_layout = QHBoxLayout(dataset_group)
        self.dataset_edit = QLineEdit()
        self.dataset_edit.setPlaceholderText("Path to data.yaml or dataset folder")
        dataset_layout.addWidget(self.dataset_edit)
        browse_btn = QPushButton("Browse")
        browse_btn.setObjectName("secondary-btn")
        browse_btn.clicked.connect(self.browse_dataset)
        dataset_layout.addWidget(browse_btn)
        layout.addWidget(dataset_group)

        controls_group = QGroupBox("Training Controls")
        controls_group.setObjectName("dashboard-group")
        controls_layout = QHBoxLayout(controls_group)
        self.start_btn = QPushButton("Start Training")
        self.start_btn.setObjectName("quick-action-btn")
        self.start_btn.clicked.connect(self.start_training)
        controls_layout.addWidget(self.start_btn)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("secondary-btn")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_training)
        controls_layout.addWidget(self.stop_btn)
        layout.addWidget(controls_group)

        progress_group = QGroupBox("Training Progress")
        progress_group.setObjectName("dashboard-group")
        progress_layout = QVBoxLayout(progress_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setFixedHeight(80)
        progress_layout.addWidget(self.status_text)
        layout.addWidget(progress_group)

        results_group = QGroupBox("Results Visualization")
        results_group.setObjectName("dashboard-group")
        results_layout = QVBoxLayout(results_group)
        self.results_label = QLabel("No results yet.")
        self.results_label.setWordWrap(True)
        results_layout.addWidget(self.results_label)
        layout.addWidget(results_group)

        layout.addStretch()

    def get_supported_models(self) -> list[str]:
        """
        Get the list of supported YOLO models.

        Returns:
            List of model names.
        """
        if self.trainer and hasattr(self.trainer, "get_supported_models"):
            return self.trainer.get_supported_models()
        return ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]

    def browse_dataset(self) -> None:
        """
        Open a file dialog to select a data.yaml file or dataset folder.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select data.yaml",
            os.getcwd(),
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if file_path:
            self.dataset_edit.setText(file_path)
        else:
            dir_path = QFileDialog.getExistingDirectory(
                self, "Select Dataset Folder", os.getcwd()
            )
            if dir_path:
                self.dataset_edit.setText(dir_path)

    def start_training(self) -> None:
        """
        Start the training process in a background thread.
        """
        if not self.trainer:
            QMessageBox.critical(self, "Error", "YOLOTrainer not available.")
            return
        data_path = self.dataset_edit.text().strip()
        if not data_path or not Path(data_path).exists():
            QMessageBox.warning(
                self,
                "Dataset Required",
                "Please select a valid data.yaml or dataset folder.",
            )
            return
        model_name = self.model_combo.currentText()
        epochs = self.epochs_spin.value()
        batch_size = self.batch_spin.value()
        lr = self.lr_spin.value()
        img_size = self.img_size_spin.value()
        try:
            training_config = self.trainer.prepare_training_config(
                model_name=model_name,
                data_yaml_path=data_path,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=lr,
                image_size=img_size,
            )
        except Exception as e:
            QMessageBox.critical(self, "Config Error", str(e))
            return
        self.training_thread = TrainingThread(self.trainer, data_path, training_config)
        self.training_thread.progress_update.connect(self.on_progress_update)
        self.training_thread.status_update.connect(self.on_status_update)
        self.training_thread.training_finished.connect(self.on_training_finished)
        self.training_thread.training_failed.connect(self.on_training_failed)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.status_text.clear()
        self.results_label.setText("Training in progress...")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.training_thread.start()

    @Slot(int, int)
    def on_progress_update(self, epoch: int, total_epochs: int) -> None:
        """
        Update the progress bar and status text for each epoch.

        Args:
            epoch: Current epoch.
            total_epochs: Total number of epochs.
        """
        percent = int((epoch / max(1, total_epochs)) * 100)
        self.progress_bar.setValue(percent)
        self.status_text.append(f"Epoch {epoch}/{total_epochs}")

    @Slot(str)
    def on_status_update(self, msg: str) -> None:
        """
        Append a status message to the status text area.

        Args:
            msg: Status message.
        """
        self.status_text.append(msg)

    @Slot(object)
    def on_training_finished(self, results: Any) -> None:
        """
        Handle training completion and display results.

        Args:
            results: Training results object.
        """
        self.progress_bar.setVisible(False)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if results:
            summary = (
                f"<b>Training Complete!</b><br>"
                f"Best mAP50: {getattr(results, 'best_map50', 'N/A')}<br>"
                f"Best Precision: {getattr(results, 'best_precision', 'N/A')}<br>"
                f"Best Recall: {getattr(results, 'best_recall', 'N/A')}<br>"
                f"Final Loss: {getattr(results, 'final_loss', 'N/A')}<br>"
                f"Training Time: {getattr(results, 'training_time', 'N/A')}s<br>"
                f"Model Path: {getattr(results, 'model_path', 'N/A')}"
            )
            self.results_label.setText(summary)
        else:
            self.results_label.setText("Training finished, but no results available.")

    @Slot(str)
    def on_training_failed(self, error_msg: str) -> None:
        """
        Handle training failure and display error message.

        Args:
            error_msg: Error message string.
        """
        self.progress_bar.setVisible(False)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.results_label.setText(f"<b>Training failed:</b> {error_msg}")
        QMessageBox.critical(self, "Training Failed", error_msg)

    def stop_training(self) -> None:
        """
        Stop the training process if running.
        """
        if self.training_thread and self.training_thread.isRunning():
            if hasattr(self.trainer, "stop_training"):
                self.trainer.stop_training()
            self.training_thread.terminate()
            self.training_thread.wait()
            self.progress_bar.setVisible(False)
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status_text.append("Training stopped by user.")
            self.results_label.setText("Training stopped.")

    def cleanup(self) -> None:
        """
        Cleanup resources and stop any running training thread.
        """
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.terminate()
            self.training_thread.wait()
        self.training_thread = None
