#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DMS GUI Icons.

Icon management for the DMS GUI application.
Provides a centralized way to manage and access icons.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QIcon, QPainter, QPixmap


class IconManager:
    """
    Manages icons for the DMS GUI application.

    Provides a centralized way to access icons with fallback
    to built-in icons when custom icons are not available.
    """

    # Icon cache
    _icon_cache: Dict[str, QIcon] = {}

    # Built-in icon definitions (using Unicode symbols)
    _builtin_icons: Dict[str, str] = {
        "app": "🎯",
        "logo": "🎯",
        "dashboard": "📊",
        "projects": "📁",
        "capture": "📷",
        "annotation": "✏️",
        "training": "🧠",
        "monitor": "📈",
        "settings": "⚙️",
        "new": "➕",
        "open": "📂",
        "save": "💾",
        "edit": "✏️",
        "delete": "🗑️",
        "refresh": "🔄",
        "play": "▶️",
        "pause": "⏸️",
        "stop": "⏹️",
        "camera": "📷",
        "brain": "🧠",
        "folder": "📁",
        "image": "🖼️",
        "video": "🎥",
        "model": "🤖",
        "chart": "📊",
        "gear": "⚙️",
        "info": "ℹ️",
        "warning": "⚠️",
        "error": "❌",
        "success": "✅",
        "help": "❓",
        "close": "✕",
        "minimize": "−",
        "maximize": "□",
        "restore": "❐",
        "export": "📤",
        "import": "📥",
        "download": "⬇️",
        "upload": "⬆️",
        "search": "🔍",
        "filter": "🔧",
        "sort": "↕️",
        "zoom_in": "🔍+",
        "zoom_out": "🔍-",
        "fit": "🔍=",
        "undo": "↶",
        "redo": "↷",
        "cut": "✂️",
        "copy": "📋",
        "paste": "📋",
        "select_all": "☑️",
        "clear": "🗑️",
        "reset": "🔄",
        "apply": "✓",
        "cancel": "✗",
        "ok": "✓",
        "yes": "✓",
        "no": "✗",
        "back": "←",
        "forward": "→",
        "up": "↑",
        "down": "↓",
        "home": "🏠",
        "end": "🏁",
        "first": "⏮️",
        "last": "⏭️",
        "previous": "⏪",
        "next": "⏩",
        "step_back": "⏪",
        "step_forward": "⏩",
        "record": "🔴",
        "live": "🔴",
        "offline": "⚫",
        "online": "🟢",
        "connected": "🔗",
        "disconnected": "🔌",
        "sync": "🔄",
        "sync_of": "⏸️",
        "auto": "🤖",
        "manual": "👤",
        "batch": "📦",
        "single": "1️⃣",
        "multiple": "🔢",
        "all": "☑️",
        "none": "☐",
        "partial": "☑️",
        "enabled": "✅",
        "disabled": "❌",
        "visible": "👁️",
        "hidden": "🙈",
        "locked": "🔒",
        "unlocked": "🔓",
        "public": "🌐",
        "private": "🔒",
        "shared": "🤝",
        "local": "💻",
        "cloud": "☁️",
        "database": "🗄️",
        "server": "🖥️",
        "client": "💻",
        "network": "🌐",
        "wifi": "📶",
        "bluetooth": "📡",
        "usb": "🔌",
        "ethernet": "🌐",
        "gpu": "🎮",
        "cpu": "🖥️",
        "ram": "💾",
        "storage": "💿",
        "ssd": "💿",
        "hdd": "💾",
        "optical": "💿",
        "flash": "💾",
        "battery": "🔋",
        "power": "⚡",
        "temperature": "🌡️",
        "fan": "💨",
        "clock": "🕐",
        "calendar": "📅",
        "timer": "⏱️",
        "stopwatch": "⏱️",
        "alarm": "⏰",
        "schedule": "📅",
        "reminder": "🔔",
        "notification": "🔔",
        "message": "💬",
        "email": "📧",
        "phone": "📞",
        "mobile": "📱",
        "tablet": "📱",
        "laptop": "💻",
        "desktop": "🖥️",
        "monitor_display": "🖥️",
        "keyboard": "⌨️",
        "mouse": "🖱️",
        "printer": "🖨️",
        "scanner": "📷",
        "speaker": "🔊",
        "headphones": "🎧",
        "microphone": "🎤",
        "camera_device": "📷",
        "webcam": "📷",
        "projector": "📽️",
        "tv": "📺",
        "radio": "📻",
        "game": "🎮",
        "music": "🎵",
        "video_file": "🎥",
        "photo_file": "🖼️",
        "document": "📄",
        "text": "📝",
        "code": "💻",
        "script": "📜",
        "config": "⚙️",
        "log": "📋",
        "report": "📊",
        "analytics": "📈",
        "statistics": "📊",
        "metrics": "📊",
        "performance": "⚡",
        "speed": "🏃",
        "efficiency": "⚡",
        "quality": "⭐",
        "accuracy": "🎯",
        "precision": "🎯",
        "recall": "🎯",
        "f1": "🎯",
        "map": "🎯",
        "iou": "🎯",
        "confidence": "🎯",
        "threshold": "🎯",
        "epoch": "🔄",
        "iteration": "🔄",
        "batch_process": "📦",
        "sample": "📋",
        "dataset": "📊",
        "training_process": "🏋️",
        "validation": "✅",
        "testing": "🧪",
        "inference": "🔮",
        "prediction": "🔮",
        "detection": "🎯",
        "classification": "🏷️",
        "segmentation": "✂️",
        "tracking": "🎯",
        "recognition": "👁️",
        "identification": "🆔",
        "verification": "✅",
        "authentication": "🔐",
        "authorization": "🔑",
        "permission": "🔐",
        "role": "👤",
        "user": "👤",
        "admin": "👑",
        "guest": "👤",
        "anonymous": "👤",
        "profile": "👤",
        "account": "👤",
        "login": "🔑",
        "logout": "🚪",
        "register": "📝",
        "signup": "📝",
        "signin": "🔑",
        "signout": "🚪",
        "password": "🔐",
        "username": "👤",
        "email_address": "📧",
        "avatar": "👤",
        "picture": "🖼️",
        "photo_image": "📷",
        "image_file": "🖼️",
        "video_media": "🎥",
        "audio": "🎵",
        "file": "📄",
        "folder_directory": "📁",
        "directory": "📁",
        "drive": "💾",
        "partition": "💾",
        "volume": "💾",
        "mount": "📌",
        "unmount": "📌",
        "eject": "📌",
        "format_disk": "💾",
        "backup": "💾",
        "restore_data": "💾",
        "archive": "📦",
        "compress": "📦",
        "extract": "📦",
        "zip": "📦",
        "rar": "📦",
        "tar": "📦",
        "gz": "📦",
        "7z": "📦",
        "iso": "💿",
        "dmg": "💿",
        "exe": "💻",
        "msi": "💻",
        "deb": "💻",
        "rpm": "💻",
        "pkg": "💻",
        "app_file": "💻",
        "dll": "💻",
        "so": "💻",
        "dylib": "💻",
        "lib": "💻",
        "bin": "💻",
        "src": "💻",
        "include": "💻",
        "lib_file": "💻",
        "test": "🧪",
        "docs": "📚",
        "readme": "📖",
        "license": "📄",
        "changelog": "📝",
        "todo": "📝",
        "bug": "🐛",
        "feature": "✨",
        "enhancement": "🚀",
        "fix": "🔧",
        "patch": "🔧",
        "hotfix": "🔥",
        "release": "🎉",
        "version": "🏷️",
        "tag": "🏷️",
        "branch": "🌿",
        "commit": "💾",
        "merge": "🔀",
        "pull": "⬇️",
        "push": "⬆️",
        "clone": "📋",
        "fork": "🍴",
        "star": "⭐",
        "like": "👍",
        "dislike": "👎",
        "heart": "❤️",
        "favorite": "⭐",
        "bookmark": "🔖",
        "pin": "📌",
        "flag": "🚩",
        "mark": "✅",
        "check": "✅",
        "uncheck": "☐",
        "select": "☑️",
        "deselect": "☐",
        "choose": "☑️",
        "pick": "☑️",
        "option": "☐",
        "radio_button": "🔘",
        "checkbox": "☐",
        "toggle": "🔄",
        "switch": "🔄",
        "button": "🔘",
        "link": "🔗",
        "url": "🔗",
        "web": "🌐",
        "http": "🌐",
        "https": "🔒",
        "ftp": "📁",
        "sftp": "🔒",
        "ssh": "🔒",
        "telnet": "🔌",
        "ping": "🏓",
        "trace": "🔍",
        "route": "🗺️",
        "dns": "🌐",
        "dhcp": "🌐",
        "ip": "🌐",
        "mac": "🌐",
        "port": "🔌",
        "socket": "🔌",
        "pipe": "🔌",
        "stream": "🌊",
        "buffer": "💾",
        "cache": "💾",
        "memory": "💾",
        "heap": "💾",
        "stack": "📚",
        "queue": "📋",
        "list": "📋",
        "array": "📋",
        "vector": "📋",
        "matrix": "📊",
        "tensor": "📊",
        "scalar": "📊",
        "variable": "📊",
        "constant": "📊",
        "function": "⚙️",
        "method": "⚙️",
        "class": "🏗️",
        "object": "📦",
        "instance": "📦",
        "property": "📋",
        "attribute": "📋",
        "field": "📋",
        "parameter": "📋",
        "argument": "📋",
        "return": "↩️",
        "input": "📥",
        "output": "📤",
        "result": "📊",
        "value": "📊",
        "data": "📊",
        "type": "🏷️",
        "format_type": "📋",
        "encoding": "📋",
        "charset": "📋",
        "language": "🌐",
        "locale": "🌐",
        "timezone": "🕐",
        "currency": "💰",
        "unit": "📏",
        "measure": "📏",
        "scale": "⚖️",
        "weight": "⚖️",
        "length": "📏",
        "width": "📏",
        "height": "📏",
        "depth": "📏",
        "area": "📐",
        "volume_measure": "📦",
        "density": "📊",
        "speed_measure": "🏃",
        "velocity": "🏃",
        "acceleration": "🏃",
        "force": "💪",
        "energy": "⚡",
        "power_measure": "⚡",
        "temperature_measure": "🌡️",
        "pressure": "📊",
        "humidity": "💧",
        "light": "💡",
        "sound": "🔊",
        "color": "🎨",
        "palette": "🎨",
        "gradient": "🎨",
        "pattern": "🎨",
        "texture": "🎨",
        "material": "🏗️",
        "surface": "🏗️",
        "edge": "📐",
        "corner": "📐",
        "angle": "📐",
        "curve": "📐",
        "line": "📐",
        "point": "📍",
        "vertex": "📍",
        "polygon": "📐",
        "circle": "⭕",
        "ellipse": "⭕",
        "square": "⬜",
        "rectangle": "⬜",
        "triangle": "🔺",
        "diamond": "💎",
        "star_shape": "⭐",
        "cross": "✝️",
        "plus": "➕",
        "minus": "➖",
        "times": "✖️",
        "divide": "➗",
        "equals": "=",
        "not_equals": "≠",
        "less_than": "<",
        "greater_than": ">",
        "less_equal": "≤",
        "greater_equal": "≥",
        "approximately": "≈",
        "infinity": "∞",
        "pi": "π",
        "theta": "θ",
        "alpha": "α",
        "beta": "β",
        "gamma": "γ",
        "delta": "δ",
        "epsilon": "ε",
        "zeta": "ζ",
        "eta": "η",
        "iota": "ι",
        "kappa": "κ",
        "lambda": "λ",
        "mu": "μ",
        "nu": "ν",
        "xi": "ξ",
        "omicron": "ο",
        "rho": "ρ",
        "sigma": "σ",
        "tau": "τ",
        "upsilon": "υ",
        "phi": "φ",
        "chi": "χ",
        "psi": "ψ",
        "omega": "ω",
    }

    @classmethod
    def get_icon(cls, icon_name: str) -> QIcon:
        """
        Get an icon by name.

        Args:
            icon_name: Name of the icon to retrieve

        Returns:
            QIcon object
        """
        # Check cache first
        if icon_name in cls._icon_cache:
            return cls._icon_cache[icon_name]

        # Try to load from custom icon path
        icon_path = cls._get_icon_path(icon_name)
        if icon_path and icon_path.exists():
            icon = QIcon(str(icon_path))
            cls._icon_cache[icon_name] = icon
            return icon

        # Fallback to built-in icon
        icon = cls._create_builtin_icon(icon_name)
        cls._icon_cache[icon_name] = icon
        return icon

    @classmethod
    def _get_icon_path(cls, icon_name: str) -> Optional[Path]:
        """
        Get the path to a custom icon file.

        Args:
            icon_name: Name of the icon

        Returns:
            Path to the icon file or None if not found
        """
        # Look for icons in various locations
        possible_paths = [
            Path(__file__).parent.parent.parent.parent
            / "assets"
            / "icons"
            / f"{icon_name}.png",
            Path(__file__).parent.parent.parent.parent
            / "assets"
            / "icons"
            / f"{icon_name}.svg",
            Path(__file__).parent.parent.parent.parent
            / "assets"
            / "icons"
            / f"{icon_name}.ico",
            Path(__file__).parent / "icons" / f"{icon_name}.png",
            Path(__file__).parent / "icons" / f"{icon_name}.svg",
            Path(__file__).parent / "icons" / f"{icon_name}.ico",
        ]

        for path in possible_paths:
            if path.exists():
                return path

        return None

    @classmethod
    def _create_builtin_icon(cls, icon_name: str) -> QIcon:
        """
        Create a built-in icon using Unicode symbols.

        Args:
            icon_name: Name of the icon

        Returns:
            QIcon object
        """
        # Get the Unicode symbol
        symbol = cls._builtin_icons.get(icon_name, "❓")

        # Create a pixmap with the symbol
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)

        # Create painter to draw the symbol
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Set font
        font = QFont()
        font.setPointSize(16)
        painter.setFont(font)

        # Draw the symbol
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, symbol)
        painter.end()

        return QIcon(pixmap)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the icon cache."""
        cls._icon_cache.clear()

    @classmethod
    def get_available_icons(cls) -> List[str]:
        """
        Get a list of available icon names.

        Returns:
            List of available icon names
        """
        return list(cls._builtin_icons.keys())

    @classmethod
    def add_custom_icon(cls, icon_name: str, icon_path: str) -> None:
        """
        Add a custom icon to the cache.

        Args:
            icon_name: Name of the icon
            icon_path: Path to the icon file

        Raises:
            FileNotFoundError: If the icon file is not found
        """
        if os.path.exists(icon_path):
            icon = QIcon(icon_path)
            cls._icon_cache[icon_name] = icon
        else:
            raise FileNotFoundError(f"Icon file not found: {icon_path}")

    @classmethod
    def remove_icon(cls, icon_name: str) -> None:
        """
        Remove an icon from the cache.

        Args:
            icon_name: Name of the icon to remove
        """
        if icon_name in cls._icon_cache:
            del cls._icon_cache[icon_name]

    @classmethod
    def get_icon_size(cls, icon_name: str) -> Tuple[int, int]:
        """
        Get the size of an icon.

        Args:
            icon_name: Name of the icon

        Returns:
            Tuple of (width, height) in pixels
        """
        icon = cls.get_icon(icon_name)
        if not icon.isNull():
            sizes = icon.availableSizes()
            if sizes:
                size = sizes[0]
                return (size.width(), size.height())
        return (32, 32)

    @classmethod
    def create_icon_from_text(
        cls, text: str, size: Tuple[int, int] = (32, 32)
    ) -> QIcon:
        """
        Create an icon from text.

        Args:
            text: Text to display in the icon
            size: Size of the icon as (width, height)

        Returns:
            QIcon object
        """
        pixmap = QPixmap(*size)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Set font
        font = QFont()
        font.setPointSize(min(size) // 2)
        painter.setFont(font)

        # Draw the text
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, text)
        painter.end()

        return QIcon(pixmap)
