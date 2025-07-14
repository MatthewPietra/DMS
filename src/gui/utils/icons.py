import os
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap
        from PySide6.QtGui import QFont, QPainter
        from PySide6.QtGui import QFont, QPainter

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMS GUI Icons

Icon management for the DMS GUI application.
Provides a centralized way to manage and access icons.
"""

class IconManager:
    """
    Manages icons for the DMS GUI application.

    Provides a centralized way to access icons with fallback
    to built-in icons when custom icons are not available.
    """

    # Icon cache
    _icon_cache = {}

    # Built-in icon definitions (using Unicode symbols)
    _builtin_icons = {
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
        "monitor": "🖥️",
        "keyboard": "⌨️",
        "mouse": "🖱️",
        "printer": "🖨️",
        "scanner": "📷",
        "speaker": "🔊",
        "headphones": "🎧",
        "microphone": "🎤",
        "camera": "📷",
        "webcam": "📷",
        "projector": "📽️",
        "tv": "📺",
        "radio": "📻",
        "game": "🎮",
        "music": "🎵",
        "video": "🎥",
        "photo": "🖼️",
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
        "batch": "📦",
        "sample": "📋",
        "dataset": "📊",
        "training": "🏋️",
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
        "email": "📧",
        "avatar": "👤",
        "picture": "🖼️",
        "photo": "📷",
        "image": "🖼️",
        "video": "🎥",
        "audio": "🎵",
        "file": "📄",
        "folder": "📁",
        "directory": "📁",
        "drive": "💾",
        "partition": "💾",
        "volume": "💾",
        "mount": "📌",
        "unmount": "📌",
        "eject": "📌",
        "format": "💾",
        "partition": "💾",
        "backup": "💾",
        "restore": "💾",
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
        "app": "💻",
        "dll": "💻",
        "so": "💻",
        "dylib": "💻",
        "lib": "💻",
        "bin": "💻",
        "src": "💻",
        "include": "💻",
        "lib": "💻",
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
        "radio": "🔘",
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
        "format": "📋",
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
        "volume": "📦",
        "density": "📊",
        "speed": "🏃",
        "velocity": "🏃",
        "acceleration": "🏃",
        "force": "💪",
        "energy": "⚡",
        "power": "⚡",
        "temperature": "🌡️",
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
        "star": "⭐",
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
        "theta": "θ",
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
    def _get_icon_path(cls, icon_name: str) -> Path:
        """Get the path to a custom icon file."""
        # Look for icons in various locations
        possible_paths = [
            Path(__file__).parent.parent.parent.parent
            / "assets"
            / "icons"
            / "{icon_name}.png",
            Path(__file__).parent.parent.parent.parent
            / "assets"
            / "icons"
            / "{icon_name}.svg",
            Path(__file__).parent.parent.parent.parent
            / "assets"
            / "icons"
            / "{icon_name}.ico",
            Path(__file__).parent / "icons" / "{icon_name}.png",
            Path(__file__).parent / "icons" / "{icon_name}.svg",
            Path(__file__).parent / "icons" / "{icon_name}.ico",
        ]

        for path in possible_paths:
            if path.exists():
                return path

        return None

    @classmethod
    def _create_builtin_icon(cls, icon_name: str) -> QIcon:
        """Create a built-in icon using Unicode symbols."""
        # Get the Unicode symbol
        symbol = cls._builtin_icons.get(icon_name, "❓")

        # Create a pixmap with the symbol
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)

        # Create painter to draw the symbol
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # Set font
        font = QFont()
        font.setPointSize(16)
        painter.setFont(font)

        # Draw the symbol
        painter.drawText(pixmap.rect(), Qt.AlignCenter, symbol)
        painter.end()

        return QIcon(pixmap)

    @classmethod
    def clear_cache(cls):
        """Clear the icon cache."""
        cls._icon_cache.clear()

    @classmethod
    def get_available_icons(cls) -> list:
        """Get a list of available icon names."""
        return list(cls._builtin_icons.keys())

    @classmethod
    def add_custom_icon(cls, icon_name: str, icon_path: str):
        """Add a custom icon to the cache."""
        if os.path.exists(icon_path):
            icon = QIcon(icon_path)
            cls._icon_cache[icon_name] = icon
        else:
            raise FileNotFoundError("Icon file not found: {icon_path}")

    @classmethod
    def remove_icon(cls, icon_name: str):
        """Remove an icon from the cache."""
        if icon_name in cls._icon_cache:
            del cls._icon_cache[icon_name]

    @classmethod
    def get_icon_size(cls, icon_name: str) -> tuple:
        """Get the size of an icon."""
        icon = cls.get_icon(icon_name)
        if not icon.isNull():
            return icon.availableSizes()[0] if icon.availableSizes() else (32, 32)
        return (32, 32)

    @classmethod
    def create_icon_from_text(cls, text: str, size: tuple = (32, 32)) -> QIcon:
        """Create an icon from text."""
        pixmap = QPixmap(*size)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # Set font
        font = QFont()
        font.setPointSize(min(size) // 2)
        painter.setFont(font)

        # Draw the text
        painter.drawText(pixmap.rect(), Qt.AlignCenter, text)
        painter.end()

        return QIcon(pixmap)
