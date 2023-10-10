"""A help dialog showing keyboard shortcuts for Qt application."""
# pylint: disable=import-error
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6 import QtCore, QtWidgets

from arpes.utilities.ui import PRETTY_KEYS, label, vertical

if TYPE_CHECKING:
    from _typeshed import Incomplete
    from PySide6.QtGui import QKeyEvent
__all__ = ("BasicHelpDialog",)


class BasicHelpDialog(QtWidgets.QDialog):
    """A help dialog showing keyboard shortcuts for Qt application."""

    def __init__(self, shortcuts: Incomplete | None = None) -> None:
        """Initialize the help window and build widgets for the registered shortcuts."""
        super().__init__()

        if shortcuts is None:
            shortcuts = []

        self.layout = QtWidgets.QVBoxLayout()

        keyboard_shortcuts_info = QtWidgets.QGroupBox(title="Keyboard Shortcuts")
        keyboard_shortcuts_layout = QtWidgets.QGridLayout()
        for i, shortcut in enumerate(shortcuts):
            keyboard_shortcuts_layout.addWidget(
                label(", ".join(PRETTY_KEYS[k] for k in shortcut.chord), wordWrap=True),
                i,
                0,
            )
            keyboard_shortcuts_layout.addWidget(label(shortcut.label), i, 1)

        keyboard_shortcuts_info.setLayout(keyboard_shortcuts_layout)

        aboutInfo = QtWidgets.QGroupBox(title="About")
        vertical(
            label(
                "QtTool is the work of Conrad Stansbury, with much inspiration "
                "and thanks to the authors of ImageTool. QtTool is distributed "
                "as part of the PyARPES data analysis framework.",
                wordWrap=True,
            ),
            label(
                "Complaints and feature requests should be directed to chstan@berkeley.edu.",
                wordWrap=True,
            ),
        )

        from . import qt_info

        aboutInfo.setFixedHeight(qt_info.inches_to_px(1))

        self.layout.addWidget(keyboard_shortcuts_info)
        self.layout.addWidget(aboutInfo)
        self.setLayout(self.layout)

        self.setWindowTitle("Interactive Utility Help")
        self.setFixedSize(*qt_info.inches_to_px((2, 4)))

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """If the user preset H we should toggle the dialog, or close it if they pressed Esc."""
        if event.key() == QtCore.Qt.Key_H or event.key() == QtCore.Qt.Key_Escape:
            self._main_window._help_dialog = None  # pylint: disable=protected-access
            self.close()
