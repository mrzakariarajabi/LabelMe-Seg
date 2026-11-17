from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PySide6 import QtWidgets
from PySide6.QtCore import Qt


class ClassicalSegmentationDialog(QtWidgets.QDialog):
    """Live grayscale-window segmentation that produces an editable mask."""

    def __init__(self, *, gray: np.ndarray, preview: Callable[[np.ndarray], None], parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Classical Segmentation"))
        self._gray = gray
        self._preview = preview
        self.was_applied = False
        layout = QtWidgets.QFormLayout(self)
        self.low = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.high = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        for slider, value in ((self.low, 0), (self.high, 255)):
            slider.setRange(0, 255)
            slider.setValue(value)
            slider.valueChanged.connect(self._update_preview)
        layout.addRow(self.tr("Minimum intensity:"), self.low)
        layout.addRow(self.tr("Maximum intensity:"), self.high)
        help_ = QtWidgets.QLabel(self.tr("Pixels inside this grayscale range are shown as a live mask preview."))
        help_.setWordWrap(True)
        layout.addRow(help_)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Apply | QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        apply = buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Apply)
        assert apply is not None
        apply.clicked.connect(self._apply_and_close)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        self._update_preview()

    @property
    def mask(self) -> np.ndarray:
        low, high = min(self.low.value(), self.high.value()), max(
            self.low.value(), self.high.value()
        )
        return (self._gray >= low) & (self._gray <= high)

    def _update_preview(self) -> None:
        self._preview(self.mask)

    def _apply_and_close(self) -> None:
        self.was_applied = True
        self.accept()
