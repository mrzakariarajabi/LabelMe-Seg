from __future__ import annotations

from collections.abc import Callable
from typing import Final

import PIL.Image
import PIL.ImageEnhance
import PIL.ImageFilter
import PIL.ImageOps
import PIL.ImageStat
import numpy as np
import skimage.exposure
import skimage.color
import skimage.feature
import skimage.filters
from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage

_NEUTRAL: Final = 100


class BrightnessContrastDialog(QtWidgets.QDialog):
    _publish: Callable[[QImage], None]
    _img: PIL.Image.Image
    _last_rgb: np.ndarray | None

    slider_brightness: QtWidgets.QSlider
    slider_contrast: QtWidgets.QSlider
    slider_gamma: QtWidgets.QSlider
    checkbox_grayscale: QtWidgets.QCheckBox
    filter_combo: QtWidgets.QComboBox
    filter_stack: QtWidgets.QListWidget
    filter_parameter: QtWidgets.QSlider
    _filter_parameter_label: QtWidgets.QLabel
    _range_controls: list[tuple[QtWidgets.QSlider, QtWidgets.QSlider, QtWidgets.QWidget]]

    def __init__(
        self,
        *,
        img: PIL.Image.Image,
        callback: Callable[[QImage], None],
        mask_preview: Callable[[np.ndarray | None], None] | None = None,
        brightness: int | None = None,
        contrast: int | None = None,
        state: dict | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Annotation Enhancement"))
        self._publish = callback
        self._mask_preview = mask_preview
        self.generated_mask: np.ndarray | None = None
        self._last_rgb = None

        # Alpha rides along as an untouched band; every other mode collapses
        # to RGB so one lookup table covers all color channels.
        self._img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
        self._is_grayscale = img.mode in ("1", "L", "I", "I;16", "F")
        self.slider_brightness = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.slider_contrast = QtWidgets.QSlider(Qt.Orientation.Horizontal)

        grid = QtWidgets.QGridLayout(self)
        grid.setColumnStretch(1, 1)
        captioned = (
            (self.tr("Brightness:"), self.slider_brightness, brightness),
            (self.tr("Contrast:"), self.slider_contrast, contrast),
        )
        for row, (caption, slider, value) in enumerate(captioned):
            slider.setRange(0, 3 * _NEUTRAL)
            slider.setSingleStep(2)
            slider.setPageStep(20)
            # Set before the signals connect: each change adjusts the whole
            # image, and the caller applies the starting values once.
            slider.setValue(_NEUTRAL if value is None else value)
            readout = QtWidgets.QLabel(_as_percent(value=slider.value()))
            readout.setAlignment(Qt.AlignmentFlag.AlignRight)
            slider.valueChanged.connect(
                lambda value, readout=readout: readout.setText(_as_percent(value=value))
            )
            slider.valueChanged.connect(lambda _value: self.apply())
            label = QtWidgets.QLabel(caption)
            label.setBuddy(slider)
            grid.addWidget(label, row, 0)
            grid.addWidget(slider, row, 1)
            grid.addWidget(readout, row, 2)

        self.checkbox_grayscale = QtWidgets.QCheckBox(self.tr("Grayscale preview"))
        self.checkbox_grayscale.setChecked(self._is_grayscale)
        self.checkbox_grayscale.toggled.connect(self._update_range_controls)
        self.checkbox_grayscale.toggled.connect(lambda _checked: self.apply())
        grid.addWidget(self.checkbox_grayscale, len(captioned), 0, 1, 3)

        self.filter_combo = QtWidgets.QComboBox()
        self.filter_combo.addItems(
            [
                self.tr("No filter"),
                self.tr("CLAHE (local contrast)"),
                self.tr("Auto contrast"),
                self.tr("Sharpen"),
                self.tr("Edge enhancement"),
                self.tr("Canny edge overlay"),
                self.tr("Unsharp mask"),
                self.tr("Adaptive threshold"),
                self.tr("Sobel gradient"),
                self.tr("Frangi / vesselness"),
                self.tr("LAB: L* channel"),
                self.tr("LAB: a* channel"),
                self.tr("LAB: b* channel"),
            ]
        )
        filter_label = QtWidgets.QLabel(self.tr("Add filter:"))
        filter_label.setBuddy(self.filter_combo)
        grid.addWidget(filter_label, len(captioned) + 1, 0)
        grid.addWidget(self.filter_combo, len(captioned) + 1, 1)
        add_filter = QtWidgets.QPushButton(self.tr("Add"))
        add_filter.clicked.connect(self._add_filter)
        grid.addWidget(add_filter, len(captioned) + 1, 2)

        self.filter_stack = QtWidgets.QListWidget()
        self.filter_stack.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.filter_stack.currentItemChanged.connect(self._show_filter_parameters)
        grid.addWidget(self.filter_stack, len(captioned) + 2, 0, 3, 2)
        stack_buttons = QtWidgets.QVBoxLayout()
        for text, slot in (
            (self.tr("Up"), self._move_filter_up),
            (self.tr("Down"), self._move_filter_down),
            (self.tr("Remove"), self._remove_filter),
        ):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(slot)
            stack_buttons.addWidget(button)
        stack_buttons.addStretch()
        grid.addLayout(stack_buttons, len(captioned) + 2, 2, 3, 1)

        self._filter_parameter_label = QtWidgets.QLabel()
        self.filter_parameter = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.filter_parameter.valueChanged.connect(self._store_filter_parameter)
        grid.addWidget(self._filter_parameter_label, len(captioned) + 5, 0)
        grid.addWidget(self.filter_parameter, len(captioned) + 5, 1, 1, 2)
        self._show_filter_parameters(None)

        self.slider_gamma = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.slider_gamma.setRange(10, 300)
        self.slider_gamma.setValue(_NEUTRAL)
        self.slider_gamma.setSingleStep(5)
        self.slider_gamma.setPageStep(25)
        self.slider_gamma.valueChanged.connect(lambda _value: self.apply())
        gamma_label = QtWidgets.QLabel(self.tr("Gamma:"))
        gamma_label.setBuddy(self.slider_gamma)
        grid.addWidget(gamma_label, len(captioned) + 6, 0)
        grid.addWidget(self.slider_gamma, len(captioned) + 6, 1, 1, 2)

        self._range_controls = []
        channel_names = (
            self.tr("Gray"),
            self.tr("Red"),
            self.tr("Green"),
            self.tr("Blue"),
        )
        for index, name in enumerate(channel_names):
            low = QtWidgets.QSlider(Qt.Orientation.Horizontal)
            high = QtWidgets.QSlider(Qt.Orientation.Horizontal)
            for slider, value in ((low, 0), (high, 255)):
                slider.setRange(0, 255)
                slider.setValue(value)
                slider.valueChanged.connect(lambda _value: self.apply())
            row_widget = QtWidgets.QWidget()
            row = QtWidgets.QHBoxLayout(row_widget)
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(low)
            row.addWidget(high)
            label = QtWidgets.QLabel(self.tr("%s range:") % name)
            label.setBuddy(low)
            grid.addWidget(label, len(captioned) + 7 + index, 0)
            grid.addWidget(row_widget, len(captioned) + 7 + index, 1, 1, 2)
            self._range_controls.append((low, high, row_widget))

        segmentation = QtWidgets.QGroupBox(self.tr("Classical Segmentation"))
        segmentation_layout = QtWidgets.QFormLayout(segmentation)
        self.segmentation_enabled = QtWidgets.QCheckBox(
            self.tr("Preview grayscale intensity range")
        )
        self.segmentation_low = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.segmentation_high = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        for slider, value in ((self.segmentation_low, 0), (self.segmentation_high, 255)):
            slider.setRange(0, 255)
            slider.setValue(value)
            slider.valueChanged.connect(lambda _value: self.apply())
        self.segmentation_enabled.toggled.connect(lambda _checked: self.apply())
        create_mask = QtWidgets.QPushButton(self.tr("Create Editable Mask"))
        create_mask.clicked.connect(self._create_mask)
        segmentation_layout.addRow(self.segmentation_enabled)
        segmentation_layout.addRow(self.tr("Minimum intensity:"), self.segmentation_low)
        segmentation_layout.addRow(self.tr("Maximum intensity:"), self.segmentation_high)
        segmentation_layout.addRow(create_mask)
        grid.addWidget(segmentation, len(captioned) + 11, 0, 1, 3)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults
            | QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        restore = buttons.button(
            QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults
        )
        assert restore is not None
        restore.clicked.connect(self._restore_defaults)
        buttons.rejected.connect(self.close)
        grid.addWidget(buttons, len(captioned) + 12, 0, 1, 3)
        self.setMinimumWidth(360)
        self._update_range_controls()
        if state is not None:
            self.restore_state(state=state)

    def _restore_defaults(self) -> None:
        self.slider_brightness.setValue(_NEUTRAL)
        self.slider_contrast.setValue(_NEUTRAL)
        self.checkbox_grayscale.setChecked(self._is_grayscale)
        self.filter_stack.clear()
        self.apply()
        self.slider_gamma.setValue(_NEUTRAL)
        for low, high, _ in self._range_controls:
            low.setValue(0)
            high.setValue(255)
        self.segmentation_enabled.setChecked(False)
        self.segmentation_low.setValue(0)
        self.segmentation_high.setValue(255)

    def export_state(self) -> dict:
        return {
            "brightness": self.slider_brightness.value(),
            "contrast": self.slider_contrast.value(),
            "gamma": self.slider_gamma.value(),
            "grayscale": self.checkbox_grayscale.isChecked(),
            "ranges": [(low.value(), high.value()) for low, high, _ in self._range_controls],
            "filters": [
                self.filter_stack.item(row).data(Qt.ItemDataRole.UserRole)
                for row in range(self.filter_stack.count())
            ],
        }

    def restore_state(self, *, state: dict) -> None:
        self.slider_brightness.setValue(state.get("brightness", _NEUTRAL))
        self.slider_contrast.setValue(state.get("contrast", _NEUTRAL))
        self.slider_gamma.setValue(state.get("gamma", _NEUTRAL))
        self.checkbox_grayscale.setChecked(state.get("grayscale", self._is_grayscale))
        for controls, values in zip(self._range_controls, state.get("ranges", ()), strict=False):
            controls[0].setValue(values[0])
            controls[1].setValue(values[1])
        self.filter_stack.clear()
        for index, parameter in state.get("filters", ()):
            item = QtWidgets.QListWidgetItem(self.filter_combo.itemText(index))
            item.setData(Qt.ItemDataRole.UserRole, (index, parameter))
            self.filter_stack.addItem(item)

    def _update_range_controls(self) -> None:
        grayscale = self.checkbox_grayscale.isChecked()
        for index, (_, _, widget) in enumerate(self._range_controls):
            widget.setVisible(index == 0 if grayscale else index > 0)

    def _add_filter(self) -> None:
        index = self.filter_combo.currentIndex()
        if index == 0:
            return
        item = QtWidgets.QListWidgetItem(self.filter_combo.currentText())
        item.setData(Qt.ItemDataRole.UserRole, (index, self._filter_default(index=index)))
        self.filter_stack.addItem(item)
        self.filter_stack.setCurrentItem(item)
        self.apply()

    def _move_filter_up(self) -> None:
        row = self.filter_stack.currentRow()
        if row > 0:
            self.filter_stack.insertItem(row - 1, self.filter_stack.takeItem(row))
            self.filter_stack.setCurrentRow(row - 1)
            self.apply()

    def _move_filter_down(self) -> None:
        row = self.filter_stack.currentRow()
        if 0 <= row < self.filter_stack.count() - 1:
            self.filter_stack.insertItem(row + 1, self.filter_stack.takeItem(row))
            self.filter_stack.setCurrentRow(row + 1)
            self.apply()

    def _remove_filter(self) -> None:
        row = self.filter_stack.currentRow()
        if row >= 0:
            self.filter_stack.takeItem(row)
            self.apply()

    def _filter_default(self, *, index: int) -> int:
        return {1: 3, 5: 10, 6: 100, 7: 35, 9: 2}.get(index, 0)

    def _filter_parameter_spec(self, *, index: int) -> tuple[str, int, int] | None:
        specs = {
            1: (self.tr("CLAHE clip limit:"), 1, 100),
            5: (self.tr("Canny sigma:"), 1, 50),
            6: (self.tr("Unsharp amount:"), 1, 300),
            7: (self.tr("Adaptive block size:"), 3, 101),
            9: (self.tr("Frangi scale:"), 1, 10),
        }
        return specs.get(index)

    def _show_filter_parameters(
        self,
        item: QtWidgets.QListWidgetItem | None,
        _previous: QtWidgets.QListWidgetItem | None = None,
    ) -> None:
        if item is None:
            self._filter_parameter_label.hide()
            self.filter_parameter.hide()
            return
        index, value = item.data(Qt.ItemDataRole.UserRole)
        spec = self._filter_parameter_spec(index=index)
        if spec is None:
            self._filter_parameter_label.hide()
            self.filter_parameter.hide()
            return
        label, minimum, maximum = spec
        self._filter_parameter_label.setText(label)
        self._filter_parameter_label.show()
        self.filter_parameter.show()
        self.filter_parameter.blockSignals(True)
        self.filter_parameter.setRange(minimum, maximum)
        self.filter_parameter.setValue(value)
        self.filter_parameter.blockSignals(False)

    def _store_filter_parameter(self, value: int) -> None:
        item = self.filter_stack.currentItem()
        if item is None:
            return
        index, _ = item.data(Qt.ItemDataRole.UserRole)
        if index == 7:  # threshold_local requires an odd block size.
            value = value if value % 2 else value + 1
        item.setData(Qt.ItemDataRole.UserRole, (index, value))
        self.apply()

    def _create_mask(self) -> None:
        if self._last_rgb is None:
            return
        self.generated_mask = self._segmentation_mask(rgb=self._last_rgb)
        if self.generated_mask is not None and self.generated_mask.any():
            self.accept()

    def _segmentation_mask(self, *, rgb: np.ndarray) -> np.ndarray | None:
        if not self.segmentation_enabled.isChecked():
            return None
        gray = np.asarray(PIL.Image.fromarray(rgb).convert("L"))
        low, high = sorted((self.segmentation_low.value(), self.segmentation_high.value()))
        return (gray >= low) & (gray <= high)

    def _window_channel(self, *, values: np.ndarray, low: int, high: int) -> np.ndarray:
        low, high = min(low, high), max(low, high)
        if low == high:
            return np.where(values >= high, 255, 0).astype(np.uint8)
        return np.clip((values.astype(float) - low) * 255 / (high - low), 0, 255).astype(np.uint8)

    def _apply_filter(self, *, rgb: np.ndarray, index: int, parameter: int) -> np.ndarray:
        """Apply one pipeline stage; callers preserve the user-selected order."""
        if index == 1:
            return (
                skimage.exposure.equalize_adapthist(rgb, clip_limit=parameter / 100)
                * 255
            ).astype(np.uint8)
        if index == 2:
            return np.asarray(PIL.ImageOps.autocontrast(PIL.Image.fromarray(rgb)))
        if index == 3:
            return np.asarray(PIL.Image.fromarray(rgb).filter(PIL.ImageFilter.SHARPEN))
        if index == 4:
            return np.asarray(
                PIL.Image.fromarray(rgb).filter(PIL.ImageFilter.EDGE_ENHANCE_MORE)
            )
        gray = np.asarray(PIL.Image.fromarray(rgb).convert("L"))
        if index == 5:
            result = rgb.copy()
            result[skimage.feature.canny(gray, sigma=parameter / 10)] = (255, 0, 0)
            return result
        if index == 6:
            return np.clip(
                np.stack(
                    [
                        skimage.filters.unsharp_mask(
                            rgb[..., channel], amount=parameter / 100, preserve_range=True
                        )
                        for channel in range(3)
                    ],
                    axis=-1,
                ),
                0,
                255,
            ).astype(np.uint8)
        if index == 7:
            block = min(parameter if parameter % 2 else parameter + 1, min(gray.shape) // 2 * 2 - 1)
            threshold = (
                skimage.filters.threshold_local(gray, block_size=block)
                if block >= 3
                else np.full_like(gray, gray.mean())
            )
            return np.repeat((gray >= threshold)[..., None] * 255, 3, axis=2).astype(np.uint8)
        if index == 8:
            displayed = skimage.exposure.rescale_intensity(
                skimage.filters.sobel(gray), out_range=(0, 255)
            )
            return np.repeat(displayed[..., None], 3, axis=2).astype(np.uint8)
        if index == 9:
            displayed = skimage.exposure.rescale_intensity(
                skimage.filters.frangi(gray, sigmas=range(1, parameter + 1)),
                out_range=(0, 255),
            )
            return np.repeat(displayed[..., None], 3, axis=2).astype(np.uint8)
        lab = skimage.color.rgb2lab(rgb)
        channel = lab[..., index - 10]
        bounds = ((0, 100), (-128, 127), (-128, 127))[index - 10]
        displayed = skimage.exposure.rescale_intensity(
            channel, in_range=bounds, out_range=(0, 255)
        )
        return np.repeat(displayed[..., None], 3, axis=2).astype(np.uint8)

    def apply(self) -> None:
        gain = self.slider_brightness.value() / _NEUTRAL
        spread = self.slider_contrast.value() / _NEUTRAL
        identity = list(range(256))
        alpha_lut = identity if self._img.mode == "RGBA" else []
        ramp = PIL.Image.frombytes("L", (256, 1), bytes(identity))
        bright_ramp = PIL.ImageEnhance.Brightness(ramp).enhance(gain)
        bright_lut = list(bright_ramp.tobytes())
        bright_img = (
            self._img if gain == 1.0 else self._img.point(bright_lut * 3 + alpha_lut)
        )

        if spread == 1.0:
            img = bright_img
        else:
            # Pillow derives the contrast pivot after brightness has clipped
            # and rounded, so the combined table must use that intermediate.
            pivot = int(PIL.ImageStat.Stat(bright_img.convert("L")).mean[0] + 0.5)
            contrast_ramp = PIL.Image.blend(
                PIL.Image.new("L", ramp.size, pivot), bright_ramp, spread
            )
            contrast_lut = list(contrast_ramp.tobytes())
            img = self._img.point(contrast_lut * 3 + alpha_lut)

        rgb = np.asarray(img.convert("RGB")).copy()
        if self.checkbox_grayscale.isChecked():
            gray = np.asarray(img.convert("L"))
            low, high, _ = self._range_controls[0]
            gray = self._window_channel(values=gray, low=low.value(), high=high.value())
            rgb = np.repeat(gray[..., None], 3, axis=2)
        else:
            for channel, (low, high, _) in enumerate(self._range_controls[1:]):
                rgb[..., channel] = self._window_channel(
                    values=rgb[..., channel], low=low.value(), high=high.value()
                )
        rgb = (
            skimage.exposure.adjust_gamma(
                rgb / 255.0, gamma=self.slider_gamma.value() / _NEUTRAL
            )
            * 255
        ).astype(np.uint8)
        for row in range(self.filter_stack.count()):
            index, parameter = self.filter_stack.item(row).data(Qt.ItemDataRole.UserRole)
            rgb = self._apply_filter(rgb=rgb, index=index, parameter=parameter)
        self._last_rgb = rgb
        if self._mask_preview is not None:
            self._mask_preview(self._segmentation_mask(rgb=rgb))
        rgb_image = PIL.Image.fromarray(rgb, mode="RGB")
        img = (
            rgb_image
            if self._img.mode != "RGBA"
            else PIL.Image.merge("RGBA", (*rgb_image.split(), self._img.getchannel("A")))
        )

        image_format = (
            QImage.Format.Format_RGBA8888
            if img.mode == "RGBA"
            else QImage.Format.Format_RGB888
        )
        qimage = QImage(
            img.tobytes(),
            img.width,
            img.height,
            img.width * len(img.getbands()),
            image_format,
        )
        self._publish(qimage)


def _as_percent(*, value: int) -> str:
    return f"{value}%"
