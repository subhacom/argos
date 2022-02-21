# -*- coding: utf-8 -*-
# Author: Subhasis Ray <ray dot subhasis at gmail dot com>
# Created: 2022-02-20 11:13 PM
from typing import List, Dict

from PyQt5 import QtWidgets as qw, QtCore as qc, QtGui as qg
from argos import utility as ut


settings = ut.init()


class TrackList(qw.QListWidget):
    """
    Attributes
    ----------
    keepSelection: bool
        Whether to maintain selection of list item across frames. When the path
        of the selected item is drawn, this makes things VERY SLOW.

    selected: list of int
        IDs of selected objects (in Review tool only a single selection is
        allowed).

    """

    # Map tracks: source-id, target-id, end-frame, swap
    sigMapTracks = qc.pyqtSignal(int, int, int, bool)
    sigSelected = qc.pyqtSignal(list)

    def __init__(self, *args, **kwargs):
        super(TrackList, self).__init__(*args, **kwargs)
        self._drag_button = qc.Qt.NoButton
        self.setSelectionMode(qw.QAbstractItemView.SingleSelection)
        self.itemSelectionChanged.connect(self.sendSelected)
        self.itemClicked.connect(self.sendSelected)
        self.keepSelection = settings.value('review/keepselection', type=bool)
        self.currentFrame = -1
        self.selected = []

    @qc.pyqtSlot(int)
    def setCurrentFrame(self, val):
        self.currentFrame = val

    @qc.pyqtSlot(bool)
    def setKeepSelection(self, val):
        self.keepSelection = val
        settings.setValue('review/keepselection', val)

    def decode_item_data(
        self, mime_data: qc.QMimeData
    ) -> List[Dict[qc.Qt.ItemDataRole, qc.QVariant]]:
        """This was a test trick found here:
        https://wiki.python.org/moin/PyQt/Handling%20Qt%27s%20internal%20item%20MIME%20type
        but a much simpler solution for my case was here:
        https://stackoverflow.com/questions/9715171/how-to-drop-items-on-qlistwidget-between-some-items
        """
        data = mime_data.data('application/x-qabstractitemmodeldatalist')
        ds = qc.QDataStream(data)
        item = {}
        item_list = []
        while not ds.atEnd():
            ds.readInt32()  # row
            ds.readInt32()  # col
            map_items = ds.readInt32()
            for ii in range(map_items):
                key = ds.readInt32()
                value = qc.QVariant()
                ds >> value
                item[qc.Qt.ItemDataRole(key)] = value
            item_list.append(item)
        return item_list

    def dragMoveEvent(self, e: qg.QDragMoveEvent) -> None:
        """This is just for tracking left vs right mouse button drag"""
        self._drag_button = e.mouseButtons()
        super(TrackList, self).dragMoveEvent(e)

    def dropEvent(self, event: qg.QDropEvent) -> None:
        """If dragged with left button, assign dropped trackid to the target
        trackid, if right button, swap the two.  If Shift key was
        pressed, then apply these only for the current frame,
        otherwise also all future frames.

        """
        # items = self.decode_item_data(event.mimeData())
        # assert  len(items) == 1, 'Only allowed to drop a single item'
        # item = items[0]
        # logging.debug(f'data: {item[qc.Qt.DisplayRole].value()}')
        # If dragged with left button, rename. if right button, swap
        source = event.source().currentItem()
        target = self.itemAt(event.pos())
        if target is None:
            event.ignore()
            return
        endFrame = -1
        if qw.QApplication.keyboardModifiers() == qc.Qt.AltModifier:
            endFrame, accept = qw.QInputDialog.getInt(
                self,
                'Frame range',
                'Apply till frame',
                self.currentFrame,
                self.currentFrame,
                2 ** 31 - 1,
            )
            if not accept:
                endFrame = -1
        elif qw.QApplication.keyboardModifiers() == qc.Qt.ShiftModifier:
            endFrame = self.currentFrame

        self.sigMapTracks.emit(
            int(source.text()),
            int(target.text()),
            endFrame,
            self._drag_button == qc.Qt.RightButton,
        )
        event.ignore()

    @qc.pyqtSlot(list)
    def replaceAll(self, track_list: List[int]):
        """Replace all items with keys from new tracks dictionary"""
        self.blockSignals(True)
        self.clear()
        sorted_tracks = sorted(track_list)
        self.addItems([str(x) for x in sorted_tracks])
        # print(self, 'keep selection', self.keepSelection, 'selected:', self.selected)
        if self.keepSelection and len(self.selected) > 0:
            try:
                idx = sorted_tracks.index(self.selected[0])
                self.setCurrentRow(idx)
            except ValueError:
                pass
            self.blockSignals(False)
            # self.sigSelected.emit(self.selected)
            return
        self.blockSignals(False)
        # print('Updating selection')
        self.sendSelected()

    @qc.pyqtSlot()
    def sendSelected(self):
        """Intermediate slot to convert text labels into integer track ids"""
        self.selected = [int(item.text()) for item in self.selectedItems()]
        self.sigSelected.emit(self.selected)
        # Note: even if this is sent multiple times (e.g., both
        # itemSelectionChanged and itemClicked connected to this slot,
        # the destination FrameView keeps track of current selection and
        # ignores if the selection has not changed).
