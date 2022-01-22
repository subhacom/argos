# Changelog

## [Unreleased]

## [0.1.2] - 2021-11-17

### Added
- Light and dark theme.
- Single-pane view in review tool.
- Unsaved changes in Review tool indicated by a `*` in front of the
  window title.
- Details about ROI selection in docs.
- Version number as a module attribute.
- Detailed documentation on reading ChangeList saved in reviewed track
  data file.
- Help menu with about info.
### Changed
- The Change-Window in Review tool adjusts size to contents.
- Font size persists across sessions.
- Documentation updated to reflect changes in the UI.
- Fixed persisting path display despite toggling `Show tracks`.

## [0.1.1] - 2021-08-20

### Changed
- Latest stable version of PyTorch 1.9.0 introduced some bugs. This
  version adds a workaround for generator type error in use of
  DataLoader: pytorch/pytorch#44714
- Fixed bug in the Hungarian algorithm.
- Fixed warning in YOLACT augmentation code.
- Fixed crash when Tracker was reset in CSRTracker mode.
- Modularized YOLACT training script - earlier pretty much everything
  was at the top level - now moved under main.
- Small optimization in Capture tool. Also the Cython version should
  now show diff.

## [0.1.0-12] - 2021-06-09

### Added
- Review tool: mouse-wheel scroll now lets you seek in video. Up -
  previous frame, Down - next frame, combined with Shift modifier-
  jump 10 frames back or forward.

## [0.1.0-13] - 2022-01-22

### Changed
- Review tool: mouse wheel scroll modifier Shift for single frame seek
  and Ctrl + Shift for jumps as just scroll interferes with scrollbar
  functionality.
  
- Review tool: made HDF5 data format `table` - same as track tool -
  which produces smaller and more flexible files with slower
  read/write.
