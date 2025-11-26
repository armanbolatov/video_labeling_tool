# Video Labeling Tool

Multi-class video labeling application.

## Setup

Install dependencies:
```bash
pip install opencv-python numpy
```

## Directory Structure

```
labelling/
├── raw_datasets/              # Input videos
│   └── {folder_name}/         # Videos to label (*.mp4)
├── conflict_filenames/        # Output labels
│   └── {folder_name}.txt      # Tab-separated labels
└── filtered_datasets/         # Filtered output
    └── {folder_name}_filtered/
        ├── conflict/
        ├── hugs_greeting/
        ├── martial_arts/
        └── ... (14 class folders)
```

## Usage

Run the labeler:
```bash
python label.py {folder_name}
```

Example:
```bash
python label.py 03112025_2
```

Optional arguments:
- `--rows N` - Grid rows (default: 4)
- `--cols N` - Grid columns (default: 4)
- `--fps N` - Playback FPS (default: 25)

## Controls

**Navigation:**
- `N` / Click "Next" - Next page
- `B` / Click "Back" - Previous page
- `Q` / `ESC` - Quit and save

**Labeling:**
- Click video → Enter class number (0-13) → Enter
- `0` - Remove label (deselect)
- `ESC` - Cancel dialog

**Filtering:**
- `F` - Filter labeled videos into class folders

## Classes

You can change the classes directly in the code

| Code | Class | Code | Class |
|------|-------|------|-------|
| 0 | nothing (deselect) | 7 | friendly_pull |
| 1 | conflict | 8 | friendly_push |
| 2 | hugs_greeting | 9 | friendly_slap |
| 3 | sportzal | 10 | play_around |
| 4 | friendly_kick | 11 | running |
| 5 | kiss | 12 | falling |
| 6 | martial_arts | 13 | maybe |

## Workflow

1. Place videos in `raw_datasets/{folder_name}/`
2. Run `python label.py {folder_name}`
3. Click videos and assign classes
4. Press `F` to filter into class folders
5. Output saved to `filtered_datasets/{folder_name}_filtered/`

## Notes

- Labels auto-save after each classification
- Videos loop continuously during labeling
- Color-coded borders indicate assigned classes
