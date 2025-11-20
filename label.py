import argparse
import glob
import math
import os
import shutil
from pathlib import Path
import ctypes

import cv2
import numpy as np

CLASSES = {
    0: "nothing",
    1: "conflict",
    2: "hugs_greeting",
    3: "sportzal",
    4: "friendly_kick",
    5: "kiss",
    6: "martial_arts",
    7: "friendly_pull",
    8: "friendly_push",
    9: "friendly_slap",
    10: "play_around",
    11: "running",
    12: "falling",
    13: "maybe"
}

CLASS_COLORS = {
    0: (80, 80, 80),       # Gray
    1: (0, 0, 255),        # Red
    2: (255, 105, 180),    # Pink
    3: (0, 255, 255),      # Yellow
    4: (0, 69, 255),       # Orange-Red
    5: (255, 0, 255),      # Magenta
    6: (0, 140, 255),      # Dark Orange
    7: (203, 192, 255),    # Light Pink
    8: (130, 0, 75),       # Dark Purple
    9: (147, 20, 255),     # Deep Pink
    10: (0, 255, 0),       # Green
    11: (255, 255, 0),     # Cyan
    12: (128, 128, 128),   # Gray
    13: (0, 165, 255)      # Orange
}

CLASS_NAMES = {v: k for k, v in CLASSES.items()}


def list_videos(folder):
    return sorted(glob.glob(os.path.join(folder, "*.mp4")))


def load_labels(path):
    labels = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    labels[parts[0]] = parts[1]
    return labels


def save_labels(path, labels):
    with open(path, "w", encoding="utf-8") as f:
        for key in sorted(labels.keys()):
            f.write(f"{key}\t{labels[key]}\n")


def truncate_caption(text, tile_w):
    if not text:
        return ""
    max_chars = max(4, tile_w // 8)
    return text[:max_chars] + "..." if len(text) > max_chars else text


def get_screen_size():
    try:
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except:
        return 1920, 1080


def filter_dataset(folder_name, videos_dir, labels_file):
    """Filter and organize videos into class folders. Returns (success, messages)"""
    output_dir = Path("filtered_datasets") / f"{folder_name}_filtered"
    if output_dir.exists():
        return False, [f"Filtered folder already exists:", "", "{output_dir}", "", "Delete it if you want to filter again"]
    
    labeled = load_labels(str(labels_file))
    video_files = list_videos(str(videos_dir))
    if not video_files:
        return False, ["No videos found!"]
    
    messages = [f"Filtering: {folder_name}", f"Videos: {len(video_files)} | Labeled: {len(labeled)}", ""]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    for class_name in list(CLASSES.values()):
        (output_dir / class_name).mkdir(exist_ok=True)
    
    class_counts = {}
    unclassified = 0
    
    for video_path in video_files:
        stem = Path(video_path).stem
        label_key = f"{stem}_{folder_name}"
        
        if label_key in labeled:
            class_name = labeled[label_key]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        else:
            class_name = "nothing"
            unclassified += 1
        
        dest_path = output_dir / class_name / Path(video_path).name
        shutil.copy2(video_path, dest_path)
    
    messages.extend(["Copying complete!", ""])
    messages.extend(f"{c}: {class_counts[c]}" for c in sorted(class_counts.keys()))
    messages.extend([f"nothing: {unclassified}", "", f"Total: {len(video_files)}"])
    
    return True, messages


def show_filtering_window(folder_name, videos_dir, labels_file, screen_w, screen_h, caps, grid_func, delay_ms):
    """Show filtering progress in a window while videos continue playing"""
    success, messages = filter_dataset(folder_name, videos_dir, labels_file)
    
    win_name = "Filtering Dataset"
    win_w, win_h = 600, 400
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, win_w, win_h)
    cv2.moveWindow(win_name, (screen_w - win_w) // 2, (screen_h - win_h) // 2)
    
    img = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)
    
    y = 40
    line_height = 25
    
    for msg in messages:
        if msg:
            cv2.putText(img, msg, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += line_height
    
    # Instructions
    y = win_h - 40
    if success:
        cv2.putText(img, "Press any key to continue...", (20, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    else:
        cv2.putText(img, "Press any key to continue...", (20, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1, cv2.LINE_AA)
    
    # Keep videos playing until key pressed or window closed
    while True:
        cv2.imshow(win_name, img)
        grid_func()  # Update main video grid
        
        key = cv2.waitKey(delay_ms) & 0xFF
        if key != 255:  # Any key pressed
            break
        
        # Check if window was closed (must be after waitKey)
        try:
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except:
            break
    
    try:
        cv2.destroyWindow(win_name)
    except:
        pass


def create_dialog_image(dialog_w, dialog_h, input_str=""):
    img = np.zeros((dialog_h, dialog_w, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)
    
    cv2.putText(img, "Select Class (type number + Enter)", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Display classes in two columns
    classes_list = sorted(CLASSES.items())
    col_width = dialog_w // 2
    y_start = 80
    y_spacing = 35
    
    for i, (code, name) in enumerate(classes_list):
        col = i % 2  # 0 for left, 1 for right
        row = i // 2
        
        x_base = 20 + col * col_width
        y = y_start + row * y_spacing
        
        color = CLASS_COLORS[code]
        cv2.rectangle(img, (x_base, y - 20), (x_base + 30, y), color, -1)
        cv2.rectangle(img, (x_base, y - 20), (x_base + 30, y), (200, 200, 200), 1)
        cv2.putText(img, f"{code}: {name}", (x_base + 40, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    
    cv2.putText(img, "Press ESC to cancel", (10, dialog_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
    
    cv2.rectangle(img, (10, dialog_h - 100), (dialog_w - 10, dialog_h - 50), (60, 60, 60), -1)
    cv2.rectangle(img, (10, dialog_h - 100), (dialog_w - 10, dialog_h - 50), (200, 200, 200), 2)
    cv2.putText(img, f"Input: {input_str}", (20, dialog_h - 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    return img


def main():
    parser = argparse.ArgumentParser(description="4x4 video labeling with multiple classes")
    parser.add_argument("folder_name", help="Folder name in raw_datasets/")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--fps", type=float, default=25)
    args = parser.parse_args()

    # Setup paths
    videos_dir = Path("raw_datasets") / args.folder_name
    if not videos_dir.exists():
        print(f"Error: {videos_dir} does not exist")
        return
    
    files = list_videos(str(videos_dir))
    if not files:
        print(f"No videos found in {videos_dir}")
        return

    folder_name = args.folder_name
    labels_dir = Path("conflict_filenames")
    labels_dir.mkdir(exist_ok=True)
    labels_file = labels_dir / f"{folder_name}.txt"
    
    labeled = load_labels(str(labels_file))
    PAGE_SIZE = args.rows * args.cols
    num_pages = math.ceil(len(files) / PAGE_SIZE)

    window_name = "Video Labeler - Click to classify | N=next B=back Q=quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    group_stems = []
    nav_action = None
    selected_video_idx = None

    def on_mouse(event, x, y, flags, userdata):
        nonlocal nav_action, selected_video_idx
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        tile_w, tile_h, cols, rows, x_offset, y_offset, back_rect, next_rect = userdata
        # Handle top bar buttons first
        bx1, by1, bx2, by2 = back_rect
        nx1, ny1, nx2, ny2 = next_rect
        if bx1 <= x <= bx2 and by1 <= y <= by2:
            nav_action = 'back'
            return
        if nx1 <= x <= nx2 and ny1 <= y <= ny2:
            nav_action = 'next'
            return
        # Exclude clicks above the grid start (header/top margin area)
        if y < y_offset or x < x_offset:
            return
        rel_x = x - x_offset
        rel_y = y - y_offset
        col = rel_x // tile_w
        row = rel_y // tile_h
        if 0 <= col < cols and 0 <= row < rows:
            idx = int(row) * cols + int(col)
            if 0 <= idx < len(group_stems):
                stem = group_stems[idx]
                if stem:
                    selected_video_idx = idx

    screen_w, screen_h = get_screen_size()
    HEADER_H, TOP_MARGIN = 24, 30
    available_h = screen_h - TOP_MARGIN - HEADER_H
    
    tile_w = screen_w // args.cols
    tile_h = available_h // args.rows
    grid_w, grid_h = tile_w * args.cols, tile_h * args.rows
    x_offset = (screen_w - grid_w) // 2
    y_offset = TOP_MARGIN + HEADER_H + (available_h - grid_h) // 2
    
    black = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    cv2.resizeWindow(window_name, screen_w, screen_h)
    cv2.moveWindow(window_name, 0, 0)

    page_idx = 0
    while page_idx < num_pages:
        start = page_idx * PAGE_SIZE
        end = min(len(files), start + PAGE_SIZE)
        group_files = files[start:end]
        group_stems = [Path(p).stem for p in group_files]

        pad = PAGE_SIZE - len(group_files)
        if pad > 0:
            group_files += [None] * pad
            group_stems += [None] * pad

        BTN_W, BTN_H, BTN_PAD = 120, 18, 8
        btn_y1 = TOP_MARGIN + (HEADER_H - BTN_H) // 2
        btn_y2 = btn_y1 + BTN_H
        back_rect = (screen_w - 2 * BTN_W - 2 * BTN_PAD, btn_y1, screen_w - BTN_W - 2 * BTN_PAD, btn_y2)
        next_rect = (screen_w - BTN_W - BTN_PAD, btn_y1, screen_w - BTN_PAD, btn_y2)

        cv2.setMouseCallback(window_name, on_mouse, (tile_w, tile_h, args.cols, args.rows, x_offset, y_offset, back_rect, next_rect))

        caps = [cv2.VideoCapture(p) if p else None for p in group_files]
        delay_ms = max(1, int(1000 / args.fps))
        
        dialog_active, dialog_name, dialog_w, dialog_h, dialog_input = False, "Select Class", 600, 400, ""
        
        while True:
            # Check if window was closed
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    for cap in caps:
                        if cap:
                            cap.release()
                    cv2.destroyAllWindows()
                    print(f"Done. Labels saved to {labels_file}")
                    return
            except:
                for cap in caps:
                    if cap:
                        cap.release()
                cv2.destroyAllWindows()
                print(f"Done. Labels saved to {labels_file}")
                return
            
            if selected_video_idx is not None and not dialog_active:
                dialog_active, dialog_input = True, ""
                sw, sh = get_screen_size()
                cv2.namedWindow(dialog_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(dialog_name, dialog_w, dialog_h)
                cv2.moveWindow(dialog_name, (sw - dialog_w) // 2, (sh - dialog_h) // 2)
            
            grid = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

            cv2.rectangle(grid, (0, TOP_MARGIN), (screen_w, TOP_MARGIN + HEADER_H), (0, 0, 0), -1)
            cv2.putText(grid, f"Page {page_idx + 1}/{num_pages} - Click=classify F=filter N=next B=back Q=quit",
                       (8, TOP_MARGIN + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            for rect, text in [(back_rect, "Back (B)"), (next_rect, "Next (N)")]:
                cv2.rectangle(grid, (rect[0], rect[1]), (rect[2], rect[3]), (60, 60, 60), -1)
                cv2.rectangle(grid, (rect[0], rect[1]), (rect[2], rect[3]), (200, 200, 200), 1)
                cv2.putText(grid, text, (rect[0] + 10, rect[3] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            for i in range(PAGE_SIZE):
                r = i // args.cols
                c = i % args.cols
                y0, y1 = y_offset + r * tile_h, y_offset + (r + 1) * tile_h
                x0, x1 = x_offset + c * tile_w, x_offset + (c + 1) * tile_w

                tile = black.copy()
                cap = caps[i]
                if cap is not None and cap.isOpened():
                    ok, frame = cap.read()
                    if not ok:
                        # Loop when video ends
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = cap.read()
                    if ok:
                        tile = cv2.resize(frame, (tile_w, tile_h))

                stem = group_stems[i]
                if stem:
                    label_key = f"{stem}_{folder_name}"
                    
                    cv2.rectangle(tile, (0, tile_h - 18), (tile_w, tile_h), (0, 0, 0), -1)
                    cv2.putText(tile, truncate_caption(stem, tile_w), (4, tile_h - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    if label_key in labeled:
                        class_name = labeled[label_key]
                        class_code = CLASS_NAMES.get(class_name, 0)
                        color = CLASS_COLORS.get(class_code, (80, 80, 80))
                        
                        cv2.rectangle(tile, (0, 0), (tile_w - 1, tile_h - 1), color, 3)
                        cv2.rectangle(tile, (0, 0), (tile_w, 22), color, -1)
                        text_color = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
                        cv2.putText(tile, class_name[:12], (4, 15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)
                    else:
                        cv2.rectangle(tile, (0, 0), (tile_w - 1, tile_h - 1), (80, 80, 80), 1)

                grid[y0:y1, x0:x1] = tile

            cv2.imshow(window_name, grid)
            
            if dialog_active:
                cv2.imshow(dialog_name, create_dialog_image(dialog_w, dialog_h, dialog_input))
            
            key = cv2.waitKey(delay_ms) & 0xFF
            
            if dialog_active:
                # Check if dialog window was closed
                try:
                    if cv2.getWindowProperty(dialog_name, cv2.WND_PROP_VISIBLE) < 1:
                        dialog_active, selected_video_idx = False, None
                        try:
                            cv2.destroyWindow(dialog_name)
                        except:
                            pass
                        continue
                except:
                    dialog_active, selected_video_idx = False, None
                    continue
                
                if key == 27:
                    dialog_active, selected_video_idx = False, None
                    cv2.destroyWindow(dialog_name)
                elif key == 13 and dialog_input:
                    try:
                        class_num = int(dialog_input)
                        if class_num in CLASSES:
                            stem = group_stems[selected_video_idx]
                            label_key = f"{stem}_{folder_name}"
                            if class_num == 0:
                                # Deselect - remove from labeled
                                if label_key in labeled:
                                    del labeled[label_key]
                                    print(f"[DESELECTED] {label_key}")
                            else:
                                class_name = CLASSES[class_num]
                                labeled[label_key] = class_name
                                print(f"[{class_name.upper()}] {label_key}")
                            save_labels(str(labels_file), labeled)
                            dialog_active, selected_video_idx = False, None
                            cv2.destroyWindow(dialog_name)
                        else:
                            dialog_input = ""
                    except ValueError:
                        dialog_input = ""
                elif key == 8:
                    dialog_input = dialog_input[:-1]
                elif 48 <= key <= 57:
                    dialog_input += chr(key)
                continue
            
            if key == ord('f'):
                def update_grid():
                    g = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
                    cv2.rectangle(g, (0, TOP_MARGIN), (screen_w, TOP_MARGIN + HEADER_H), (0, 0, 0), -1)
                    cv2.putText(g, f"Page {page_idx + 1}/{num_pages} - Click=classify F=filter N=next B=back Q=quit",
                               (8, TOP_MARGIN + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    for rect, text in [(back_rect, "Back (B)"), (next_rect, "Next (N)")]:
                        cv2.rectangle(g, (rect[0], rect[1]), (rect[2], rect[3]), (60, 60, 60), -1)
                        cv2.rectangle(g, (rect[0], rect[1]), (rect[2], rect[3]), (200, 200, 200), 1)
                        cv2.putText(g, text, (rect[0] + 10, rect[3] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    for i in range(PAGE_SIZE):
                        r = i // args.cols
                        c = i % args.cols
                        y0, y1 = y_offset + r * tile_h, y_offset + (r + 1) * tile_h
                        x0, x1 = x_offset + c * tile_w, x_offset + (c + 1) * tile_w
                        tile = black.copy()
                        cap = caps[i]
                        if cap is not None and cap.isOpened():
                            ok, frame = cap.read()
                            if not ok:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                ok, frame = cap.read()
                            if ok:
                                tile = cv2.resize(frame, (tile_w, tile_h))
                        stem = group_stems[i]
                        if stem:
                            label_key = f"{stem}_{folder_name}"
                            cv2.rectangle(tile, (0, tile_h - 18), (tile_w, tile_h), (0, 0, 0), -1)
                            cv2.putText(tile, truncate_caption(stem, tile_w), (4, tile_h - 5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                            if label_key in labeled:
                                class_name = labeled[label_key]
                                class_code = CLASS_NAMES.get(class_name, 0)
                                color = CLASS_COLORS.get(class_code, (80, 80, 80))
                                cv2.rectangle(tile, (0, 0), (tile_w - 1, tile_h - 1), color, 3)
                                cv2.rectangle(tile, (0, 0), (tile_w, 22), color, -1)
                                text_color = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
                                cv2.putText(tile, class_name[:12], (4, 15),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)
                            else:
                                cv2.rectangle(tile, (0, 0), (tile_w - 1, tile_h - 1), (80, 80, 80), 1)
                        g[y0:y1, x0:x1] = tile
                    cv2.imshow(window_name, g)
                
                show_filtering_window(folder_name, videos_dir, labels_file, screen_w, screen_h, caps, update_grid, delay_ms)
                continue
            
            if nav_action == 'next' or key == ord('n') or key == 13:
                page_delta, nav_action = 1, None
                break
            if nav_action == 'back' or key == ord('b'):
                page_delta, nav_action = -1, None
                break
            if key == ord('q') or key == 27:
                for cap in caps:
                    if cap:
                        cap.release()
                cv2.destroyAllWindows()
                print(f"Done. Labels saved to {labels_file}")
                return

        for cap in caps:
            if cap:
                cap.release()

        if 'page_delta' in locals():
            page_idx += page_delta if page_delta == 1 else -1 if page_idx > 0 else 0
            del page_delta
        else:
            page_idx += 1

    cv2.destroyAllWindows()
    print(f"Done. Labels saved to {labels_file}")


if __name__ == "__main__":
    main()
