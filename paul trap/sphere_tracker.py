import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import csv
import math
from collections import defaultdict
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


MIN_BLOB_AREA = 100
DEFAULT_HUE_MIN = 45
DEFAULT_HUE_MAX = 75
DEFAULT_SAT_MIN = 195
DEFAULT_VAL_MIN = 27
PLAYBACK_INTERVAL_MS = 30  # ~33 fps max
SEARCH_RADIUS = 50  # radius of per-sphere local search window (pixels)

# Color palette for up to 6 spheres
_SPHERE_COLORS_BGR = [
    (0, 220, 0), (0, 100, 255), (255, 100, 0),
    (255, 0, 220), (0, 220, 220), (0, 165, 255),
]
_SPHERE_COLORS_HEX = [
    "#00dc00", "#ff6400", "#0064ff",
    "#dc00dc", "#dcdc00", "#ffa500",
]


class RobustParticleTracker:
    """
    Particle tracker using Normalized Cross-Correlation (NCC) template matching
    with adaptive template updating. Fast (~0.2ms), lighting-invariant, robust
    against scale drift, and has zero external dependencies (built into OpenCV core).
    Works on both automatically detected blobs and manually drawn particle boxes.
    """
    def __init__(self):
        self.bbox = (0, 0, 0, 0)
        self.w = 0
        self.h = 0
        self.cx = 0
        self.cy = 0
        self.template = None
        self.template_gray = None
        self.lost_count = 0
        self.max_lost = 12

    def init(self, frame, bbox):
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = frame.shape[:2]
        x = max(0, min(x, fw - 2))
        y = max(0, min(y, fh - 2))
        w = max(4, min(w, fw - x))
        h = max(4, min(h, fh - y))
        self.bbox = (x, y, w, h)
        self.w = w
        self.h = h
        self.cx = x + w // 2
        self.cy = y + h // 2
        self.template = frame[y:y+h, x:x+w].copy()
        self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        self.lost_count = 0
        return True

    def update(self, frame):
        if self.template_gray is None:
            return False, self.bbox

        fh, fw = frame.shape[:2]
        s_radius = max(50, max(self.w, self.h) * 2)
        x1 = max(0, self.cx - s_radius)
        y1 = max(0, self.cy - s_radius)
        x2 = min(fw, self.cx + s_radius)
        y2 = min(fh, self.cy + s_radius)

        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] < self.h or crop.shape[1] < self.w:
            self.lost_count += 1
            return (self.lost_count <= self.max_lost), self.bbox

        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(crop_gray, self.template_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val >= 0.25:
            match_x = x1 + max_loc[0]
            match_y = y1 + max_loc[1]
            self.cx = match_x + self.w // 2
            self.cy = match_y + self.h // 2
            self.bbox = (match_x, match_y, self.w, self.h)
            self.lost_count = 0

            # Subtle adaptive template update to handle gradual lighting changes
            new_patch = frame[match_y:match_y+self.h, match_x:match_x+self.w]
            if new_patch.shape == self.template.shape:
                self.template = cv2.addWeighted(self.template, 0.95, new_patch, 0.05, 0)
                self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
            return True, self.bbox
        else:
            self.lost_count += 1
            return (self.lost_count <= self.max_lost), self.bbox


def _make_tracker():
    """Create a tracker instance."""
    return RobustParticleTracker()


def _format_time(seconds):
    """Format seconds into MM:SS.s (or HH:MM:SS.s if >= 1 hour)."""
    if seconds < 0:
        seconds = 0
    m = int(seconds // 60)
    s = seconds % 60
    if m >= 60:
        h = m // 60
        m = m % 60
        return f"{h:02d}:{m:02d}:{s:04.1f}"
    return f"{m:02d}:{s:04.1f}"


class TimelineWidget(tk.Frame):
    """
    Media player timeline scrubber widget.
    Shows video playback progress, start/end range markers, and hover time preview.
    Supports clicking, dragging, and mousewheel scrubbing.
    """
    def __init__(self, parent, on_seek=None, bg="#242424"):
        super().__init__(parent, bg=bg)
        self.on_seek = on_seek
        self.total_frames = 0
        self.current_frame = 0
        self.start_frame = 0
        self.end_frame = 0
        self.fps = 30.0

        self.padding = 16
        self._dragging = False
        self._hover_x = None
        self._hover_thumb = False

        self.canvas = tk.Canvas(
            self, height=34, bg="#1e1e1e",
            highlightthickness=0, cursor="hand2"
        )
        self.canvas.pack(fill=tk.X, expand=True, padx=4, pady=1)

        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", self._on_leave)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)

    def set_range(self, total_frames, fps=30.0):
        self.total_frames = max(0, total_frames)
        self.fps = max(1.0, fps)
        self.current_frame = 0
        self._draw()

    def set_position(self, frame):
        frame = max(0, min(frame, max(0, self.total_frames - 1)))
        if frame != self.current_frame:
            self.current_frame = frame
            self._draw()

    def set_markers(self, start_frame, end_frame):
        self.start_frame = start_frame
        self.end_frame = end_frame
        self._draw()

    def set_start_marker(self, start_frame):
        self.start_frame = start_frame
        self._draw()

    def set_end_marker(self, end_frame):
        self.end_frame = end_frame
        self._draw()

    def _frame_to_pixel(self, frame):
        w = max(self.canvas.winfo_width(), 100)
        avail = w - 2 * self.padding
        total_steps = max(1, self.total_frames - 1)
        if avail <= 0 or self.total_frames <= 1:
            return self.padding
        frac = max(0.0, min(1.0, frame / total_steps))
        return self.padding + frac * avail

    def _pixel_to_frame(self, x):
        w = max(self.canvas.winfo_width(), 100)
        avail = w - 2 * self.padding
        total_steps = max(1, self.total_frames - 1)
        if avail <= 0 or self.total_frames <= 1:
            return 0
        frac = max(0.0, min(1.0, (x - self.padding) / avail))
        return int(round(frac * total_steps))

    def _on_configure(self, event):
        self._draw()

    def _on_press(self, event):
        try:
            self.winfo_toplevel().focus_set()
        except Exception:
            pass
        if self.total_frames <= 0:
            return
        self._dragging = True
        frame = self._pixel_to_frame(event.x)
        self.current_frame = frame
        self._draw()
        if self.on_seek:
            self.on_seek(frame, is_dragging=True)

    def _on_drag(self, event):
        if not self._dragging or self.total_frames <= 0:
            return
        frame = self._pixel_to_frame(event.x)
        if frame != self.current_frame:
            self.current_frame = frame
            self._draw()
            if self.on_seek:
                self.on_seek(frame, is_dragging=True)

    def _on_release(self, event):
        if not self._dragging:
            return
        self._dragging = False
        frame = self._pixel_to_frame(event.x)
        self.current_frame = frame
        self._draw()
        if self.on_seek:
            self.on_seek(frame, is_dragging=False)

    def _on_motion(self, event):
        if self.total_frames <= 0:
            return
        self._hover_x = event.x
        cx = self._frame_to_pixel(self.current_frame)
        self._hover_thumb = abs(event.x - cx) <= 10
        self._draw()

    def _on_leave(self, event):
        self._hover_x = None
        self._hover_thumb = False
        self._draw()

    def _on_mousewheel(self, event):
        if self.total_frames <= 0:
            return
        delta = 1 if event.delta > 0 else -1
        if event.state & 0x0001:  # Shift key pressed
            delta *= 10
        new_f = max(0, min(self.total_frames - 1, self.current_frame + delta))
        if new_f != self.current_frame:
            self.current_frame = new_f
            self._draw()
            if self.on_seek:
                self.on_seek(new_f, is_dragging=False)

    def _draw(self):
        w = max(self.canvas.winfo_width(), 1)
        h = max(self.canvas.winfo_height(), 1)
        if w <= 1 or h <= 1:
            return

        self.canvas.delete("all")
        y_track = 20
        x0 = self.padding
        x1 = max(x0 + 1, w - self.padding)

        if self.total_frames <= 0:
            self.canvas.create_rectangle(x0, y_track - 3, x1, y_track + 3, fill="#383838", outline="")
            return

        # Base track
        self.canvas.create_rectangle(x0, y_track - 3, x1, y_track + 3, fill="#383838", outline="")

        # Selected range highlight (Start to End)
        sx = self._frame_to_pixel(self.start_frame)
        ex = self._frame_to_pixel(self.end_frame)
        if ex >= sx:
            self.canvas.create_rectangle(sx, y_track - 3, ex, y_track + 3, fill="#1e3a8a", outline="")

        # Progress bar (0 to Current Frame)
        cx = self._frame_to_pixel(self.current_frame)
        self.canvas.create_rectangle(x0, y_track - 3, cx, y_track + 3, fill="#00e5ff", outline="")

        # Start Marker (Green)
        self.canvas.create_line(sx, y_track - 10, sx, y_track + 8, fill="#22c55e", width=2)
        self.canvas.create_polygon(sx - 4, y_track - 10, sx + 4, y_track - 10, sx, y_track - 5, fill="#22c55e", outline="")
        self.canvas.create_text(sx, y_track - 14, text="S", fill="#22c55e", font=("Segoe UI", 7, "bold"))

        # End Marker (Red)
        self.canvas.create_line(ex, y_track - 10, ex, y_track + 8, fill="#ef4444", width=2)
        self.canvas.create_polygon(ex - 4, y_track - 10, ex + 4, y_track - 10, ex, y_track - 5, fill="#ef4444", outline="")
        self.canvas.create_text(ex, y_track - 14, text="E", fill="#ef4444", font=("Segoe UI", 7, "bold"))

        # Scrubber Thumb
        r = 8 if (self._dragging or self._hover_thumb) else 6
        self.canvas.create_oval(cx - (r + 2), y_track - (r + 2), cx + (r + 2), y_track + (r + 2), fill="#00e5ff", outline="")
        self.canvas.create_oval(cx - r, y_track - r, cx + r, y_track + r, fill="#ffffff", outline="")

        # Hover Guideline & Tooltip
        if self._hover_x is not None:
            hx = max(x0, min(x1, self._hover_x))
            self.canvas.create_line(hx, 4, hx, h - 2, fill="#94a3b8", dash=(2, 2))
            hf = self._pixel_to_frame(hx)
            ht = _format_time(hf / self.fps)
            txt = f"{ht}  (Fr {hf})"
            tx = max(x0 + 40, min(x1 - 40, hx))
            tw = 45
            self.canvas.create_rectangle(tx - tw, 1, tx + tw, 13, fill="#0f172a", outline="#38bdf8", width=1)
            self.canvas.create_text(tx, 7, text=txt, fill="#38bdf8", font=("Consolas", 7, "bold"))


class SphereTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Green Sphere Video Tracker")
        self.root.geometry("1150x840")

        self.cap = None
        self.total_frames = 0
        self.current_frame_idx = 0
        self.raw_frame = None
        self.video_path = ""
        self.fps = 30.0
        self._playing = False
        self._previewing = False
        self._preview_after_id = None
        self._use_cv2_tracker = _make_tracker() is not None

        self.trackers = []   # list of tracker-info dicts
        self.next_id = 1
        self.records = []    # list of dicts: frame, sphere_id, x_px, y_px

        self.hue_min = tk.IntVar(value=DEFAULT_HUE_MIN)
        self.hue_max = tk.IntVar(value=DEFAULT_HUE_MAX)
        self.sat_min = tk.IntVar(value=DEFAULT_SAT_MIN)
        self.val_min = tk.IntVar(value=DEFAULT_VAL_MIN)
        self.min_area_var = tk.IntVar(value=MIN_BLOB_AREA)
        self.start_frame_var = tk.IntVar(value=0)
        self.end_frame_var = tk.IntVar(value=0)

        self.tk_image = None
        self.scale = 1.0

        # ROI selection state (x, y, w, h in original frame pixels; None = full frame)
        self.roi = None
        self._roi_mode = False
        self._roi_start = None   # canvas coords of press
        self._roi_rect_id = None  # temporary canvas rectangle item

        # Manual tracker-draw state
        self._tracker_draw_mode = False
        self._tracker_draw_rect_id = None

        self._build_ui()
        for var in (self.hue_min, self.hue_max, self.sat_min, self.val_min, self.min_area_var):
            var.trace_add("write", self._on_hsv_change)
        self.start_frame_var.trace_add("write", self._on_start_var_change)
        self.end_frame_var.trace_add("write", self._on_end_var_change)
        self._bind_shortcuts()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Row 1: file picker + playback controls
        row1 = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        row1.pack(side=tk.TOP, fill=tk.X)

        tk.Button(row1, text="Open Video", command=self.open_video).pack(side=tk.LEFT, padx=4, pady=2)
        self.file_label = tk.Label(row1, text="No video loaded", width=32, anchor=tk.W)
        self.file_label.pack(side=tk.LEFT, padx=4)

        _sep(row1)

        self.init_btn = tk.Button(row1, text="Initialize Trackers",
                                  command=self.initialize, state=tk.DISABLED)
        self.init_btn.pack(side=tk.LEFT, padx=4, pady=2)

        self.draw_tracker_btn = tk.Button(row1, text="✚ Draw Tracker",
                                          command=self.toggle_tracker_draw_mode,
                                          state=tk.DISABLED)
        self.draw_tracker_btn.pack(side=tk.LEFT, padx=2, pady=2)

        self.clear_trackers_btn = tk.Button(row1, text="Clear Trackers",
                                            command=self.clear_trackers,
                                            state=tk.DISABLED)
        self.clear_trackers_btn.pack(side=tk.LEFT, padx=2, pady=2)

        self.roi_btn = tk.Button(row1, text="Select ROI", command=self.toggle_roi_mode,
                                 state=tk.DISABLED)
        self.roi_btn.pack(side=tk.LEFT, padx=2, pady=2)
        self.clear_roi_btn = tk.Button(row1, text="Clear ROI", command=self.clear_roi,
                                       state=tk.DISABLED)
        self.clear_roi_btn.pack(side=tk.LEFT, padx=2, pady=2)

        self.play_btn = tk.Button(row1, text="▶  Track", command=self.play, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=2, pady=2)
        self.stop_btn = tk.Button(row1, text="⏹  Stop", command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=2, pady=2)

        _sep(row1)

        self.export_btn = tk.Button(row1, text="Export CSV + Plot",
                                    command=self.export, state=tk.DISABLED)
        self.export_btn.pack(side=tk.LEFT, padx=4, pady=2)

        # Row 2: time range + HSV sliders
        row2 = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        row2.pack(side=tk.TOP, fill=tk.X)

        tk.Label(row2, text="Start frame:").pack(side=tk.LEFT, padx=(6, 1), pady=2)
        self.start_spin = tk.Spinbox(row2, from_=0, to=999999,
                                     textvariable=self.start_frame_var, width=7,
                                     command=self._on_range_spinbox)
        self.start_spin.pack(side=tk.LEFT, padx=2, pady=2)
        self.start_spin.bind("<Return>", self._on_range_spinbox)

        tk.Label(row2, text="End frame:").pack(side=tk.LEFT, padx=(6, 1), pady=2)
        self.end_spin = tk.Spinbox(row2, from_=0, to=999999,
                                   textvariable=self.end_frame_var, width=7,
                                   command=self._on_range_spinbox)
        self.end_spin.pack(side=tk.LEFT, padx=2, pady=2)
        self.end_spin.bind("<Return>", self._on_range_spinbox)

        _sep(row2)

        tk.Label(row2, text="Hue:").pack(side=tk.LEFT)
        tk.Scale(row2, from_=0, to=179, orient=tk.HORIZONTAL, variable=self.hue_min,
                 length=70, label="min", showvalue=True).pack(side=tk.LEFT)
        tk.Scale(row2, from_=0, to=179, orient=tk.HORIZONTAL, variable=self.hue_max,
                 length=70, label="max", showvalue=True).pack(side=tk.LEFT)

        tk.Label(row2, text=" Sat≥").pack(side=tk.LEFT)
        tk.Scale(row2, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.sat_min,
                 length=70, showvalue=True).pack(side=tk.LEFT)

        tk.Label(row2, text=" Val≥").pack(side=tk.LEFT)
        tk.Scale(row2, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.val_min,
                 length=70, showvalue=True).pack(side=tk.LEFT)

        self.swatch = tk.Label(row2, text="  color  ", width=8, relief=tk.SUNKEN)
        self.swatch.pack(side=tk.LEFT, padx=6)
        self._update_swatch()

        tk.Label(row2, text="  Area≥").pack(side=tk.LEFT)
        tk.Spinbox(row2, from_=10, to=100000, textvariable=self.min_area_var,
                   width=8).pack(side=tk.LEFT, padx=2)

        # Status bar (at very bottom)
        self.status_var = tk.StringVar(value="Open a video file to start.")
        status_bar = tk.Label(self.root, textvariable=self.status_var, anchor=tk.W,
                              relief=tk.SUNKEN, bd=1)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Media Player Timeline Frame (directly above status bar)
        self.timeline_frame = tk.Frame(self.root, bd=1, relief=tk.RAISED, bg="#242424")
        self.timeline_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.timeline = TimelineWidget(self.timeline_frame, on_seek=self._on_timeline_seek)
        self.timeline.pack(fill=tk.X, expand=True)

        transport_bar = tk.Frame(self.timeline_frame, bg="#2d2d2d", bd=1, relief=tk.FLAT)
        transport_bar.pack(fill=tk.X, padx=4, pady=(1, 3))

        self.seek_start_btn = tk.Button(transport_bar, text="⏮", width=3,
                                        command=self.seek_to_start, state=tk.DISABLED)
        self.seek_start_btn.pack(side=tk.LEFT, padx=1, pady=1)

        self.skip_back_btn = tk.Button(transport_bar, text="⏪ -1s",
                                       command=lambda: self.skip_seconds(-1.0), state=tk.DISABLED)
        self.skip_back_btn.pack(side=tk.LEFT, padx=1, pady=1)

        self.step_back_btn = tk.Button(transport_bar, text="◀ 1f",
                                       command=lambda: self.step_frames(-1), state=tk.DISABLED)
        self.step_back_btn.pack(side=tk.LEFT, padx=1, pady=1)

        self.preview_play_btn = tk.Button(transport_bar, text="▶ Play", width=7,
                                          font=("Segoe UI", 9, "bold"),
                                          command=self.toggle_play_preview, state=tk.DISABLED)
        self.preview_play_btn.pack(side=tk.LEFT, padx=3, pady=1)

        self.step_fwd_btn = tk.Button(transport_bar, text="1f ▶",
                                      command=lambda: self.step_frames(1), state=tk.DISABLED)
        self.step_fwd_btn.pack(side=tk.LEFT, padx=1, pady=1)

        self.skip_fwd_btn = tk.Button(transport_bar, text="+1s ⏩",
                                      command=lambda: self.skip_seconds(1.0), state=tk.DISABLED)
        self.skip_fwd_btn.pack(side=tk.LEFT, padx=1, pady=1)

        self.seek_end_btn = tk.Button(transport_bar, text="⏭", width=3,
                                      command=self.seek_to_end, state=tk.DISABLED)
        self.seek_end_btn.pack(side=tk.LEFT, padx=1, pady=1)

        _sep(transport_bar)

        self.time_label = tk.Label(transport_bar, text="--:-- / --:--  •  Frame: -- / --",
                                   font=("Consolas", 9), anchor=tk.W, bg="#2d2d2d", fg="#e2e8f0")
        self.time_label.pack(side=tk.LEFT, padx=4)

        _sep(transport_bar)

        tk.Label(transport_bar, text="Frame:", bg="#2d2d2d", fg="#cbd5e1").pack(side=tk.LEFT, padx=(2, 1))
        self.jump_entry = tk.Entry(transport_bar, width=6, justify=tk.CENTER, state=tk.DISABLED)
        self.jump_entry.pack(side=tk.LEFT, padx=1)
        self.jump_entry.bind("<Return>", self._on_jump_go)
        self.jump_btn = tk.Button(transport_bar, text="Go", command=self._on_jump_go, state=tk.DISABLED)
        self.jump_btn.pack(side=tk.LEFT, padx=1)

        _sep(transport_bar)

        self.set_start_btn = tk.Button(transport_bar, text="⚑ Set Start",
                                       command=self.set_current_as_start, state=tk.DISABLED)
        self.set_start_btn.pack(side=tk.LEFT, padx=2, pady=1)

        self.to_start_btn = tk.Button(transport_bar, text="|◀ To Start",
                                      command=self.jump_to_start_marker, state=tk.DISABLED)
        self.to_start_btn.pack(side=tk.LEFT, padx=1, pady=1)

        self.to_end_btn = tk.Button(transport_bar, text="To End ▶|",
                                    command=self.jump_to_end_marker, state=tk.DISABLED)
        self.to_end_btn.pack(side=tk.LEFT, padx=1, pady=1)

        self.set_end_btn = tk.Button(transport_bar, text="⚐ Set End",
                                     command=self.set_current_as_end, state=tk.DISABLED)
        self.set_end_btn.pack(side=tk.LEFT, padx=2, pady=1)

        # Canvas (fills remaining center space)
        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="#2b2b2b", cursor="crosshair")
        vscroll = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        hscroll = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vscroll.set, xscrollcommand=hscroll.set)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        hscroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._roi_press)
        self.canvas.bind("<B1-Motion>", self._roi_drag)
        self.canvas.bind("<ButtonRelease-1>", self._roi_release)

    # ── Video loading ─────────────────────────────────────────────────────────

    def open_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v"), ("All files", "*.*")]
        )
        if not path:
            return
        if self._previewing:
            self.pause_preview()
        if self._playing:
            self.stop()
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Could not open video:\n{path}")
            self.cap = None
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames <= 0:
            self.total_frames = 1
        fps_val = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps_val if (fps_val and fps_val > 0 and not math.isnan(fps_val)) else 30.0

        self.video_path = path
        self.trackers = []
        self.records = []
        self._playing = False
        self._previewing = False
        self.roi = None
        self._roi_mode = False
        self._tracker_draw_mode = False

        end = max(0, self.total_frames - 1)
        self.start_frame_var.set(0)
        self.end_frame_var.set(end)
        self.start_spin.config(to=end)
        self.end_spin.config(to=end)

        name = os.path.basename(path)
        self.file_label.config(text=f"{name}  ({self.total_frames} fr)")
        self.root.title(f"Green Sphere Video Tracker — {name}")

        self.init_btn.config(state=tk.NORMAL)
        self.draw_tracker_btn.config(state=tk.NORMAL)
        self.roi_btn.config(state=tk.NORMAL)
        self.clear_trackers_btn.config(state=tk.DISABLED)
        self.play_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self.export_btn.config(state=tk.DISABLED)

        self.timeline.set_range(self.total_frames, self.fps)
        self.timeline.set_markers(0, end)
        self._enable_timeline_controls(True)

        self._seek(0)
        tracker_info = "NCC template + centroid"
        self.status_var.set(f"Loaded {name}. Tracker: {tracker_info}. Use timeline to skip frames or click ▶ Play to preview.")

    # ── Seeking + display ─────────────────────────────────────────────────────

    def _seek(self, n, from_timeline=False):
        if self.cap is None:
            return
        n = max(0, min(n, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, frame = self.cap.read()
        if ok:
            self.raw_frame = frame
            self.current_frame_idx = n
            self._show_frame(self._annotate_detections(frame))
            if not from_timeline:
                self.timeline.set_position(n)
            self._update_time_label()

    def _on_timeline_seek(self, frame, is_dragging=False):
        if self._playing:
            self.stop()
        if self._previewing and is_dragging:
            self.pause_preview()
        self._seek(frame, from_timeline=True)

    def _on_start_var_change(self, *_):
        try:
            val = self.start_frame_var.get()
            self.timeline.set_start_marker(val)
        except (tk.TclError, ValueError):
            pass

    def _on_end_var_change(self, *_):
        try:
            val = self.end_frame_var.get()
            self.timeline.set_end_marker(val)
        except (tk.TclError, ValueError):
            pass

    def _update_time_label(self):
        if self.cap is None or self.total_frames <= 0:
            self.time_label.config(text="--:-- / --:--  •  Frame: -- / --")
            return
        cur = self.current_frame_idx
        end = max(0, self.total_frames - 1)
        cur_sec = cur / self.fps if self.fps > 0 else 0
        tot_sec = end / self.fps if self.fps > 0 else 0
        cur_str = _format_time(cur_sec)
        tot_str = _format_time(tot_sec)
        self.time_label.config(
            text=f"{cur_str} / {tot_str}  •  Frame: {cur} / {end}  ({self.fps:.1f} fps)"
        )
        if self.root.focus_get() != self.jump_entry:
            self.jump_entry.config(state=tk.NORMAL)
            self.jump_entry.delete(0, tk.END)
            self.jump_entry.insert(0, str(cur))

    def _enable_timeline_controls(self, enabled=True):
        state = tk.NORMAL if enabled else tk.DISABLED
        for btn in (
            self.preview_play_btn, self.seek_start_btn, self.skip_back_btn,
            self.step_back_btn, self.step_fwd_btn, self.skip_fwd_btn,
            self.seek_end_btn, self.jump_btn, self.set_start_btn,
            self.set_end_btn, self.to_start_btn, self.to_end_btn,
        ):
            btn.config(state=state)
        self.jump_entry.config(state=state)

    def toggle_play_preview(self):
        if self.cap is None:
            return
        if self._playing:
            self.stop()
        elif self._previewing:
            self.pause_preview()
        else:
            self.start_preview()

    def start_preview(self):
        if self.cap is None:
            return
        if self._playing:
            self.stop()
        if self.current_frame_idx >= max(0, self.total_frames - 1):
            self._seek(0)
        self._previewing = True
        self.preview_play_btn.config(text="⏸ Pause")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        self.status_var.set("Playing preview. Press Space or ⏸ Pause to pause.")
        self._preview_step()

    def pause_preview(self):
        self._previewing = False
        if self._preview_after_id:
            try:
                self.root.after_cancel(self._preview_after_id)
            except Exception:
                pass
            self._preview_after_id = None
        self.preview_play_btn.config(text="▶ Play")
        self.status_var.set(f"Preview paused at frame {self.current_frame_idx}.")

    def _preview_step(self):
        if not self._previewing or self.cap is None:
            return
        end = max(0, self.total_frames - 1)
        if self.current_frame_idx >= end:
            self.pause_preview()
            self.status_var.set(f"Reached end of video (frame {end}).")
            return

        ok, frame = self.cap.read()
        if not ok:
            self.pause_preview()
            return

        self.current_frame_idx += 1
        self.raw_frame = frame
        self._show_frame(self._annotate_detections(frame))
        self.timeline.set_position(self.current_frame_idx)
        self._update_time_label()

        delay = max(10, int(1000.0 / self.fps))
        self._preview_after_id = self.root.after(delay, self._preview_step)

    def step_frames(self, delta):
        if self.cap is None:
            return
        if self._playing:
            self.stop()
        if self._previewing:
            self.pause_preview()
        self._seek(self.current_frame_idx + delta)

    def skip_seconds(self, sec):
        if self.cap is None:
            return
        if self._playing:
            self.stop()
        if self._previewing:
            self.pause_preview()
        delta_frames = int(round(sec * self.fps))
        if delta_frames == 0:
            delta_frames = 1 if sec > 0 else -1
        self._seek(self.current_frame_idx + delta_frames)

    def seek_to_start(self):
        if self.cap is None:
            return
        if self._playing:
            self.stop()
        if self._previewing:
            self.pause_preview()
        self._seek(0)

    def seek_to_end(self):
        if self.cap is None:
            return
        if self._playing:
            self.stop()
        if self._previewing:
            self.pause_preview()
        self._seek(max(0, self.total_frames - 1))

    def jump_to_start_marker(self):
        if self.cap is None:
            return
        if self._playing:
            self.stop()
        if self._previewing:
            self.pause_preview()
        self._seek(self.start_frame_var.get())

    def jump_to_end_marker(self):
        if self.cap is None:
            return
        if self._playing:
            self.stop()
        if self._previewing:
            self.pause_preview()
        self._seek(self.end_frame_var.get())

    def set_current_as_start(self):
        if self.cap is None:
            return
        cur = self.current_frame_idx
        end = self.end_frame_var.get()
        if cur > end:
            self.end_frame_var.set(cur)
        self.start_frame_var.set(cur)
        self.timeline.set_start_marker(cur)
        self.status_var.set(f"Start frame set to {cur}.")

    def set_current_as_end(self):
        if self.cap is None:
            return
        cur = self.current_frame_idx
        start = self.start_frame_var.get()
        if cur < start:
            self.start_frame_var.set(cur)
        self.end_frame_var.set(cur)
        self.timeline.set_end_marker(cur)
        self.status_var.set(f"End frame set to {cur}.")

    def _on_jump_go(self, *_):
        if self.cap is None:
            return
        txt = self.jump_entry.get().strip()
        try:
            val = int(txt)
            if self._playing:
                self.stop()
            if self._previewing:
                self.pause_preview()
            self._seek(val)
        except ValueError:
            pass

    def _bind_shortcuts(self):
        self.root.bind("<Left>", lambda e: self._on_key_step(-1))
        self.root.bind("<Right>", lambda e: self._on_key_step(1))
        self.root.bind("<Shift-Left>", lambda e: self._on_key_step(-10))
        self.root.bind("<Shift-Right>", lambda e: self._on_key_step(10))
        self.root.bind("<Control-Left>", lambda e: self._on_key_skip(-1.0))
        self.root.bind("<Control-Right>", lambda e: self._on_key_skip(1.0))
        self.root.bind("<Home>", lambda e: self._on_key_home())
        self.root.bind("<End>", lambda e: self._on_key_end())
        self.root.bind("<space>", lambda e: self._on_key_space())

    def _is_entry_focused(self):
        widget = self.root.focus_get()
        return isinstance(widget, (tk.Entry, tk.Spinbox))

    def _on_key_step(self, delta):
        if not self._is_entry_focused():
            self.step_frames(delta)

    def _on_key_skip(self, sec):
        if not self._is_entry_focused():
            self.skip_seconds(sec)

    def _on_key_home(self):
        if not self._is_entry_focused():
            self.seek_to_start()

    def _on_key_end(self):
        if not self._is_entry_focused():
            self.seek_to_end()

    def _on_key_space(self):
        if not self._is_entry_focused():
            self.toggle_play_preview()

    def _on_close(self):
        self._playing = False
        self._previewing = False
        if self._preview_after_id:
            try:
                self.root.after_cancel(self._preview_after_id)
            except Exception:
                pass
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()

    def _on_range_spinbox(self, *_):
        if self.cap is not None:
            self._seek(self.start_frame_var.get())

    def _on_hsv_change(self, *_):
        self._update_swatch()
        if self.raw_frame is not None and not self._playing:
            self._show_frame(self._annotate_detections(self.raw_frame))

    def _update_swatch(self):
        h = (self.hue_min.get() + self.hue_max.get()) // 2
        s = max(self.sat_min.get(), 128)
        v = max(self.val_min.get(), 128)
        bgr = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0][0]
        self.swatch.config(bg="#{:02x}{:02x}{:02x}".format(int(bgr[2]), int(bgr[1]), int(bgr[0])))

    def _show_frame(self, cv_img):
        canvas_w = max(self.canvas.winfo_width(), 1)
        canvas_h = max(self.canvas.winfo_height(), 1)
        h, w = cv_img.shape[:2]
        self.scale = min(canvas_w / w, canvas_h / h, 1.0)
        new_w = max(1, int(w * self.scale))
        new_h = max(1, int(h * self.scale))
        interp = cv2.INTER_AREA if self.scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(cv_img, (new_w, new_h), interpolation=interp)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.tk_image = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.config(scrollregion=(0, 0, new_w, new_h))

    # ── Blob detection ────────────────────────────────────────────────────────

    def _detect_blobs(self, frame):
        """Return list of (bbox_tuple, (cx, cy)) for each qualifying green blob."""
        # Restrict to ROI if one is set
        if self.roi:
            rx, ry, rw, rh = self.roi
            search = frame[ry:ry + rh, rx:rx + rw]
            ox, oy = rx, ry
        else:
            search = frame
            ox, oy = 0, 0

        lower = np.array([self.hue_min.get(), self.sat_min.get(), self.val_min.get()])
        upper = np.array([self.hue_max.get(), 255, 255])
        hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = []
        min_area = self.min_area_var.get()
        for c in contours:
            if cv2.contourArea(c) < min_area:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"]) + ox
            cy = int(M["m01"] / M["m00"]) + oy
            bx, by, bw, bh = cv2.boundingRect(c)
            results.append(((bx + ox, by + oy, bw, bh), (cx, cy)))
        return results

    def _detect_blobs_local(self, frame, cx, cy):
        """Detect blobs within SEARCH_RADIUS pixels of (cx, cy) in original frame coords."""
        fh, fw = frame.shape[:2]
        x1 = max(0, cx - SEARCH_RADIUS)
        y1 = max(0, cy - SEARCH_RADIUS)
        x2 = min(fw, cx + SEARCH_RADIUS)
        y2 = min(fh, cy + SEARCH_RADIUS)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return []
        # Temporarily override roi so _detect_blobs uses this crop region
        saved_roi = self.roi
        self.roi = (x1, y1, x2 - x1, y2 - y1)
        results = self._detect_blobs(frame)
        self.roi = saved_roi
        return results

    def _annotate_detections(self, frame):
        """Draw detection preview overlay (used before trackers are initialized)."""
        display = frame.copy()
        self._draw_roi(display)
        for i, (bbox, (cx, cy)) in enumerate(self._detect_blobs(frame)):
            x, y, w, h = bbox
            color = _SPHERE_COLORS_BGR[i % len(_SPHERE_COLORS_BGR)]
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            cv2.circle(display, (cx, cy), 4, color, -1)
            cv2.putText(display, str(i + 1), (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(display, f"Frame {self.current_frame_idx}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return display

    # ── Tracker initialization ─────────────────────────────────────────────────

    def initialize(self):
        if self.cap is None or self.raw_frame is None:
            return
        if self._previewing:
            self.pause_preview()
        start = self.start_frame_var.get()
        self._seek(start)
        frame = self.raw_frame

        blobs = self._detect_blobs(frame)
        if not blobs:
            messagebox.showwarning(
                "No blobs found",
                "No green blobs detected on the start frame.\n"
                "Adjust the HSV sliders until the spheres are highlighted, then try again."
            )
            return

        self.trackers = []
        self.next_id = 1
        self.records = []

        for bbox, (cx, cy) in blobs:
            sid = self.next_id
            self.next_id += 1
            entry = {"id": sid, "bbox": bbox, "cx": cx, "cy": cy,
                     "lost": False, "tracker": None, "start_frame": start}
            t = _make_tracker()
            t.init(frame, bbox)
            entry["tracker"] = t
            self.trackers.append(entry)

        self.current_frame_idx = start
        self.clear_trackers_btn.config(state=tk.NORMAL)
        self.play_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)
        self._show_frame(self._annotate_tracked(frame))
        self.status_var.set(
            f"Initialized {len(self.trackers)} tracker(s) at frame {start}. Click ▶ Track to start."
        )

    # ── Playback ──────────────────────────────────────────────────────────────

    def play(self):
        if not self.trackers:
            messagebox.showinfo("Not initialized", "Click Initialize Trackers first.")
            return
        if self._previewing:
            self.pause_preview()
        self._playing = True
        self.play_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.preview_play_btn.config(text="⏹ Stop")
        # Seek cap to current position so reads are aligned
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        self._step()

    def stop(self):
        self._playing = False
        if self._previewing:
            self.pause_preview()
        self.play_btn.config(state=tk.NORMAL if self.trackers else tk.DISABLED)
        self.preview_play_btn.config(text="▶ Play")
        if self.records:
            self.export_btn.config(state=tk.NORMAL)
        self.status_var.set(
            f"Stopped at frame {self.current_frame_idx}. "
            f"{len(self.records)} data point(s) recorded."
        )

    def _step(self):
        if not self._playing:
            return
        end = self.end_frame_var.get()
        if self.current_frame_idx > end:
            self._playing = False
            self.play_btn.config(state=tk.NORMAL)
            self.preview_play_btn.config(text="▶ Play")
            self.export_btn.config(state=tk.NORMAL)
            self.status_var.set(
                f"Tracking complete — {len(self.records)} data points. Click Export CSV + Plot."
            )
            return

        ok, frame = self.cap.read()
        if not ok:
            self.stop()
            return

        frame_n = self.current_frame_idx
        self.raw_frame = frame
        self.current_frame_idx += 1

        for t_info in self.trackers:
            if t_info["lost"]:
                continue

            tracker = t_info.get("tracker")
            if tracker is not None:
                # If this frame is the frame on which the tracker was initialized,
                # keep its initial bbox and don't re-track from past
                if frame_n == t_info.get("start_frame", -1):
                    success = True
                    new_bbox = t_info["bbox"]
                    new_cx, new_cy = t_info["cx"], t_info["cy"]
                else:
                    success, new_bbox = tracker.update(frame)
                    new_cx, new_cy = tracker.cx, tracker.cy

                if success:
                    # Optional: if a qualifying green blob is near (new_cx, new_cy),
                    # refine centroid to the blob center of mass
                    local_blobs = self._detect_blobs_local(frame, new_cx, new_cy)
                    if local_blobs:
                        prev = (new_cx, new_cy)
                        best_bbox, (bcx, bcy) = min(local_blobs, key=lambda b: math.dist(b[1], prev))
                        if math.dist((bcx, bcy), prev) <= max(SEARCH_RADIUS, new_bbox[2]):
                            new_cx, new_cy = bcx, bcy
                            new_bbox = best_bbox

                    t_info["bbox"] = new_bbox
                    t_info["cx"] = new_cx
                    t_info["cy"] = new_cy
                else:
                    t_info["lost"] = True
            else:
                blobs = self._detect_blobs_local(frame, t_info["cx"], t_info["cy"])
                if blobs:
                    prev = (t_info["cx"], t_info["cy"])
                    best_bbox, (cx, cy) = min(blobs, key=lambda b: math.dist(b[1], prev))
                    t_info["bbox"] = best_bbox
                    t_info["cx"] = cx
                    t_info["cy"] = cy
                else:
                    t_info["lost"] = True

            if not t_info["lost"]:
                self.records.append({
                    "frame": frame_n,
                    "sphere_id": t_info["id"],
                    "x_px": t_info["cx"],
                    "y_px": t_info["cy"],
                })

        self._show_frame(self._annotate_tracked(frame))
        self.timeline.set_position(frame_n)
        self._update_time_label()

        active = [t for t in self.trackers if not t["lost"]]
        sphere_strs = "  |  ".join(f"#{t['id']} x={t['cx']}" for t in active)
        self.status_var.set(f"Frame {frame_n}/{end}  |  {sphere_strs}")

        self.root.after(PLAYBACK_INTERVAL_MS, self._step)

    def _annotate_tracked(self, frame):
        display = frame.copy()
        fh, fw = frame.shape[:2]
        for i, t_info in enumerate(self.trackers):
            if t_info["lost"]:
                continue
            color = _SPHERE_COLORS_BGR[i % len(_SPHERE_COLORS_BGR)]
            x, y, w, h = t_info["bbox"]
            cx, cy = t_info["cx"], t_info["cy"]
            # Moving search window (muted, thin)
            sx1 = max(0, cx - SEARCH_RADIUS)
            sy1 = max(0, cy - SEARCH_RADIUS)
            sx2 = min(fw, cx + SEARCH_RADIUS)
            sy2 = min(fh, cy + SEARCH_RADIUS)
            cv2.rectangle(display, (sx1, sy1), (sx2, sy2),
                          tuple(max(0, c - 80) for c in color), 1)
            # Tracker bounding box + centroid
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            cv2.circle(display, (cx, cy), 4, color, -1)
            cv2.putText(display, f"#{t_info['id']}  x={cx}", (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(display, f"Frame {self.current_frame_idx}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return display

    # ── Export ────────────────────────────────────────────────────────────────

    def export(self):
        if not self.records:
            messagebox.showinfo("No data", "Run tracking first (Initialize → Play).")
            return

        # Group by sphere and compute velocity + acceleration per sphere
        by_sphere = defaultdict(list)
        for r in self.records:
            by_sphere[r["sphere_id"]].append(r)

        for rows in by_sphere.values():
            rows.sort(key=lambda r: r["frame"])
            for i, row in enumerate(rows):
                if i == 0:
                    row["velocity_px_per_frame"] = ""
                    row["acceleration_px_per_frame2"] = ""
                else:
                    v = rows[i]["x_px"] - rows[i - 1]["x_px"]
                    row["velocity_px_per_frame"] = v
                    if i == 1:
                        row["acceleration_px_per_frame2"] = ""
                    else:
                        prev_v = rows[i - 1]["x_px"] - rows[i - 2]["x_px"]
                        row["acceleration_px_per_frame2"] = v - prev_v

        all_rows = sorted(self.records, key=lambda r: (r["frame"], r["sphere_id"]))

        video_dir = os.path.dirname(self.video_path)
        csv_path = os.path.join(video_dir, "sphere_tracking.csv")
        fieldnames = ["frame", "sphere_id", "x_px", "y_px",
                      "velocity_px_per_frame", "acceleration_px_per_frame2"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

        plot_path = os.path.join(video_dir, "sphere_tracking.png")
        self._plot_results(by_sphere, plot_path)
        self.status_var.set(f"Saved: {csv_path}")

    def _plot_results(self, by_sphere, plot_path):
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#fcfcfb")
        ax.set_facecolor("#fcfcfb")

        for i, (sid, rows) in enumerate(sorted(by_sphere.items())):
            rows.sort(key=lambda r: r["frame"])
            xs = [r["frame"] for r in rows]
            ys = [r["x_px"] for r in rows]
            color = _SPHERE_COLORS_HEX[i % len(_SPHERE_COLORS_HEX)]
            ax.plot(xs, ys, color=color, linewidth=1.5, label=f"Sphere {sid}")

        ax.set_xlabel("Frame", color="#4a4a48")
        ax.set_ylabel("x position (px)", color="#4a4a48")
        ax.set_title("Sphere x-position vs Frame", color="#2a2a28", pad=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#b0afa8")
        ax.spines["bottom"].set_color("#b0afa8")
        ax.tick_params(colors="#898781")
        ax.grid(True, color="#e1e0d9", linewidth=0.8)
        ax.legend(framealpha=0.7)

        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.show()


    def _draw_roi(self, cv_img):
        """Draw the current ROI rectangle onto a cv2 image in-place."""
        if self.roi:
            rx, ry, rw, rh = self.roi
            cv2.rectangle(cv_img, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
            cv2.putText(cv_img, "ROI", (rx + 4, ry + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    # ── Manual tracker drawing ────────────────────────────────────────────────

    def toggle_tracker_draw_mode(self):
        # Exit ROI mode if active
        if self._roi_mode:
            self.toggle_roi_mode()
        self._tracker_draw_mode = not self._tracker_draw_mode
        if self._tracker_draw_mode:
            self.draw_tracker_btn.config(relief=tk.SUNKEN, bg="#00e5ff")
            self.canvas.config(cursor="tcross")
            self.status_var.set(
                "Draw Tracker mode: drag a box around the particle. "
                "Repeat for each particle. Click ✚ Draw Tracker again to exit."
            )
        else:
            self.draw_tracker_btn.config(relief=tk.RAISED, bg="SystemButtonFace")
            self.canvas.config(cursor="crosshair")
            if self._tracker_draw_rect_id:
                self.canvas.delete(self._tracker_draw_rect_id)
                self._tracker_draw_rect_id = None

    def clear_trackers(self):
        self.trackers = []
        self.next_id = 1
        self.records = []
        self._tracker_draw_mode = False
        self.draw_tracker_btn.config(relief=tk.RAISED, bg="SystemButtonFace")
        self.canvas.config(cursor="crosshair")
        self.clear_trackers_btn.config(state=tk.DISABLED)
        self.play_btn.config(state=tk.DISABLED)
        self.export_btn.config(state=tk.DISABLED)
        if self.raw_frame is not None:
            self._show_frame(self._annotate_detections(self.raw_frame))
        self.status_var.set("Trackers cleared. Draw new trackers or use Initialize Trackers.")

    # ── ROI selection ─────────────────────────────────────────────────────────

    def toggle_roi_mode(self):
        # Exit tracker-draw mode if active
        if self._tracker_draw_mode:
            self.toggle_tracker_draw_mode()
        self._roi_mode = not self._roi_mode
        if self._roi_mode:
            self.roi_btn.config(relief=tk.SUNKEN, bg="#ffe066")
            self.canvas.config(cursor="tcross")
            self.status_var.set("ROI mode: click and drag on the frame to set the search area.")
        else:
            self.roi_btn.config(relief=tk.RAISED, bg="SystemButtonFace")
            self.canvas.config(cursor="crosshair")
            if self._roi_rect_id:
                self.canvas.delete(self._roi_rect_id)
                self._roi_rect_id = None

    def clear_roi(self):
        self.roi = None
        self.clear_roi_btn.config(state=tk.DISABLED)
        self.roi_btn.config(relief=tk.RAISED, bg="SystemButtonFace")
        self.canvas.config(cursor="crosshair")
        self._roi_mode = False
        if self._roi_rect_id:
            self.canvas.delete(self._roi_rect_id)
            self._roi_rect_id = None
        if self.raw_frame is not None and not self._playing:
            self._show_frame(self._annotate_detections(self.raw_frame))
        self.status_var.set("ROI cleared — detection uses the full frame.")

    def _roi_press(self, event):
        try:
            self.root.focus_set()
        except Exception:
            pass
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        if self._tracker_draw_mode:
            self._roi_start = (cx, cy)
            if self._tracker_draw_rect_id:
                self.canvas.delete(self._tracker_draw_rect_id)
            self._tracker_draw_rect_id = self.canvas.create_rectangle(
                cx, cy, cx, cy, outline="#00ffff", width=2, dash=(4, 4)
            )
        elif self._roi_mode:
            self._roi_start = (cx, cy)
            if self._roi_rect_id:
                self.canvas.delete(self._roi_rect_id)
            self._roi_rect_id = self.canvas.create_rectangle(
                cx, cy, cx, cy, outline="#ffff00", width=2, dash=(4, 4)
            )

    def _roi_drag(self, event):
        if self._roi_start is None:
            return
        x1, y1 = self._roi_start
        x2 = self.canvas.canvasx(event.x)
        y2 = self.canvas.canvasy(event.y)
        if self._tracker_draw_mode and self._tracker_draw_rect_id:
            self.canvas.coords(self._tracker_draw_rect_id, x1, y1, x2, y2)
        elif self._roi_mode and self._roi_rect_id:
            self.canvas.coords(self._roi_rect_id, x1, y1, x2, y2)

    def _roi_release(self, event):
        if self._roi_start is None:
            return
        x1, y1 = self._roi_start
        x2 = self.canvas.canvasx(event.x)
        y2 = self.canvas.canvasy(event.y)
        self._roi_start = None
        s = self.scale if self.scale > 0 else 1.0
        fx1 = int(min(x1, x2) / s)
        fy1 = int(min(y1, y2) / s)
        fx2 = int(max(x1, x2) / s)
        fy2 = int(max(y1, y2) / s)
        rw, rh = fx2 - fx1, fy2 - fy1

        if self._tracker_draw_mode:
            if self._tracker_draw_rect_id:
                self.canvas.delete(self._tracker_draw_rect_id)
                self._tracker_draw_rect_id = None
            if rw < 5 or rh < 5:
                return
            if self.raw_frame is not None:
                fh, fw = self.raw_frame.shape[:2]
                fx1 = max(0, min(fx1, fw - 1))
                fy1 = max(0, min(fy1, fh - 1))
                rw = min(rw, fw - fx1)
                rh = min(rh, fh - fy1)
            bbox = (fx1, fy1, rw, rh)
            cx = fx1 + rw // 2
            cy = fy1 + rh // 2
            sid = self.next_id
            self.next_id += 1
            entry = {"id": sid, "bbox": bbox, "cx": cx, "cy": cy,
                     "lost": False, "tracker": None, "start_frame": self.current_frame_idx}
            if self.raw_frame is not None:
                t = _make_tracker()
                t.init(self.raw_frame, bbox)
                entry["tracker"] = t
            self.trackers.append(entry)
            self.clear_trackers_btn.config(state=tk.NORMAL)
            self.play_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.NORMAL)
            if self.raw_frame is not None:
                self._show_frame(self._annotate_tracked(self.raw_frame))
            n = len(self.trackers)
            self.status_var.set(
                f"Tracker #{sid} added at ({cx}, {cy}). "
                f"{n} tracker(s) total. Draw another or click ▶ Track."
            )

        elif self._roi_mode:
            if self._roi_rect_id:
                self.canvas.delete(self._roi_rect_id)
                self._roi_rect_id = None
            if rw < 5 or rh < 5:
                self.toggle_roi_mode()
                return
            if self.raw_frame is not None:
                fh, fw = self.raw_frame.shape[:2]
                fx1 = max(0, min(fx1, fw - 1))
                fy1 = max(0, min(fy1, fh - 1))
                rw = min(rw, fw - fx1)
                rh = min(rh, fh - fy1)
            self.roi = (fx1, fy1, rw, rh)
            self.clear_roi_btn.config(state=tk.NORMAL)
            self._roi_mode = False
            self.roi_btn.config(relief=tk.RAISED, bg="SystemButtonFace")
            self.canvas.config(cursor="crosshair")
            if self.raw_frame is not None:
                self._show_frame(self._annotate_detections(self.raw_frame))
            self.status_var.set(
                f"ROI set: x={fx1} y={fy1} w={rw} h={rh}. Now click Initialize Trackers."
            )


def _sep(parent):
    tk.Frame(parent, width=2, bd=1, relief=tk.SUNKEN).pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=2)


def main():
    root = tk.Tk()
    SphereTrackerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
