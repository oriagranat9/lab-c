import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import glob


MIN_BLOB_AREA = 100

# Default HSV range for green
DEFAULT_HUE_MIN = 45
DEFAULT_HUE_MAX = 75
DEFAULT_SAT_MIN = 195
DEFAULT_VAL_MIN = 27


class BlobFinderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Green Blob Finder")
        self.root.geometry("900x700")

        self.original_image = None
        self.display_image = None
        self.scale = 1.0
        self.blobs = []
        self.tk_image = None

        self.hue_min = tk.IntVar(value=DEFAULT_HUE_MIN)
        self.hue_max = tk.IntVar(value=DEFAULT_HUE_MAX)
        self.sat_min = tk.IntVar(value=DEFAULT_SAT_MIN)
        self.val_min = tk.IntVar(value=DEFAULT_VAL_MIN)

        self.pick_mode = False
        self.zoom = 1.0
        self.scale = 1.0

        self.folder_images = []   # sorted list of image paths in current folder
        self.folder_index = 0     # index of currently displayed image

        self._build_ui()
        for var in (self.hue_min, self.hue_max, self.sat_min, self.val_min):
            var.trace_add("write", self._on_hsv_change)

    def _build_ui(self):
        toolbar = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(toolbar, text="Open Image", command=self.open_image).pack(side=tk.LEFT, padx=4, pady=2)

        tk.Frame(toolbar, width=2, bd=1, relief=tk.SUNKEN).pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=2)
        self.prev_btn = tk.Button(toolbar, text="◀ Prev", command=self.prev_image, state=tk.DISABLED)
        self.prev_btn.pack(side=tk.LEFT, padx=2, pady=2)
        self.file_label = tk.Label(toolbar, text="—", width=20, anchor=tk.CENTER, relief=tk.FLAT)
        self.file_label.pack(side=tk.LEFT, padx=4)
        self.next_btn = tk.Button(toolbar, text="Next ▶", command=self.next_image, state=tk.DISABLED)
        self.next_btn.pack(side=tk.LEFT, padx=2, pady=2)
        tk.Frame(toolbar, width=2, bd=1, relief=tk.SUNKEN).pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=2)

        tk.Button(toolbar, text="Refresh Detection", command=self.detect_and_draw).pack(side=tk.LEFT, padx=4, pady=2)

        self.pick_btn = tk.Button(toolbar, text="Pick Color", command=self.toggle_pick_mode,
                                  relief=tk.RAISED, bg="SystemButtonFace")
        self.pick_btn.pack(side=tk.LEFT, padx=4, pady=2)

        tk.Frame(toolbar, width=2, bd=1, relief=tk.SUNKEN).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        tk.Button(toolbar, text="−", width=2, command=self.zoom_out).pack(side=tk.LEFT, padx=2, pady=2)
        self.zoom_label = tk.Label(toolbar, text="100%", width=5)
        self.zoom_label.pack(side=tk.LEFT)
        tk.Button(toolbar, text="+", width=2, command=self.zoom_in).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(toolbar, text="Fit", command=self.zoom_fit).pack(side=tk.LEFT, padx=2, pady=2)

        tk.Label(toolbar, text="  Min area:").pack(side=tk.LEFT)
        self.min_area_var = tk.IntVar(value=MIN_BLOB_AREA)
        tk.Spinbox(toolbar, from_=10, to=100000, textvariable=self.min_area_var, width=8).pack(side=tk.LEFT, padx=2)

        # HSV color range controls
        tk.ttk_separator = tk.Frame(toolbar, width=2, bd=1, relief=tk.SUNKEN)
        tk.ttk_separator.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)

        tk.Label(toolbar, text="Hue:").pack(side=tk.LEFT)
        tk.Scale(toolbar, from_=0, to=179, orient=tk.HORIZONTAL, variable=self.hue_min,
                 length=80, label="min", showvalue=True).pack(side=tk.LEFT)
        tk.Scale(toolbar, from_=0, to=179, orient=tk.HORIZONTAL, variable=self.hue_max,
                 length=80, label="max", showvalue=True).pack(side=tk.LEFT)

        tk.Label(toolbar, text=" Sat≥").pack(side=tk.LEFT)
        tk.Scale(toolbar, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.sat_min,
                 length=80, showvalue=True).pack(side=tk.LEFT)

        tk.Label(toolbar, text=" Val≥").pack(side=tk.LEFT)
        tk.Scale(toolbar, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.val_min,
                 length=80, showvalue=True).pack(side=tk.LEFT)

        self.swatch = tk.Label(toolbar, text="  color  ", width=8, relief=tk.SUNKEN)
        self.swatch.pack(side=tk.LEFT, padx=6)
        self._update_swatch()

        self.status_var = tk.StringVar(value="Open an image to start.")
        tk.Label(self.root, textvariable=self.status_var, anchor=tk.W, relief=tk.SUNKEN, bd=1).pack(
            side=tk.BOTTOM, fill=tk.X
        )

        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="#2b2b2b", cursor="crosshair")
        vscroll = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        hscroll = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vscroll.set, xscrollcommand=hscroll.set)

        vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        hscroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", self.on_mousewheel)
        self.canvas.bind("<Button-5>", self.on_mousewheel)
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())

    def _on_hsv_change(self, *_):
        self._update_swatch()
        self.detect_and_draw()

    def _update_swatch(self):
        h = (self.hue_min.get() + self.hue_max.get()) // 2
        s = max(self.sat_min.get(), 128)
        v = max(self.val_min.get(), 128)
        hsv_pixel = np.uint8([[[h, s, v]]])
        bgr = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
        hex_color = "#{:02x}{:02x}{:02x}".format(int(bgr[2]), int(bgr[1]), int(bgr[0]))
        self.swatch.config(bg=hex_color)

    def zoom_in(self):
        self.zoom = min(self.zoom * 1.25, 16.0)
        self._redraw_at_zoom()

    def zoom_out(self):
        self.zoom = max(self.zoom / 1.25, 0.1)
        self._redraw_at_zoom()

    def zoom_fit(self):
        self.zoom = 1.0
        self._redraw_at_zoom()

    def on_mousewheel(self, event):
        if self.original_image is None:
            return
        # Determine scroll direction
        if event.num == 4 or event.delta > 0:
            factor = 1.1
        else:
            factor = 1 / 1.1

        # Keep the point under the cursor stationary
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)

        self.zoom = max(0.1, min(self.zoom * factor, 16.0))
        self._redraw_at_zoom(anchor_canvas=(cx, cy), anchor_img=(cx / self.scale, cy / self.scale))

    def _redraw_at_zoom(self, anchor_canvas=None, anchor_img=None):
        if self.original_image is None:
            return
        self.detect_and_draw()
        if anchor_canvas and anchor_img:
            # Scroll so the image point stays under the cursor
            new_x = anchor_img[0] * self.scale - anchor_canvas[0]
            new_y = anchor_img[1] * self.scale - anchor_canvas[1]
            img_h, img_w = self.original_image.shape[:2]
            total_w = img_w * self.scale
            total_h = img_h * self.scale
            self.canvas.xview_moveto(max(0, new_x) / total_w)
            self.canvas.yview_moveto(max(0, new_y) / total_h)

    def toggle_pick_mode(self):
        self.pick_mode = not self.pick_mode
        if self.pick_mode:
            self.pick_btn.config(relief=tk.SUNKEN, bg="#ffe066")
            self.canvas.config(cursor="tcross")
            self.status_var.set("Pick Color mode: click anywhere on the image to sample that color.")
        else:
            self.pick_btn.config(relief=tk.RAISED, bg="SystemButtonFace")
            self.canvas.config(cursor="crosshair")
            self.status_var.set("Pick Color mode off.")

    def on_mouse_move(self, event):
        if not self.pick_mode or self.original_image is None:
            return
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        orig_x = int(cx / self.scale)
        orig_y = int(cy / self.scale)
        h, w = self.original_image.shape[:2]
        if 0 <= orig_x < w and 0 <= orig_y < h:
            bgr = self.original_image[orig_y, orig_x]
            hex_color = "#{:02x}{:02x}{:02x}".format(int(bgr[2]), int(bgr[1]), int(bgr[0]))
            self.swatch.config(bg=hex_color)

    def _sample_color(self, event):
        if self.original_image is None:
            return
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        orig_x = int(cx / self.scale)
        orig_y = int(cy / self.scale)
        h_img, w_img = self.original_image.shape[:2]
        if not (0 <= orig_x < w_img and 0 <= orig_y < h_img):
            return

        bgr = self.original_image[orig_y, orig_x]
        hsv_pixel = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        hue, sat, val = int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2])

        hue_tol = 15
        new_hmin = max(0, hue - hue_tol)
        new_hmax = min(179, hue + hue_tol)
        new_smin = max(0, sat - 60)
        new_vmin = max(0, val - 60)

        # Pause traces so detect_and_draw fires only once
        for var in (self.hue_min, self.hue_max, self.sat_min, self.val_min):
            var.trace_remove("write", var.trace_info()[0][1])

        self.hue_min.set(new_hmin)
        self.hue_max.set(new_hmax)
        self.sat_min.set(new_smin)
        self.val_min.set(new_vmin)

        for var in (self.hue_min, self.hue_max, self.sat_min, self.val_min):
            var.trace_add("write", self._on_hsv_change)

        self._update_swatch()
        self.detect_and_draw()

        hex_color = "#{:02x}{:02x}{:02x}".format(int(bgr[2]), int(bgr[1]), int(bgr[0]))
        self.status_var.set(
            f"Sampled pixel ({orig_x}, {orig_y})  |  RGB: ({int(bgr[2])}, {int(bgr[1])}, {int(bgr[0])})  |  "
            f"HSV: ({hue}, {sat}, {val})  →  hue range [{new_hmin}–{new_hmax}]"
        )

        # Exit pick mode after sampling
        self.toggle_pick_mode()

    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("All files", "*.*")]
        )
        if not path:
            return
        path = os.path.normpath(path)
        folder = os.path.dirname(path)
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp")
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(folder, ext)))
            files.extend(glob.glob(os.path.join(folder, ext.upper())))
        self.folder_images = sorted(set(os.path.normpath(f) for f in files))
        self.folder_index = self.folder_images.index(path) if path in self.folder_images else 0
        self._load_image_at(self.folder_index)

    def _load_image_at(self, index):
        path = self.folder_images[index]
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", f"Could not read image:\n{path}")
            return
        self.original_image = img
        self.folder_index = index
        self.zoom = 1.0

        name = os.path.basename(path)
        total = len(self.folder_images)
        self.file_label.config(text=f"{index + 1}/{total}  {name}")
        self.root.title(f"Green Blob Finder — {name}")
        self.prev_btn.config(state=tk.NORMAL if index > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if index < total - 1 else tk.DISABLED)
        self.detect_and_draw()

    def prev_image(self):
        if self.folder_index > 0:
            self._load_image_at(self.folder_index - 1)

    def next_image(self):
        if self.folder_index < len(self.folder_images) - 1:
            self._load_image_at(self.folder_index + 1)

    def detect_and_draw(self):
        if self.original_image is None:
            return

        lower = np.array([self.hue_min.get(), self.sat_min.get(), self.val_min.get()])
        upper = np.array([self.hue_max.get(), 255, 255])

        img = self.original_image
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = self.min_area_var.get()
        self.blobs = [c for c in contours if cv2.contourArea(c) >= min_area]

        display = img.copy()
        cv2.drawContours(display, self.blobs, -1, (0, 0, 255), 2)

        for i, c in enumerate(self.blobs):
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(display, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(display, str(i + 1), (cx + 6, cy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        self._show_image(display)
        self.status_var.set(f"Found {len(self.blobs)} green blob(s). Click a blob to see its coordinates.")

    def _show_image(self, cv_img):
        canvas_w = self.canvas.winfo_width() or 860
        canvas_h = self.canvas.winfo_height() or 600

        h, w = cv_img.shape[:2]
        fit_scale = min(canvas_w / w, canvas_h / h, 1.0)
        self.scale = fit_scale * self.zoom

        new_w = max(1, int(w * self.scale))
        new_h = max(1, int(h * self.scale))

        interp = cv2.INTER_AREA if self.scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(cv_img, (new_w, new_h), interpolation=interp)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self.tk_image = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.config(scrollregion=(0, 0, new_w, new_h))
        self.zoom_label.config(text=f"{int(self.scale * 100)}%")

    def on_click(self, event):
        if self.pick_mode:
            self._sample_color(event)
            return

        if not self.blobs:
            return

        # Convert canvas click to original image coordinates
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)

        orig_x = cx / self.scale
        orig_y = cy / self.scale

        clicked_blob = None
        for i, contour in enumerate(self.blobs):
            dist = cv2.pointPolygonTest(contour, (orig_x, orig_y), False)
            if dist >= 0:
                clicked_blob = (i, contour)
                break

        if clicked_blob is None:
            # Find the nearest blob centroid within a tolerance
            best_dist = float("inf")
            best_i = None
            tolerance = 20 / self.scale  # 20 display pixels
            for i, contour in enumerate(self.blobs):
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    bx = M["m10"] / M["m00"]
                    by = M["m01"] / M["m00"]
                    d = ((orig_x - bx) ** 2 + (orig_y - by) ** 2) ** 0.5
                    if d < best_dist and d < tolerance:
                        best_dist = d
                        best_i = i
            if best_i is not None:
                clicked_blob = (best_i, self.blobs[best_i])

        if clicked_blob is None:
            self.status_var.set("No blob at that location. Click directly on a red-outlined blob.")
            return

        idx, contour = clicked_blob
        M = cv2.moments(contour)
        if M["m00"] != 0:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
        else:
            centroid_x, centroid_y = 0, 0

        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        img_height = self.original_image.shape[0]
        centroid_y_from_bottom = img_height - centroid_y

        self.status_var.set(
            f"Blob #{idx + 1}  |  Centroid: ({centroid_x}, {centroid_y})  |  "
            f"From bottom: {centroid_y_from_bottom}px  |  "
            f"Bounding box: x={x}, y={y}, w={w}, h={h}  |  Area: {area:.0f} px²"
        )


def main():
    root = tk.Tk()
    app = BlobFinderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
