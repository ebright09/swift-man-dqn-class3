"""A live window for evaluation demos, running in its own process so training never waits on it.

If a window can't open (no Tk, no display, or --no-live-window), show() returns False and
the caller just keeps saving GIFs.
"""
import multiprocessing as mp
import os
import queue
import sys


def _viewer(frames, title, frame_ms):
    import tkinter as tk
    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title(title)
    root.configure(bg="#101522")
    root.attributes("-topmost", True)
    label = tk.Label(root, bg="#101522")
    label.pack()
    caption = tk.Label(root, text="live evaluation demo · parody, not affiliated", bg="#101522", fg="#b9c4d6")
    caption.pack(pady=4)
    state = {"image": None}

    def tick():
        try:
            item = frames.get_nowait()
        except queue.Empty:
            item = False
        if item is None:        # end of demo: keep the last frame up briefly, then close
            root.after(1500, root.destroy)
            return
        if item:
            size, data, name = item
            state["image"] = ImageTk.PhotoImage(Image.frombytes("RGB", size, data), master=root)
            label.configure(image=state["image"])
            root.title(f"{title} · {name}")
        root.after(frame_ms if item else 15, tick)

    root.bind("<Escape>", lambda event: root.destroy())
    tick()
    root.mainloop()


class LiveWindow:
    def __init__(self, enabled, frame_ms):
        self.enabled, self.frame_ms, self.process, self.frames = enabled, frame_ms, None, None

    def open(self, title):
        if not self.enabled:
            return False
        try:
            import tkinter  # noqa: F401
        except ImportError:
            print("Live window unavailable (no Tk). Saving the GIF only.")
            return False
        if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            print("Live window unavailable (no display). Saving the GIF only.")
            return False
        self.close()
        context = mp.get_context("spawn")
        self.frames = context.Queue(maxsize=8)
        self.process = context.Process(target=_viewer, args=(self.frames, title, self.frame_ms), daemon=True)
        self.process.start()
        return True

    def show(self, image, name=""):
        """Send one PIL frame. Waits briefly so the demo plays at GIF speed; drops frames if the window died."""
        if self.process is None or not self.process.is_alive():
            return
        try:
            self.frames.put((image.size, image.tobytes(), name), timeout=self.frame_ms / 1000 * 4)
        except queue.Full:
            pass

    def finish(self):
        if self.process is not None and self.process.is_alive():
            try:
                self.frames.put(None, timeout=1)
            except queue.Full:
                pass

    def close(self):
        if self.process is not None and self.process.is_alive():
            self.process.terminate()
        self.process = None
