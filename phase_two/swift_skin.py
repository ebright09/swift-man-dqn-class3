"""SWIFT-MAN (Taylor's Version) skin, copied verbatim from pacman_dqn.ipynb section 3½.

Cosmetic only: it repaints recorded RGB frames and is never shown to the network.
Do not edit the code below by hand. Edit the notebook cells tagged "swift-sprites" and
"swift-stage", then re-copy them; tests/test_phase_two.py fails if the two drift apart.
"""
import math
import random
from collections import deque

import cv2
import numpy as np
from PIL import Image as PILImage

# ---- BEGIN NOTEBOOK COPY ----
# Taylor's Version, part 1: the cast, in genuine 8-bit pixel art.
# Each sprite is a little text map, one letter per pixel, like a 1985 plumber but with better bangs.
# Parody. Not affiliated with, endorsed by, or known to Taylor Swift or her lawyers.
from PIL import ImageDraw, ImageFont, ImageOps

SKIN_SCALE = 2            # GIFs are 320 × 420: twice the Atari screen, same as the popup.
PLAY_ROWS = 172           # Rows 0–171 are the maze; below that is the Atari score bar.
PAC_RGB = (210, 164, 74)  # Ms. Pac-Man's yellow. She has been replaced. She was not consulted.
GHOST_NAMES = {           # Atari ghost colors → who they really are.
    (200, 72, 72): "HATERS", (198, 89, 179): "PAPARAZZI",
    (84, 184, 153): "TICKET BOTS", (180, 122, 48): "CRITICS",
}
PALETTE = {  # One NES-ish palette for the whole cast. "." is transparent.
    "H": (248, 208, 96), "h": (200, 144, 40),     # blonde hair, hair shadow
    "S": (252, 212, 164), "K": (24, 16, 32),       # skin, outline / eyeliner
    "E": (40, 96, 232), "R": (232, 0, 56),         # blue eyes, THE red lip
    "P": (168, 40, 224), "p": (255, 150, 255),     # sequined bodysuit, sequins
    "B": (240, 24, 104), "W": (255, 255, 255),     # sparkly boots, sparkle
    "M": (96, 96, 112), "Y": (255, 214, 0),        # microphone, gold
    "t": (204, 150, 72), "D": (0, 160, 64),        # money-bag burlap, dollar green
    "L": (110, 200, 255),                          # camera lens
    "1": (255, 105, 180), "2": (150, 90, 255), "3": (0, 200, 200),  # Swiftie merch colors
    "4": (120, 70, 30), "5": (30, 20, 20), "6": (141, 85, 36),       # Swiftie hair / skin
}


def skin_font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has one tiny font and no opinions.
        return ImageFont.load_default()


def pixel_sprite(rows, px=SKIN_SCALE, swap=None):
    """Turn a text map into a crisp RGBA sprite, every art pixel drawn as a px × px block."""
    swap = swap or {}
    art = np.zeros((len(rows), len(rows[0]), 4), np.uint8)
    for y, row in enumerate(rows):
        for x, key in enumerate(row):
            if key != ".":
                art[y, x] = PALETTE[swap.get(key, key)] + (255,)
    return PILImage.fromarray(art.repeat(px, 0).repeat(px, 1), "RGBA")


# Side view, facing right, 14 × 18 art pixels: bangs, cat-eye liner, red lip, mic, sequins, boots.
TAYLOR_TOP = [
    "....KKKKKK....",
    "...KHHHHHHKK..",
    "..KHHHHHHHHHK.",
    "..KHHHHHHHHHHK",
    "..KHHhHHHHHHK.",
    ".KHHhSSKKSSSK.",
    ".KHHhSSSESSSSK",
    ".KHHhhSSSSSSK.",
]
TAYLOR_MOUTH = {False: ".KHHhhSSSRRK..", True: ".KHHhhSSSRKR.."}  # chomp, Pac-Man-style
TAYLOR_BODY = [
    ".KHHHhhSSSK.MM",
    ".KHHHhKSSK.SM.",
    "..KHKPPPPPPSK.",
    "..KKPpPPPpPK..",
    "...KPPPpPPPK..",
    "...KPpPPPpPK..",
]
TAYLOR_LEGS = {  # two-frame walk cycle
    0: ["....KSSKSSK...", "....KBBKBBK...", "...KBBBKBBBK.."],
    1: ["...KSSK.KSSK..", "..KBBK...KBBK.", ".KBBBK...KBBBK"],
}
SWIFTIE = [  # a tiny fan, arm up, friendship bracelet (Y) on the wrist
    "..hhh.S",
    ".hSSSh Y".replace(" ", ""),
    ".hSSSh.",
    "..SSS.S",
    ".TTTTTT",
    ".TTRTT.",
    "..TTTT.",
    "..S..S.",
]
MONEY_MAP = [
    "..KKK..",
    "..tYt..",
    ".KtDtK.",
    "KtDDDtK",
    "KtDtttK",
    "KttDttK",
    "KtttDtK",
    "KtDDDtK",
    ".KKKKK.",
]
SHADES_MAP = ["KKKKKKKK", "KWK..KWK", ".K....K."]
CAMERA_MAP = [".KK...", "KKKKKK", "KKLLKK", "KKLLKK", "KKKKKK"]


def _taylor(walk, mouth):
    """Taylor plus a glowing halo, so she stays visible inside the fever dream. Main-character lighting."""
    body = pixel_sprite(TAYLOR_TOP + [TAYLOR_MOUTH[mouth]] + TAYLOR_BODY + TAYLOR_LEGS[walk])
    alpha = np.pad(np.asarray(body.getchannel("A")), 3)
    glow = cv2.dilate(alpha, np.ones((5, 5), np.uint8))
    halo = np.zeros(glow.shape + (4,), np.uint8)
    halo[glow > 0] = (255, 255, 200, 220)
    framed = PILImage.fromarray(halo, "RGBA")
    framed.alpha_composite(body, (3, 3))
    return framed


def _tinted(sprite, rgb, alpha):
    """Recolor a sprite and fade it: one ghostly afterimage of Taylor for the trail."""
    solid = PILImage.new("RGBA", sprite.size, rgb + (0,))
    solid.putalpha(sprite.getchannel("A").point(lambda a: int(a * alpha)))
    return solid


# Frame f alternates walk pose and chomp together, so she walks AND eats. Multitasking queen.
TAYLOR = {(facing, f): (lambda im: ImageOps.mirror(im) if facing < 0 else im)(_taylor(f, bool(f)))
          for facing in (1, -1) for f in (0, 1)}
TRAIL_COLORS = [(255, 0, 200), (0, 240, 255), (255, 240, 0), (120, 255, 60), (255, 90, 0), (150, 80, 255)]
TRAILS = {facing: [_tinted(TAYLOR[(facing, 0)], c, 0.55 - 0.08 * i) for i, c in enumerate(TRAIL_COLORS)]
          for facing in (1, -1)}
SWIFTIES = [pixel_sprite(SWIFTIE, swap={"h": hair, "S": skin, "T": shirt})
            for hair, skin, shirt in [("4", "S", "1"), ("5", "6", "2"), ("H", "S", "3"),
                                      ("1", "S", "Y"), ("5", "S", "D"), ("4", "6", "R")]]
MONEY_BAG, BIG_MONEY_BAG = pixel_sprite(MONEY_MAP), pixel_sprite(MONEY_MAP, px=3)
SHADES, CAMERA = pixel_sprite(SHADES_MAP), pixel_sprite(CAMERA_MAP)
print("Cast assembled:", len(SWIFTIES), "Swifties, 2 sizes of money bag, 1 8-bit Taylor,",
      "4 ghosts (now haters, paparazzi, ticket bots, and critics).")


# Taylor's Version, part 2: the stage. Called once per recorded GIF frame, never during learning.
CAPTIONS = [
    "Q-VALUES (TAYLOR'S VERSION)",
    "WELCOME TO THE EPSILON-GREEDY ERA",
    "RANDOM MOVES ARE NOW 'ARTISTIC CHOICES'",
    "GAMMA = 0.99: SHE PLANS 12 ALBUMS AHEAD",
    "REPLAY BUFFER NEVER FORGETS. NOR DO FANS.",
    "TARGET NETWORK SYNCED. EASTER EGG FOUND.",
    "REWARDS CLIPPED TO [-1, 1]. PUBLICIST SOBS.",
    "THE NETWORK SEES GRAYSCALE. LUCKY NETWORK.",
    "LOSS WENT DOWN. SCORE? NO COMMENT.",
    "HUBER LOSS: FOR MISTAKES THAT GET LOUD",
    "GHOSTS HAVE BEEN SERVED. LEGALLY.",
    "4 FRAMES STACKED. 13 WOULD BE LUCKIER.",
    "BELLMAN WOULD BE PROUD. OR CONCERNED.",
    "NO SWIFTIES HARMED. THEY WERE RECRUITED.",
    "ATARI 2600, BUT MAKE IT A STADIUM TOUR",
    "RE-RECORDING THIS MAZE FROM MEMORY",
    "NOT FINANCIAL ADVICE. BARELY A GAME.",
]
TICKER = "   ///   ".join(CAPTIONS) + "   ///   "


class SwiftSkin:
    """Repaints one Ms. Pac-Man game as a psychedelic 8-bit Taylor Swift fever dream.

    Make a new SwiftSkin for every recorded game: it remembers Taylor's trail,
    who got recruited, and how many bags were secured.
    """

    def __init__(self, seed=0):
        h, w = PLAY_ROWS * SKIN_SCALE, 160 * SKIN_SCALE
        self.yy, self.xx = np.mgrid[0:h, 0:w].astype(np.float32)
        self.radius = np.hypot(self.xx - w / 2, self.yy - h / 2)
        self.rng = random.Random(seed)
        self.t, self.last_score = 0, 0.0
        self.trail, self.facing, self.last_pac = deque(maxlen=len(TRAILS[1])), 1, None
        self.last_pellets, self.swifties, self.bags = set(), 0, 0
        self.floaters, self.shout, self.shake, self.strobe = [], None, 0, 0
        self.font, self.small, self.big = skin_font(11), skin_font(9), skin_font(22)

    # ---- reading the Atari screen -------------------------------------------------
    def _find(self, play):
        """Locate maze walls, pellets, power pellets, Ms. Pac-Man, and ghosts by color."""
        colors, counts = np.unique(play[::2, ::2].reshape(-1, 3), axis=0, return_counts=True)
        order = [tuple(int(v) for v in colors[i]) for i in np.argsort(-counts)]
        order = [c for c in order if c != (0, 0, 0)]
        background, wall = order[0], order[1]
        wall_mask = np.all(play == wall, axis=2).astype(np.uint8)
        n, labels, stats, centers = cv2.connectedComponentsWithStats(wall_mask, connectivity=8)
        pellets, powers, dots = [], [], []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area <= 10 and w <= 4 and h <= 3:
                pellets.append((int(centers[i][0]), int(centers[i][1]))); dots.append(i)
            elif w <= 5 and 5 <= h <= 8 and area <= 36:
                powers.append((int(centers[i][0]), int(centers[i][1]))); dots.append(i)
        maze = wall_mask.astype(bool) & ~np.isin(labels, dots)
        pac = np.all(play == PAC_RGB, axis=2)
        ghosts = []
        other = ~np.all(play == background, axis=2) & ~wall_mask.astype(bool) & ~pac & play.any(axis=2)
        n, labels, stats, centers = cv2.connectedComponentsWithStats(other.astype(np.uint8), connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area >= 25 and 6 <= w <= 9 and 7 <= h <= 11:
                color = tuple(int(v) for v in play[labels == i][0])
                name = GHOST_NAMES.get(color, "SHOOK" if color[2] > max(color[:2]) else None)
                if name:
                    ghosts.append((x, y, w, h, name))
        pac_box = None
        if pac.any():
            ys, xs = np.nonzero(pac)
            pac_box = (int(xs.mean()), int(ys.mean()))
        return maze, pellets, powers, pac_box, ghosts, other

    # ---- painting -------------------------------------------------------------------
    def _plasma(self, maze):
        """The psychedelic part: a drifting rainbow plasma under neon walls."""
        t = self.t * (5 if self.strobe else 1)
        wave = (np.sin(self.xx * 0.045 + t * 0.35) + np.sin(self.yy * 0.06 - t * 0.27)
                + np.sin((self.xx + self.yy) * 0.03 + t * 0.2) + np.sin(self.radius * 0.05 - t * 0.4))
        hue = ((wave + 4) * 22.5 + t * 6) % 180
        big_maze = maze.repeat(SKIN_SCALE, 0).repeat(SKIN_SCALE, 1)
        hsv = np.empty(hue.shape + (3,), np.uint8)
        hsv[..., 0] = np.where(big_maze, (hue + 90) % 180, hue)
        hsv[..., 1] = np.where(big_maze, 170, 255)
        hsv[..., 2] = np.where(big_maze, 255, 70)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        outline = cv2.dilate(big_maze.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~big_maze
        rgb[outline] = (10, 0, 25)
        return rgb

    def _stamp(self, canvas, sprite, cx, cy):
        canvas.alpha_composite(sprite, (int(cx - sprite.width / 2), int(cy - sprite.height / 2)))

    def _hud(self, canvas, score):
        d = ImageDraw.Draw(canvas)
        top = PLAY_ROWS * SKIN_SCALE
        d.rectangle((0, top, canvas.width, canvas.height), fill=(12, 0, 24))
        worth = 1_600_000_000 + int(score) * 1_000_000
        d.text((6, top + 3), f"SCORE {int(score):,}", fill=(255, 230, 90), font=self.font)
        d.text((canvas.width - 6, top + 3), f"NET WORTH ${worth:,}", fill=(90, 255, 140),
               font=self.font, anchor="ra")
        d.text((6, top + 19), f"SWIFTIES RECRUITED {self.swifties}   BAGS SECURED {self.bags}",
               fill=(255, 140, 230), font=self.font)
        caption = CAPTIONS[(self.t // 12 + self.rng_offset) % len(CAPTIONS)]
        d.text((canvas.width / 2, top + 42), caption, fill=self._rainbow(0), font=self.font, anchor="ma")
        shift = (self.t * 9) % (len(TICKER) * 5)
        d.text((6 - shift, top + 60), TICKER * 2, fill=(150, 150, 190), font=self.small)

    def _rainbow(self, offset):
        hsv = np.uint8([[[(self.t * 13 + offset) % 180, 200, 255]]])
        return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0])

    def _glitch(self, frame):
        """The unhinged part: chromatic aberration, torn scanlines, strobes, and shaking."""
        s = 1 + 3 * (self.t % 7 == 0) + 2 * bool(self.strobe)
        frame[..., 0] = np.roll(frame[..., 0], s, axis=1)
        frame[..., 2] = np.roll(frame[..., 2], -s, axis=1)
        if self.t % 9 == 4 or self.shake:
            for _ in range(3):
                y = self.rng.randrange(0, frame.shape[0] - 24)
                h = self.rng.randrange(4, 24)
                frame[y:y + h] = np.roll(frame[y:y + h], self.rng.randrange(-20, 21), axis=1)
        if self.t % 29 == 13:
            frame[:PLAY_ROWS * SKIN_SCALE] = 255 - frame[:PLAY_ROWS * SKIN_SCALE]
        if self.shake:
            frame = np.roll(frame, (self.rng.randrange(-5, 6), self.rng.randrange(-5, 6)), axis=(0, 1))
            self.shake -= 1
        frame[1::2] = (frame[1::2] * 0.8).astype(np.uint8)  # CRT scanlines, for authenticity
        return frame

    def render(self, rgb, score):
        """Atari RGB frame (210 × 160) + score so far → one Taylor's Version GIF frame."""
        if self.t == 0:
            self.rng_offset = self.rng.randrange(len(CAPTIONS))
        maze, pellets, powers, pac, ghosts, other = self._find(rgb[:PLAY_ROWS])
        gained = score - self.last_score
        if gained >= 200:
            self.shout, self.shake = ("PAPARAZZO DEFEATED", 8), 6
        elif gained >= 50 and not self.strobe and len(powers) < getattr(self, "last_powers", 4):
            self.shout, self.strobe = ("BAG SECURED. MARKETS REACT.", 8), 8
        self.last_powers = len(powers)

        stage = np.zeros((210 * SKIN_SCALE, 160 * SKIN_SCALE, 3), np.uint8)
        stage[:PLAY_ROWS * SKIN_SCALE] = self._plasma(maze)
        keep = other.repeat(SKIN_SCALE, 0).repeat(SKIN_SCALE, 1)  # ghosts and fruit keep their pixels
        stage[:PLAY_ROWS * SKIN_SCALE][keep] = rgb[:PLAY_ROWS].repeat(SKIN_SCALE, 0).repeat(SKIN_SCALE, 1)[keep]
        canvas = PILImage.fromarray(stage).convert("RGBA")

        current = set(pellets)
        for x, y in pellets:  # checkerboard: half Swifties, half money
            kind = (x // 8 + y // 12) % 2
            sprite = MONEY_BAG if kind else SWIFTIES[(x * 7 + y * 3) % len(SWIFTIES)]
            self._stamp(canvas, sprite, x * SKIN_SCALE, y * SKIN_SCALE + (self.t + x) % 2)
        if pac and self.last_pellets:
            for x, y in self.last_pellets - current:
                if abs(x - pac[0]) + abs(y - pac[1]) <= 24:
                    money = (x // 8 + y // 12) % 2
                    self.bags += money
                    self.swifties += 1 - money
                    self.floaters.append(["+$1M" if money else "+1 SWIFTIE",
                                          pac[0] * SKIN_SCALE, pac[1] * SKIN_SCALE - 14, 5])
        self.last_pellets = current if pac else self.last_pellets

        d = ImageDraw.Draw(canvas)
        for x, y in powers:
            cx, cy = x * SKIN_SCALE, y * SKIN_SCALE
            spin = self.t * 0.5
            for k in range(8):
                a = spin + k * math.pi / 4
                d.line((cx, cy, cx + 20 * math.cos(a), cy + 20 * math.sin(a)), fill=self._rainbow(k * 20), width=2)
            self._stamp(canvas, BIG_MONEY_BAG, cx, cy)

        for x, y, w, h, name in ghosts:
            cx, top = (x + w / 2) * SKIN_SCALE, y * SKIN_SCALE
            self._stamp(canvas, SHADES, cx, top + 7)
            self._stamp(canvas, CAMERA, cx + 12, top + 14)
            if (self.t + x) % 5 == 0:  # flash photography, always
                d.ellipse((cx + 4, top - 2, cx + 26, top + 20), fill=(255, 255, 255, 170))
            d.text((cx, top - 3), name, fill=(255, 255, 255), font=self.small, anchor="md",
                   stroke_width=2, stroke_fill=(0, 0, 0))

        if pac:
            if self.last_pac and pac[0] != self.last_pac[0]:
                self.facing = 1 if pac[0] > self.last_pac[0] else -1
            for i, (tx, ty, facing) in enumerate(reversed(self.trail)):
                self._stamp(canvas, TRAILS[facing][i], tx, ty)
            cx, cy = pac[0] * SKIN_SCALE, pac[1] * SKIN_SCALE - 4
            self._stamp(canvas, TAYLOR[(self.facing, self.t % 2)], cx, cy)
            self.trail.append((cx, cy, self.facing))
            self.last_pac = pac

        for f in self.floaters:
            d.text((f[1], f[2]), f[0], fill=self._rainbow(f[2]), font=self.small, anchor="ms",
                   stroke_width=2, stroke_fill=(0, 0, 0))
            f[2] -= 4; f[3] -= 1
        self.floaters = [f for f in self.floaters if f[3] > 0]

        if self.t < 8:
            d.text((canvas.width / 2, 120), "SWIFT-MAN", fill=self._rainbow(0), font=self.big,
                   anchor="mm", stroke_width=3, stroke_fill=(0, 0, 0))
            d.text((canvas.width / 2, 146), "(TAYLOR'S VERSION)", fill=self._rainbow(60), font=self.font,
                   anchor="mm", stroke_width=2, stroke_fill=(0, 0, 0))
        if self.shout:
            text, ttl = self.shout
            d.text((canvas.width / 2, 170), text, fill=self._rainbow(90), font=self.font,
                   anchor="mm", stroke_width=3, stroke_fill=(0, 0, 0))
            self.shout = (text, ttl - 1) if ttl > 1 else None

        self._hud(canvas, score)
        frame = self._glitch(np.asarray(canvas.convert("RGB")).copy())
        self.strobe = max(0, self.strobe - 1)
        self.last_score = score
        self.t += 1
        return PILImage.fromarray(frame)
# ---- END NOTEBOOK COPY ----
