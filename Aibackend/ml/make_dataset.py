#!/usr/bin/env python3
"""
Synthetic ID-card dataset generator for the document tamper model.

It draws made-up ID cards (fictional country, random names, no real person),
then, for the "tampered" classes, edits ONE field the way a forger would:
erases the old text and writes new text with a slightly different font / size /
spacing / baseline / colour, with an optional blur and JPEG recompression of the
edited patch. Every card is then passed through a "camera" step (rotation,
lighting, noise, blur, JPEG) so genuine and tampered images look like photos.

Classes (folder names):
    genuine     untouched card
    name        the NAME field was edited
    dob         the DATE OF BIRTH field was edited
    expiry      the EXPIRY DATE field was edited
    id_number   the ID NUMBER field was edited

Usage:
    python make_dataset.py --preview            # writes data/preview.jpg to eyeball samples
    python make_dataset.py                      # 500/100/100 images per class
    python make_dataset.py --per-class 1500 --val-per-class 200 --test-per-class 200
"""

import argparse
import glob
import math
import os
import random
import string
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

CARD_W, CARD_H = 640, 400
CANVAS_W, CANVAS_H = 720, 480
CLASSES = ["genuine", "name", "dob", "expiry", "id_number"]
FIELDS = ["name", "dob", "expiry", "id_number"]
LABELS = {"name": "NAME", "dob": "DATE OF BIRTH", "expiry": "EXPIRY DATE", "id_number": "ID NUMBER"}
FIELD_Y = {"name": 88, "dob": 148, "expiry": 208, "id_number": 268}
FIELD_X = 224

FIRST = ["ARJUN", "ANANYA", "ROHAN", "PRIYA", "KABIR", "MEERA", "VIHAAN", "ISHA", "SAMIR", "NEHA",
         "ADITYA", "KAVYA", "RAHUL", "DIYA", "VIKRAM", "SANYA", "MAYA", "OMAR", "LEILA", "DANIEL",
         "SOFIA", "LUCAS", "EMMA", "NOAH", "AMINA", "YUSUF", "CHEN", "MEI", "HIRO", "SAKURA"]
LAST = ["SHARMA", "VERMA", "PATEL", "SINGH", "GUPTA", "REDDY", "NAIR", "KHAN", "IYER", "MEHTA",
        "JOSHI", "KAPOOR", "BOSE", "DAS", "RAO", "MALIK", "SILVA", "GARCIA", "MULLER", "ROSSI",
        "TANAKA", "WANG", "KIM", "NGUYEN", "BROWN", "SMITH", "COHEN", "HUSSAIN", "ALI", "FERRARI"]

FONT_DIRS = [
    "C:/Windows/Fonts", "/usr/share/fonts", "/usr/local/share/fonts",
    "/System/Library/Fonts", "/Library/Fonts", os.path.expanduser("~/.fonts"),
]


# ----------------------------------------------------------------- fonts
def _render(font, ch):
    im = Image.new("L", (64, 64), 0)
    ImageDraw.Draw(im).text((6, 6), ch, font=font, fill=255)
    return im.tobytes()


def supports_latin(path):
    """False for fonts that would draw empty boxes for A-Z, 0-9 or '/'."""
    try:
        font = ImageFont.truetype(path, 28)
        missing = _render(font, "\u0378")               # an unassigned code point -> the font's "box"
        for ch in "AEKMRSWZ0123456789/":
            glyph = _render(font, ch)
            if glyph == missing or not any(glyph):
                return False
        return True
    except Exception:
        return False


def find_fonts(extra_dir=None):
    """Regular-weight TrueType fonts found on this machine (needs at least 2 for a good dataset)."""
    dirs = FONT_DIRS + ([extra_dir] if extra_dir else [])
    found = []
    for d in dirs:
        for p in glob.glob(os.path.join(d, "**", "*.ttf"), recursive=True):
            name = os.path.basename(p).lower()
            if any(bad in name for bad in ("bold", "italic", "oblique", "light", "thin", "black", "symbol",
                                             "emoji", "dingbat", "mono", "wingding", "math", "icon", "narrow")):
                continue
            found.append(p)
    return sorted(p for p in set(found) if supports_latin(p))


_font_cache = {}


def load_font(path, size):
    key = (path, size)
    if key in _font_cache:
        return _font_cache[key]
    try:
        font = ImageFont.truetype(path, size) if path else ImageFont.load_default(size)
    except TypeError:               # old Pillow: load_default() has no size argument
        font = ImageFont.load_default()
    except OSError:
        font = ImageFont.load_default()
    _font_cache[key] = font
    return font


# ------------------------------------------------------------ value makers
def rand_date(rng, y0, y1):
    return f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(y0, y1)}"


def rand_value(rng, field):
    if field == "name":
        return f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    if field == "dob":
        return rand_date(rng, 1958, 2006)
    if field == "expiry":
        return rand_date(rng, 2026, 2037)
    letters = "".join(rng.choice(string.ascii_uppercase) for _ in range(rng.choice([1, 2])))
    digits = "".join(rng.choice(string.digits) for _ in range(rng.choice([8, 9, 10])))
    return letters + digits


# --------------------------------------------------------------- the card
def make_background(rng):
    base = tuple(rng.randint(222, 246) for _ in range(3))
    img = Image.new("RGB", (CARD_W, CARD_H), base)
    d = ImageDraw.Draw(img)
    tint = tuple(max(0, c - rng.randint(8, 24)) for c in base)
    for _ in range(14):                                 # faint wavy security lines
        amp, freq, phase = rng.uniform(6, 22), rng.uniform(0.008, 0.03), rng.uniform(0, 6.28)
        y0 = rng.randint(50, CARD_H)
        d.line([(x, y0 + amp * math.sin(freq * x + phase)) for x in range(0, CARD_W + 1, 8)], fill=tint, width=1)
    return img


def draw_card(rng, fonts):
    """Returns (card, clean_background, boxes, spec)."""
    bg = make_background(rng)
    img = bg.copy()
    d = ImageDraw.Draw(img)

    band = (rng.randint(20, 70), rng.randint(40, 100), rng.randint(90, 160))
    d.rectangle([0, 0, CARD_W, 64], fill=band)
    title_font = load_font(rng.choice(fonts) if fonts else None, 24)
    d.ellipse([20, 14, 56, 50], outline=(240, 220, 150), width=3)
    d.text((70, 18), "REPUBLIC OF EXAMPLIA", font=title_font, fill=(255, 255, 255))
    d.text((440, 22), "IDENTITY CARD", font=load_font(None, 16), fill=(220, 230, 245))

    skin = (rng.randint(150, 220), rng.randint(110, 180), rng.randint(90, 160))
    photo = Image.new("RGB", (160, 200), (rng.randint(170, 215),) * 3)
    pdraw = ImageDraw.Draw(photo)
    pdraw.ellipse([-10, 120, 170, 300], fill=(rng.randint(60, 130),) * 3)      # shoulders
    pdraw.ellipse([45, 30, 115, 115], fill=skin)                               # head
    img.paste(photo, (30, 92))
    d.rectangle([30, 92, 189, 291], outline=(120, 120, 130), width=1)

    for _ in range(70):                                  # barcode strip
        x = rng.randint(30, 600)
        d.line([(x, 346), (x, 388)], fill=(30, 30, 30), width=rng.choice([1, 1, 2, 3]))

    font_path = rng.choice(fonts) if fonts else None
    size = rng.randint(25, 30)
    color = tuple([rng.randint(18, 45)] * 3)
    label_font = load_font(None, 14)
    spec = {"font": font_path, "size": size, "color": color, "values": {}}
    boxes = {}
    for field in FIELDS:
        y = FIELD_Y[field]
        d.text((FIELD_X, y), LABELS[field], font=label_font, fill=(110, 110, 120))
        value = rand_value(rng, field)
        spec["values"][field] = value
        font = load_font(font_path, size)
        d.text((FIELD_X, y + 22), value, font=font, fill=color)
        boxes[field] = d.textbbox((FIELD_X, y + 22), value, font=font)
    return img, bg, boxes, spec


# -------------------------------------------------------------- tampering
def tamper_field(rng, img, bg, boxes, spec, field, fonts):
    """Edit one field in place, the way a forger would."""
    x0, y0, x1, y1 = boxes[field]
    old = spec["values"][field]
    new = rand_value(rng, field)
    while new == old:
        new = rand_value(rng, field)

    # forged text style: usually a different font, otherwise a different size
    others = [f for f in fonts if f != spec["font"]]
    path = rng.choice(others) if (others and rng.random() < 0.7) else spec["font"]
    size = max(18, spec["size"] + rng.choice([-3, -2, -1, 0, 1, 2, 3]))
    font = load_font(path, size)
    tint = rng.randint(-14, 14)
    color = tuple(max(0, min(255, c + tint)) for c in spec["color"])
    spacing = rng.choice([-1, 0, 0, 1, 2])
    dx, dy = rng.randint(-2, 3), rng.randint(-3, 3)

    def text_width(t):
        return int(sum(font.getlength(ch) + spacing for ch in t)) if spacing else int(font.getlength(t))

    # align the INK of the new text with the old one (fonts have different internal offsets),
    # then make the edit region big enough that nothing is clipped
    nb = font.getbbox(new)
    new_h = nb[3] - nb[1]
    pad = 5
    left = max(0, x0 - pad)
    top = max(0, min(y0, y0 + dy) - pad)
    right = min(CARD_W, max(x1, x0 + dx + text_width(new)) + pad + 4)
    bottom = min(CARD_H, max(y1, y0 + dy + new_h) + pad)
    region = (left, top, right, bottom)

    # 1) erase the old text
    if rng.random() < 0.5:                                # exact restore of the background
        img.paste(bg.crop(region), region[:2])
    else:                                                 # flat patch (leaves a faint rectangle)
        mean = tuple(int(v) for v in np.asarray(bg.crop(region)).reshape(-1, 3).mean(axis=0))
        ImageDraw.Draw(img).rectangle([region[0], region[1], region[2] - 1, region[3] - 1], fill=mean)

    # 2) write the new text on a patch
    patch = img.crop(region).copy()
    pd = ImageDraw.Draw(patch)
    px = x0 - left + dx
    py = y0 - top + dy - nb[1]
    if spacing:
        for ch in new:
            pd.text((px, py), ch, font=font, fill=color)
            px += font.getlength(ch) + spacing
    else:
        pd.text((px, py), new, font=font, fill=color)

    # 3) forger artefacts: optional blur + JPEG recompression of just the patch
    if rng.random() < 0.4:
        patch = patch.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 0.9)))
    if rng.random() < 0.7:
        buf = BytesIO()
        patch.save(buf, "JPEG", quality=rng.randint(50, 85))
        patch = Image.open(buf).convert("RGB")
    img.paste(patch, region[:2])
    return img


# ---------------------------------------------------------- camera effect
def camera_sim(rng, nrng, card):
    W, H = CANVAS_W, CANVAS_H
    base = np.array([rng.randint(40, 200) for _ in range(3)], dtype=np.float32)
    canvas = np.tile(base, (H, W, 1))
    canvas += nrng.normal(0, 7, canvas.shape)
    bg = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))

    tw = rng.randint(int(W * 0.78), int(W * 0.95))
    th = int(tw * CARD_H / CARD_W)
    c = card.resize((tw, th), Image.LANCZOS).convert("RGBA")
    c = c.rotate(rng.uniform(-5, 5), expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))
    ox = rng.randint(min(0, W - c.width), max(0, W - c.width))
    oy = rng.randint(min(0, H - c.height), max(0, H - c.height))
    bg.paste(c, (ox, oy), c)

    img = bg
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.75, 1.2))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.2))
    arr = np.asarray(img).astype(np.float32)
    light = np.linspace(rng.uniform(0.82, 1.08), rng.uniform(0.82, 1.08), W)[None, :, None]
    arr = arr * light + nrng.normal(0, rng.uniform(1.5, 6), arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    return img


def make_sample(rng, nrng, fonts, cls):
    card, bg, boxes, spec = draw_card(rng, fonts)
    if cls != "genuine":
        tamper_field(rng, card, bg, boxes, spec, cls, fonts)
    return camera_sim(rng, nrng, card)


# ------------------------------------------------------------------ main
def build_split(name, per_class, root, fonts, seed):
    count = 0
    for ci, cls in enumerate(CLASSES):
        folder = root / name / cls
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            s = seed * 1_000_003 + ci * 100_003 + i
            rng = random.Random(s)
            nrng = np.random.default_rng(s)
            img = make_sample(rng, nrng, fonts, cls)
            img.save(folder / f"{i:05d}.jpg", "JPEG", quality=rng.randint(72, 95))
            count += 1
            if count % 200 == 0:
                print(f"  {name}: {count} images", flush=True)
    print(f"{name}: {count} images written to {root / name}")


def make_preview(root, fonts, seed):
    rng0 = random.Random(seed)
    tiles = []
    for i, cls in enumerate(["genuine", "genuine", "dob", "name", "expiry", "id_number", "dob", "name", "expiry", "id_number"]):
        s = seed + 17 * i
        img = make_sample(random.Random(s), np.random.default_rng(s), fonts, cls)
        tiles.append((cls, img.resize((CANVAS_W // 2, CANVAS_H // 2))))
    tw, th = tiles[0][1].size
    sheet = Image.new("RGB", (tw * 2, (th + 26) * 5), (245, 245, 245))
    d = ImageDraw.Draw(sheet)
    for i, (cls, im) in enumerate(tiles):
        x, y = (i % 2) * tw, (i // 2) * (th + 26)
        sheet.paste(im, (x, y + 26))
        d.text((x + 8, y + 5), "GENUINE" if cls == "genuine" else f"TAMPERED: {cls}", fill=(20, 20, 20), font=load_font(None, 16))
    root.mkdir(parents=True, exist_ok=True)
    out = root / "preview.jpg"
    sheet.save(out, quality=90)
    print("Preview written to", out)


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic ID tamper dataset")
    ap.add_argument("--out", default="data")
    ap.add_argument("--per-class", type=int, default=500, help="training images per class")
    ap.add_argument("--val-per-class", type=int, default=100)
    ap.add_argument("--test-per-class", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fonts-dir", default=None, help="extra folder with .ttf fonts")
    ap.add_argument("--preview", action="store_true", help="only write a preview sheet and stop")
    args = ap.parse_args()

    fonts = find_fonts(args.fonts_dir)
    print(f"Found {len(fonts)} usable fonts")
    if len(fonts) < 2:
        print("WARNING: fewer than 2 fonts found, so 'different font' forgeries are impossible. "
              "Pass --fonts-dir with some .ttf files for a better dataset.")
    root = Path(args.out)
    if args.preview:
        make_preview(root, fonts, args.seed)
        return
    build_split("train", args.per_class, root, fonts, args.seed)
    build_split("val", args.val_per_class, root, fonts, args.seed + 1)
    if args.test_per_class > 0:
        build_split("test", args.test_per_class, root, fonts, args.seed + 2)
    print("Done. Next: python train.py")


if __name__ == "__main__":
    main()
