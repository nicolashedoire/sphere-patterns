"""Prepare a fixed T-LESS candidate-recognition panel with bounded ZIP reads."""
import argparse
import io
import json
import math
from pathlib import Path
import subprocess
import zipfile

from PIL import Image, ImageDraw, ImageFont

REVISION = "5fd309a04476a842d93abfb584fba9ee7caecdf1"
BASE = f"https://huggingface.co/datasets/bop-benchmark/tless/resolve/{REVISION}/"
ARCHIVES = {
    "train": ("tless_train_primesense.zip", 2486152818),
    "test": ("tless_test_primesense_bop19.zip", 825276992),
}
GRID = 25
PART = 1
TRAIN_FRAMES = (0, 12)
SCENES = (5, 7, 9, 17, 18, 20)
BANDS = (("high", .85, 1.01, .92), ("partial", .35, .70, .52))


class RemoteZipFile(io.RawIOBase):
    BLOCK = 1024 * 1024

    def __init__(self, url, length):
        self.url = url
        self.length = length
        self.offset = 0
        self.cache = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.offset

    def seek(self, count, whence=0):
        self.offset = (0, self.offset, self.length)[whence] + count
        if self.offset < 0:
            raise OSError("Negative ZIP offset")
        return self.offset

    def read(self, size=-1):
        limit = self.length if size < 0 else min(self.length, self.offset + size)
        data = bytearray()
        while self.offset < limit:
            block = self.offset // self.BLOCK
            lo = block * self.BLOCK
            hi = min(lo + self.BLOCK, self.length) - 1
            if block not in self.cache:
                reply = subprocess.check_output([
                    "curl", "-fLsS", "--retry", "2", "--max-time", "30",
                    "--max-filesize", str(self.BLOCK), "-r", f"{lo}-{hi}",
                    "-w", "\n%{http_code} %{size_download}", self.url,
                ])
                payload, status = reply.rsplit(b"\n", 1)
                if status != f"206 {hi - lo + 1}".encode() or len(payload) != hi - lo + 1:
                    raise OSError(f"Unexpected range response for bytes {lo}-{hi}")
                if len(self.cache) >= 24:
                    self.cache.pop(next(iter(self.cache)))
                self.cache[block] = payload
            data.extend(self.cache[block][self.offset - lo:min(limit, hi + 1) - lo])
            self.offset = min(limit, hi + 1)
        return bytes(data)


def load_archive(split):
    name, length = ARCHIVES[split]
    return zipfile.ZipFile(RemoteZipFile(BASE + name, length))


def image(archive, name):
    return Image.open(io.BytesIO(archive.read(name))).copy()


def frame_inputs(archive, scene, frame):
    rgb = image(archive, f"{scene}/rgb/{frame:06d}.png").convert("RGB")
    depth = image(archive, f"{scene}/depth/{frame:06d}.png")
    camera = json.loads(archive.read(scene + "/scene_camera.json"))[str(frame)]
    return rgb, depth, camera["depth_scale"]


def candidate(archive, scene, frame, index, part, meta, loaded, band):
    rgb, depth, depth_scale = loaded
    x, y, width, height = meta["bbox_obj"]
    side = math.ceil(max(width, height) * 1.25)
    left, top = x + (width - side) // 2, y + (height - side) // 2
    square = (left, top, left + side, top + side)
    visual = rgb.crop(square).convert("L").resize((GRID, GRID), Image.Resampling.BOX)
    distance = depth.crop(square).resize((GRID, GRID), Image.Resampling.NEAREST)
    visible = image(archive, f"{scene}/mask_visib/{frame:06d}_{index:06d}.png").convert("L")
    # The annotated mask is separate from all camera-derived inputs.
    truth = visible.crop(square).resize((GRID, GRID), Image.Resampling.BOX)
    return {
        "scene": int(scene.rsplit("/", 1)[1]), "frame": frame, "index": index, "part": part,
        "band": band, "visibility": round(meta["visib_fract"], 6),
        "box": meta["bbox_obj"], "crop": square,
        "rgb": list(visual.get_flattened_data()),
        "depthMm": [round(x * depth_scale) for x in distance.get_flattened_data()],
        "visibleMask": [int(x >= 128) for x in truth.get_flattened_data()],
    }


def frame_choice(gt, metadata, available, low, high, center):
    candidates = [
        (abs(info["visib_fract"] - center), frame, index)
        for frame in sorted(available)
        for index, (item, info) in enumerate(zip(gt[str(frame)], metadata[str(frame)]))
        if item["obj_id"] == PART and low <= info["visib_fract"] < high and min(info["bbox_obj"][2:]) >= 30
        and distractor(gt[str(frame)], metadata[str(frame)], index) is not None
    ]
    _, frame, index = min(candidates)
    return frame, index


def distractor(gt, metadata, positive):
    _, _, w, h = metadata[positive]["bbox_obj"]
    ranked = []
    for i, (item, info) in enumerate(zip(gt, metadata)):
        if item["obj_id"] == PART or info["visib_fract"] < .5:
            continue
        _, _, dw, dh = info["bbox_obj"]
        if min(dw, dh) < 20:
            continue
        ranked.append((abs(math.log(dw / w)) + abs(math.log(dh / h)), i))
    return min(ranked)[1] if ranked else None


def figure(frames, destination):
    font = ImageFont.load_default(size=22)
    labels = ImageFont.load_default(size=16)
    out = Image.new("RGB", (1040, 828), "#f2f5f4")
    draw = ImageDraw.Draw(out)
    draw.text((34, 23), "Real-image pilot · T-LESS", font=font, fill="#132d3a")
    draw.text((34, 57), "Six held-out scenes · Candidate boxes supplied by the dataset", font=labels, fill="#395360")
    for n, (photo, a, b) in enumerate(frames):
        x, y = 34 + n % 2 * 503, 98 + n // 2 * 346
        width, height = 468, 292
        thumbnail = photo.resize((width, round(photo.height * width / photo.width)), Image.Resampling.LANCZOS)
        # Preserve the full image and use its true aspect ratio when drawing boxes.
        thumbnail.thumbnail((width, height), Image.Resampling.LANCZOS)
        out.paste(thumbnail, (x, y))
        sx, sy = thumbnail.width / photo.width, thumbnail.height / photo.height
        for item, color in [(a, "#15dca2"), (b, "#ffbf6c")]:
            bx, by, bw, bh = item["box"]
            draw.rectangle((x + bx * sx, y + by * sy, x + (bx + bw) * sx, y + (by + bh) * sy),
                           width=3, outline=color)
        line = f"Scene {a['scene']}  ·  Target {a['visibility']:.0%} visible"
        draw.text((x, y + height + 8), line, font=labels, fill="#132d3a")
    draw.text((34, 797), "Mint = target     Amber = similarly sized distractor     T-LESS · CC BY 4.0",
              font=labels, fill="#395360")
    out.save(destination)


def main():
    out = Path(__file__).resolve().parent.parent / "outputs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=out / "sphere-tless-fixtures.json")
    parser.add_argument("--figure", type=Path, default=out / "sphere-tless-scenes.png")
    args = parser.parse_args()
    examples, cases, frames = [], [], []
    with load_archive("train") as archive:
        scene = f"train_primesense/{PART:06d}"
        info = json.loads(archive.read(scene + "/scene_gt_info.json"))
        for frame in TRAIN_FRAMES:
            examples.append(candidate(archive, scene, frame, 0, PART, info[str(frame)][0],
                                      frame_inputs(archive, scene, frame), "training"))
    with load_archive("test") as archive:
        names = set(archive.namelist())
        for scene_id in SCENES:
            scene = f"test_primesense/{scene_id:06d}"
            available = {int(name.rsplit("/", 1)[-1][:-4]) for name in names if name.startswith(scene + "/rgb/")}
            gt = json.loads(archive.read(scene + "/scene_gt.json"))
            info = json.loads(archive.read(scene + "/scene_gt_info.json"))
            for band, low, high, center in BANDS:
                frame, target = frame_choice(gt, info, available, low, high, center)
                detections, metadata = gt[str(frame)], info[str(frame)]
                decoy = distractor(detections, metadata, target)
                loaded = frame_inputs(archive, scene, frame)
                a, b = [candidate(archive, scene, frame, i, detections[i]["obj_id"], metadata[i], loaded, band)
                        for i in (target, decoy)]
                cases.extend((a, b))
                if scene_id in (SCENES[0], SCENES[-1]):
                    frames.append((loaded[0], a, b))
                print(f"Scene {scene_id}, frame {frame}, {band}: target {a['visibility']:.1%}, distractor {b['part']}", flush=True)
    manifest = {
        "source": {"title": "T-LESS", "authors": "Tomas Hodan et al.", "license": "CC BY 4.0",
                   "licenseURL": "https://creativecommons.org/licenses/by/4.0/",
                   "project": "https://cmp.felk.cvut.cz/t-less/", "revision": REVISION,
                   "archiveURLs": [BASE + pair[0] for pair in ARCHIVES.values()]},
        "size": GRID, "targetPart": PART, "examples": examples, "cases": cases,
    }
    args.fixtures.write_text(json.dumps(manifest, separators=(",", ":")) + "\n", encoding="utf-8")
    figure(frames, args.figure)


if __name__ == "__main__":
    main()
