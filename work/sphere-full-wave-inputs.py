"""Encode isolated examples and complete T-LESS photos as luminance rasters."""
import argparse
import json
from pathlib import Path
import runpy

from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
PHOTOS = runpy.run_path(str(HERE / "sphere-tless-photos-fetch.py"))
REFERENCE_FRAMES = PHOTOS["REFERENCES"][::2]
REFERENCE_SIZE = 25
FRAME_SIZE = (144, 108)
ISOLATED_IDS = {"cal-01-000430", "cal-01-000865", "cal-01-001295", "cal-02-000000",
                "cal-03-000000", "cal-04-000000", "cal-07-000000"}
PADDING = (80, 10, 80, 10)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def values(image, size):
    return list(image.convert("L").resize(size, Image.Resampling.BOX).get_flattened_data())


def reference(archive, folder, frame, annotation):
    path = folder / f"{frame:06d}.png"
    if not path.is_file():
        path.write_bytes(archive.read(f"train_primesense/000001/rgb/{frame:06d}.png"))
    x, y, w, h = annotation[str(frame)][0]["bbox_visib"]
    side = round(max(w, h) * 1.25)
    left, top = x + (w - side) // 2, y + (h - side) // 2
    with Image.open(path) as photo:
        return values(photo.crop((left, top, left + side, top + side)), (REFERENCE_SIZE, REFERENCE_SIZE))


def prepare_references(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with PHOTOS["ARCHIVE"]["load_archive"]("train") as archive:
        info = PHOTOS["annotation"](archive, "train_primesense/000001", "gt_info")
        return [{"frames": [frame, frame + 1], "a": reference(archive, folder, frame, info),
                 "b": reference(archive, folder, frame + 1, info)} for frame in REFERENCE_FRAMES]


def synthetic_frame(ref):
    width, height = FRAME_SIZE
    left, top = 94, 29
    pixels = [12] * (width * height)
    for y in range(REFERENCE_SIZE):
        for x in range(REFERENCE_SIZE):
            i = y * REFERENCE_SIZE + x
            pixels[(top + y) * width + left + x] = round((ref["a"][i] * ref["b"][i]) ** .5)
    for y in range(60, 76):
        for x in range(41, 49):
            pixels[y * width + x] = 146
    return {"id": "synthetic-transfer", "originalSize": list(FRAME_SIZE), "pixels": pixels}


def encoded_frames(manifest, isolated):
    for row in read(manifest):
        if isolated and row["id"] not in ISOLATED_IDS:
            continue
        with Image.open(manifest.parent / row["rgb"]) as photo:
            image = ImageOps.expand(photo, border=PADDING, fill=0) if isolated else photo
            yield {"id": row["id"], "originalSize": list(image.size), "pixels": values(image, FRAME_SIZE)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    panel = parser.add_mutually_exclusive_group()
    panel.add_argument("--scenes", type=Path)
    panel.add_argument("--isolated", type=Path, help="Encode the seven fixed isolated calibration photographs")
    panel.add_argument("--synthetic", action="store_true")
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--references", type=Path, help="Reuse previously encoded reference pairs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    refs = read(args.references)["references"] if args.references else prepare_references(args.reference_cache)
    records = list(encoded_frames(args.scenes or args.isolated, bool(args.isolated))) if args.scenes or args.isolated else []
    if args.synthetic:
        records.append(synthetic_frame(refs[0]))
    PHOTOS["dump"](args.output, {"frameSize": FRAME_SIZE, "referenceSize": REFERENCE_SIZE,
                                "references": refs, "frames": records})


if __name__ == "__main__":
    main()
