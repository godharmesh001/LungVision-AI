from pathlib import Path
from PIL import Image
from collections import Counter

# =====================================================
# DATASET CONFIGURATION
# =====================================================

DATASET_PATH = Path("/Users/dharmeshkumar/Downloads/archive (5)")

CLASSES = [
    "adenocarcinoma",
    "benign",
    "squamous_cell_carcinoma"
]

REPORT_PATH = Path("reports/dataset_report.txt")


# =====================================================
# DATASET INSPECTOR
# =====================================================

formats = Counter()
sizes = Counter()
color_modes = Counter()
corrupted = []
total_images = 0

report = []

report.append("=" * 60)
report.append("LUNG CANCER DATASET REPORT")
report.append("=" * 60)

for cls in CLASSES:

    folder = DATASET_PATH / cls
    images = list(folder.iterdir())

    report.append(f"\nClass : {cls}")
    report.append(f"Images : {len(images)}")

    total_images += len(images)

    for image_path in images:

        try:

            with Image.open(image_path) as img:

                formats[img.format] += 1
                sizes[img.size] += 1
                color_modes[img.mode] += 1

        except Exception:

            corrupted.append(image_path.name)

report.append("\n" + "=" * 60)
report.append("SUMMARY")
report.append("=" * 60)

report.append(f"Total Images : {total_images}")
report.append(f"Formats : {dict(formats)}")
report.append(f"Color Modes : {dict(color_modes)}")
report.append(f"Unique Image Sizes : {len(sizes)}")
report.append(f"Corrupted Images : {len(corrupted)}")

REPORT_PATH.parent.mkdir(exist_ok=True)

with open(REPORT_PATH, "w") as file:

    file.write("\n".join(report))

print("\n".join(report))
print("\n✅ Report saved to:", REPORT_PATH)