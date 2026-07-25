"""
PixelSentinel Collection Manager

NEW Dataset Collection
Unique ROIs not included in the original dataset.

These batches increase geographical diversity for Pix2Pix training.
"""

from preprocessing.dataset_builder import build_dataset


# ==========================================================
# Batch 1
# Agriculture + Forest
# Years 2023-2024
# ==========================================================

BATCH_1 = (
    "Agriculture_Forest_New_2023_2024",
    {

        # Agriculture
        "Ludhiana_Farmland": [75.72, 30.82, 76.02, 31.02],
        "Godavari_Delta": [81.72, 16.55, 82.02, 16.75],
        "Krishna_Basin": [80.55, 16.05, 80.85, 16.25],
        "Cauvery_Delta": [79.20, 10.85, 79.50, 11.05],
        "Indore_Farmland": [75.65, 22.55, 75.95, 22.75],
        "Raipur_Farmland": [81.45, 21.15, 81.75, 21.35],

        # Forest
        "Periyar": [77.05, 9.45, 77.35, 9.65],
        "Dandeli": [74.55, 15.10, 74.85, 15.30],
        "Simlipal": [86.20, 21.70, 86.50, 21.90],
        "Satpura": [78.00, 22.40, 78.30, 22.60],
        "Melghat": [77.00, 21.20, 77.30, 21.40],
        "Nagarhole": [76.00, 12.00, 76.30, 12.20],

    },
    [2023, 2024],
)


# ==========================================================
# Batch 2
# Agriculture + Forest
# Years 2025-2026
# ==========================================================

BATCH_2 = (
    "Agriculture_Forest_New_2025_2026",
    {

        # Agriculture
        "Ludhiana_Farmland": [75.72, 30.82, 76.02, 31.02],
        "Godavari_Delta": [81.72, 16.55, 82.02, 16.75],
        "Krishna_Basin": [80.55, 16.05, 80.85, 16.25],
        "Cauvery_Delta": [79.20, 10.85, 79.50, 11.05],
        "Indore_Farmland": [75.65, 22.55, 75.95, 22.75],
        "Raipur_Farmland": [81.45, 21.15, 81.75, 21.35],

        # Forest
        "Periyar": [77.05, 9.45, 77.35, 9.65],
        "Dandeli": [74.55, 15.10, 74.85, 15.30],
        "Simlipal": [86.20, 21.70, 86.50, 21.90],
        "Satpura": [78.00, 22.40, 78.30, 22.60],
        "Melghat": [77.00, 21.20, 77.30, 21.40],
        "Nagarhole": [76.00, 12.00, 76.30, 12.20],

    },
    [2025, 2026],
)


# ==========================================================
# Batch 3
# Industrial + Airports + Ports
# Years 2025-2026
# ==========================================================

BATCH_3 = (
    "Industrial_Airports_Ports_New_2025_2026",
    {

        # Industrial
        "AngulIndustrial": [85.05, 20.85, 85.35, 21.05],
        "KorbaIndustrial": [82.55, 22.20, 82.85, 22.40],
        "NeyveliIndustrial": [79.45, 11.45, 79.75, 11.65],

        # Airports
        "HyderabadAirport": [78.35, 17.15, 78.50, 17.30],
        "KolkataAirport": [88.40, 22.62, 88.55, 22.77],
        "KochiAirport": [76.35, 10.10, 76.50, 10.25],

        # Ports
        "ParadipPort": [86.55, 20.20, 86.70, 20.35],
        "KandlaPort": [70.15, 22.95, 70.30, 23.10],
        "VizhinjamPort": [76.90, 8.30, 77.05, 8.45],

    },
    [2025, 2026],
)


# ==========================================================
# Batches
# ==========================================================

BATCHES = [
    BATCH_1,
    BATCH_2,
    BATCH_3,
]


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    for batch_name, roi_group, years in BATCHES:

        print("=" * 80)
        print(f"Starting {batch_name}")
        print("=" * 80)

        build_dataset(
            roi_list=roi_group,
            years=years,
        )

        print(f"{batch_name} completed.\n")