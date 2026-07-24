

"""
PixelSentinel Collection Manager

Automatically collects the remaining dataset batches.

Already Downloaded
------------------
✓ Urban (2023-2026)
✓ Water Bodies (2023-2026)
✓ Desert (2023-2026)
✓ Mountains (2023-2026)
✓ Industrial (2023-2024)
✓ Airports (2023-2024)
✓ Ports (2023-2024)

Remaining
---------
• Agriculture (2023-2026)
• Forest (2023-2026)
• Industrial (2025-2026)
• Airports (2025-2026)
• Ports (2025-2026)
"""

from preprocessing.dataset_builder import build_dataset


# ==========================================================
# Batch 1
# Agriculture + Forest
# Years 2023-2024
# ==========================================================

BATCH_1 = (
    "Agriculture_Forest_2023_2024",
    {
        "Punjab_Farmland": [75.10, 30.70, 75.40, 30.90],
        "TamilNadu_Farmland": [78.90, 10.80, 79.20, 11.00],
        "Haryana_Farmland": [76.55, 29.15, 76.85, 29.35],
        "Andhra_Farmland": [80.55, 16.30, 80.85, 16.50],
        "Maharashtra_Farmland": [74.60, 20.10, 74.90, 20.30],
        "Bihar_Farmland": [85.10, 25.40, 85.40, 25.60],

        "WesternGhats": [76.10, 10.00, 76.40, 10.20],
        "Assam_Forest": [91.50, 26.00, 91.80, 26.20],
        "Nilgiris": [76.55, 11.30, 76.85, 11.50],
        "Bandipur": [76.45, 11.60, 76.75, 11.80],
        "Kaziranga": [93.20, 26.50, 93.50, 26.70],
        "JimCorbett": [78.80, 29.40, 79.10, 29.60],
    },
    [2023, 2024],
)


# ==========================================================
# Batch 2
# Agriculture + Forest
# Years 2025-2026
# ==========================================================

BATCH_2 = (
    "Agriculture_Forest_2025_2026",
    {
        "Punjab_Farmland": [75.10, 30.70, 75.40, 30.90],
        "TamilNadu_Farmland": [78.90, 10.80, 79.20, 11.00],
        "Haryana_Farmland": [76.55, 29.15, 76.85, 29.35],
        "Andhra_Farmland": [80.55, 16.30, 80.85, 16.50],
        "Maharashtra_Farmland": [74.60, 20.10, 74.90, 20.30],
        "Bihar_Farmland": [85.10, 25.40, 85.40, 25.60],

        "WesternGhats": [76.10, 10.00, 76.40, 10.20],
        "Assam_Forest": [91.50, 26.00, 91.80, 26.20],
        "Nilgiris": [76.55, 11.30, 76.85, 11.50],
        "Bandipur": [76.45, 11.60, 76.75, 11.80],
        "Kaziranga": [93.20, 26.50, 93.50, 26.70],
        "JimCorbett": [78.80, 29.40, 79.10, 29.60],
    },
    [2025, 2026],
)


# ==========================================================
# Batch 3
# Only datasets still missing
# Industrial + Airports + Ports
# Years 2025-2026
# ==========================================================

BATCH_3 = (
    "Industrial_Airports_Ports_2025_2026",
    {
        "Jamshedpur": [86.10, 22.70, 86.40, 22.90],
        "Hazira": [72.60, 21.05, 72.90, 21.25],
        "VizagIndustrial": [83.15, 17.60, 83.45, 17.80],

        "ChennaiAirport": [80.12, 12.96, 80.22, 13.06],
        "DelhiAirport": [77.05, 28.52, 77.18, 28.62],
        "BengaluruAirport": [77.65, 13.15, 77.78, 13.28],

        "ChennaiPort": [80.27, 13.08, 80.34, 13.15],
        "MumbaiPort": [72.82, 18.90, 72.90, 18.98],
        "MundraPort": [69.65, 22.72, 69.78, 22.85],
    },
    [2025, 2026],
)


# ==========================================================
# Remaining batches
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