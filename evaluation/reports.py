"""Generate evaluation reports."""

import json
from typing import Dict


def generate_report(
    results: Dict,
    output_path: str = 'reports/evaluation_report.json'
) -> None:
    """Generate evaluation report.
    
    Args:
        results: Evaluation results
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)