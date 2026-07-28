"""
PixelSentinel — Pipeline Manager (V2.0)

Orchestrates sequential dataset preprocessing stages:
  1. Quality Validation
  2. Band Extraction
  3. Normalization
  4. Image Registration Alignment
  5. Tiling (256x256 Overlapping Patches)
  6. Data Augmentation
  7. Dataset Partitioning (Train / Val / Test)

Manages execution order, stage result reporting, failure handling, 
and automatic pipeline termination on critical errors.
"""

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
from preprocessing.resource_monitor import (
    stage_start,
    stage_end,
)
# Import configuration and pipeline utilities
try:
    from preprocessing.config import (
        AUTO_CLEAN_OUTPUT,
        AUTO_COPY_INCOMING,
        GENERATE_MANIFEST,
        GENERATE_REPORT,
        PRINT_PIPELINE_SUMMARY,
        STOP_ON_ERROR,
        VERIFY_AFTER_EACH_STAGE,
    )
    from preprocessing.pipeline_utils import (
        PipelineTimer,
        clean_processed_dataset,
        copy_incoming_to_raw,
        get_dataset_counts,
        print_pipeline_summary,
        write_dataset_manifest,
        write_pipeline_report,
    )
except ImportError:
    # Fallback default flags
    AUTO_COPY_INCOMING = True
    AUTO_CLEAN_OUTPUT = True
    STOP_ON_ERROR = True
    VERIFY_AFTER_EACH_STAGE = True
    GENERATE_MANIFEST = True
    GENERATE_REPORT = True
    PRINT_PIPELINE_SUMMARY = True

# Dynamic import helper for preprocessing modules
def _import_stage_module(module_name: str):
    """Safely imports a preprocessing module given its module name."""
    try:
        return __import__(f"preprocessing.{module_name}", fromlist=["*"])
    except ImportError as e:
        logger.warning(f"Could not import stage module 'preprocessing.{module_name}': {e}")
        return None

logger = logging.getLogger("PixelSentinel.PipelineManager")


@dataclass
class StageResult:
    """Unified result data structure returned by each pipeline stage."""
    stage_name: str
    success: bool = False
    processed: int = 0
    failed: int = 0
    execution_time: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "success": self.success,
            "processed": self.processed,
            "failed": self.failed,
            "execution_time": round(self.execution_time, 2),
            "details": self.details,
        }


class PipelineManager:
    """
    Coordinates end-to-end execution of PixelSentinel preprocessing stages.
    """

    def __init__(
        self,
        auto_copy_incoming: bool = AUTO_COPY_INCOMING,
        auto_clean_output: bool = AUTO_CLEAN_OUTPUT,
        stop_on_error: bool = STOP_ON_ERROR,
        verify_after_stage: bool = VERIFY_AFTER_EACH_STAGE,
    ):
        self.auto_copy_incoming = auto_copy_incoming
        self.auto_clean_output = auto_clean_output
        self.stop_on_error = stop_on_error
        self.verify_after_stage = verify_after_stage
        
        self.results: Dict[str, Dict[str, Any]] = {}
        self.overall_timer = PipelineTimer()

    def run_stage(self, stage_name: str, module_name: str, entry_function_name: str = "process_dataset") -> StageResult:
        """
        Executes a single pipeline stage safely with timing and error isolation.
        """

        logger.info("=" * 80)
        logger.info(f"\n>>> Starting Stage: {stage_name}")

        # Wait until system resources are safe
        stage_start(stage_name)

        timer = PipelineTimer().start()
        
        mod = _import_stage_module(module_name)
        if mod is None or not hasattr(mod, entry_function_name):
            err_msg = f"Module '{module_name}' or entry point '{entry_function_name}' not found."
            logger.error(f"Stage '{stage_name}' failed: {err_msg}")
            res = StageResult(
                stage_name=stage_name,
                success=False,
                execution_time=timer.stop(),
                details={"error": err_msg}
            )
            self.results[stage_name] = res.to_dict()
            return res

        try:
            func = getattr(mod, entry_function_name)
            output = func()

            # Normalize output structure
            if isinstance(output, StageResult):
                result = output
                result.execution_time = timer.stop()
            elif isinstance(output, dict):
                result = StageResult(
                    stage_name=stage_name,
                    success=output.get("success", True),
                    processed=output.get("processed", 0),
                    failed=output.get("failed", 0),
                    execution_time=timer.stop(),
                    details=output
                )
            else:
                # Fallback for void/legacy functions returning booleans or counts
                success = bool(output) if output is not None else True
                result = StageResult(
                    stage_name=stage_name,
                    success=success,
                    processed=output if isinstance(output, int) else 0,
                    failed=0,
                    execution_time=timer.stop()
                )

            logger.info(
                f"Completed Stage '{stage_name}' in {result.execution_time:.2f}s | "
                f"Processed: {result.processed}, Failed: {result.failed}"
            )
            stage_end(
                stage_name,
                result.execution_time,
            )

        except Exception as e:
            logger.exception(f"Unhandled exception in stage '{stage_name}': {e}")
            result = StageResult(
                stage_name=stage_name,
                success=False,
                execution_time=timer.stop(),
                details={"exception": str(e)}
            )
            stage_end(
                stage_name,
                result.execution_time,
            )
            

        self.results[stage_name] = result.to_dict()

        status = "SUCCESS" if result.success else "FAILED"

        logger.info(
            f"{stage_name} : {status}"
        )
      
        return result

    def run(self) -> bool:
        """
        Runs the full end-to-end preprocessing pipeline sequentially.
        """
        self.overall_timer.start()
        logger.info("Initializing PixelSentinel Preprocessing Pipeline V2.0...")

        # --------------------------------------------------
        # STAGE 0: ENVIRONMENT PREPARATION & STAGING
        # --------------------------------------------------
        copied = 0
        skipped = 0

        if self.auto_clean_output:
            clean_processed_dataset()

        if self.auto_copy_incoming:
            copied, skipped = copy_incoming_to_raw()

        logger.info(
            f"Incoming Files Copied : {copied}"
        )

        logger.info(
            f"Incoming Files Skipped : {skipped}"
        )

        # --------------------------------------------------
        # STAGE EXECUTION PIPELINE
        # --------------------------------------------------
        stages = [

            (
                "Quality Validation",
                "quality",
                "validate_dataset",
            ),

            (
                "Band Extraction",
                "band_extraction",
                "process_dataset",
            ),

            (
                "Normalization",
                "normalization",
                "process_dataset",
            ),

            (
                "Image Registration",
                "image_registration",
                "verify_registration",
            ),

            (
                "Tiling",
                "tiling",
                "process_dataset",
            ),

            (
                "Dataset Split",
                "dataset_split",
                "split_dataset",
            ),
        ]

        pipeline_successful = True

        for stage_name, module_name, entry_fn in stages:
            stage_res = self.run_stage(stage_name, module_name, entry_fn)
            
            if not stage_res.success:
                pipeline_successful = False
                logger.error(f"Stage '{stage_name}' reported failure.")
                
                if self.stop_on_error:
                    logger.critical(f"Stopping pipeline execution due to error in stage: {stage_name}")
                    break

        total_time = self.overall_timer.stop()

        # --------------------------------------------------
        # REPORTS & MANIFEST GENERATION
        # --------------------------------------------------
        if GENERATE_REPORT:
            write_pipeline_report({
                "pipeline_success": pipeline_successful,
                "total_time_seconds": round(total_time, 2),
                "stages": self.results
            })

        if GENERATE_MANIFEST and pipeline_successful:
            write_dataset_manifest(
                stage_results=self.results,
                total_execution_time=total_time
            )

        if PRINT_PIPELINE_SUMMARY:
            print_pipeline_summary(self.results, total_time)

        return pipeline_successful


if __name__ == "__main__":
    # Configure standalone logger if executed directly
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    manager = PipelineManager()
    success = manager.run()
    sys.exit(0 if success else 1)