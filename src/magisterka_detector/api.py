"""Public API for GUI/CLI - main entrypoint."""

import os
from typing import List, Optional

from .types import AnalyzeOptions, AnalysisResult, ProgressCb, CancelCb
from .reporting.run_dir import create_run_dir
from .core.pipeline import run_pipeline


def begin_run(reports_root: str = "reports") -> str:
    """Create a new run directory for this analysis session.
    
    Args:
        reports_root: Base directory for all reports
        
    Returns:
        Path to the created run directory
    """
    return create_run_dir(reports_root)


def analyze_media(
    path: str,
    *,
    opts: Optional[AnalyzeOptions] = None,
    progress: Optional[ProgressCb] = None,
    cancel: Optional[CancelCb] = None,
    run_dir: Optional[str] = None,
) -> AnalysisResult:
    """Analyze a single media file (video or image) for deepfake detection.
    
    Args:
        path: Path to video or image file
        opts: Analysis options (uses defaults if None)
        progress: Optional callback for progress updates (current, total)
        cancel: Optional callback to check if analysis should be cancelled
        run_dir: Optional run directory (created via begin_run())
        
    Returns:
        AnalysisResult with verdict, score, features, and report paths
    """
    if opts is None:
        opts = AnalyzeOptions()
    
    if run_dir is None:
        run_dir = create_run_dir(opts.reports_root)
    
    return run_pipeline(
        path=path,
        opts=opts,
        progress=progress,
        cancel=cancel,
        run_dir=run_dir,
    )


def analyze_batch(
    paths: List[str],
    *,
    opts: Optional[AnalyzeOptions] = None,
    progress: Optional[ProgressCb] = None,
    cancel: Optional[CancelCb] = None,
    run_dir: Optional[str] = None,
) -> List[AnalysisResult]:
    """Analyze multiple media files in batch.
    
    Args:
        paths: List of paths to video/image files
        opts: Analysis options (uses defaults if None)
        progress: Optional global progress callback
        cancel: Optional callback to check if batch should be cancelled
        run_dir: Optional run directory (created via begin_run())
        
    Returns:
        List of AnalysisResult, one per input file
    """
    if opts is None:
        opts = AnalyzeOptions()
    
    if run_dir is None:
        run_dir = create_run_dir(opts.reports_root)
    
    results = []
    total = len(paths)
    
    for idx, path in enumerate(paths):
        if cancel and cancel():
            break
        
        def file_progress(curr: int, tot: int):
            if progress:
                # Global progress: (files_done * frames_per_file + curr) / (total_files * frames_per_file)
                global_curr = idx * 100 + int(curr * 100 / max(tot, 1))
                global_total = total * 100
                progress(global_curr, global_total)
        
        result = run_pipeline(
            path=path,
            opts=opts,
            progress=file_progress,
            cancel=cancel,
            run_dir=run_dir,
        )
        results.append(result)
    
    return results
