"""
Roboflow SDK MCP Server
Exposes Roboflow platform API (datasets, training, Universe, inference) as Claude Code tools.
"""

import io
import os
import sys
import contextlib
import requests
from fastmcp import FastMCP

mcp = FastMCP("Roboflow")

# ---------------------------------------------------------------------------
# Lazy SDK init — authenticate once per session, suppress stdout
# (Roboflow SDK prints to stdout on init which corrupts MCP stdio transport)
# ---------------------------------------------------------------------------

_rf = None
_ws = None


@contextlib.contextmanager
def _suppress_stdout():
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


def get_rf():
    global _rf
    if _rf is None:
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY environment variable is required")
        import roboflow
        with _suppress_stdout():
            _rf = roboflow.Roboflow(api_key=api_key)
    return _rf


def get_ws(workspace_url: str = ""):
    global _ws
    rf = get_rf()
    if workspace_url:
        with _suppress_stdout():
            return rf.workspace(workspace_url)
    if _ws is None:
        with _suppress_stdout():
            _ws = rf.workspace()
    return _ws


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def list_workspaces() -> dict:
    """List the workspace accessible with the current API key.
    Returns workspace name, URL slug, and number of projects."""
    try:
        ws = get_ws()
        return {
            "name": getattr(ws, "name", None),
            "url": getattr(ws, "url", None),
            "project_count": len(getattr(ws, "project_list", [])),
        }
    except Exception as e:
        return {"error": type(e).__name__, "message": str(e)}


@mcp.tool()
def list_projects(workspace_url: str = "") -> list:
    """List all projects in a workspace.
    Each entry includes project id, name, type, annotation group, and image count.
    Leave workspace_url empty to use the default workspace."""
    try:
        ws = get_ws(workspace_url)
        return [
            {
                "id": p.get("id"),
                "name": p.get("name"),
                "type": p.get("type"),
                "annotation": p.get("annotation", ""),
                "images": p.get("images", 0),
                "classes": p.get("classes", {}),
            }
            for p in getattr(ws, "project_list", [])
        ]
    except Exception as e:
        return [{"error": type(e).__name__, "message": str(e)}]


@mcp.tool()
def get_project(project_url: str, workspace_url: str = "") -> dict:
    """Get detailed info about a project: class list, image count, annotation type.
    project_url is the URL slug (e.g. 'my-dataset').
    Leave workspace_url empty for the default workspace."""
    try:
        ws = get_ws(workspace_url)
        with _suppress_stdout():
            proj = ws.project(project_url)
        return {
            "id": getattr(proj, "id", None),
            "name": getattr(proj, "name", None),
            "type": getattr(proj, "type", None),
            "annotation": getattr(proj, "annotation", None),
            "classes": getattr(proj, "classes", {}),
        }
    except Exception as e:
        return {"error": type(e).__name__, "message": str(e)}


@mcp.tool()
def list_versions(project_url: str, workspace_url: str = "") -> list:
    """List all dataset versions for a project.
    Returns version number, id, and image counts per split."""
    try:
        ws = get_ws(workspace_url)
        with _suppress_stdout():
            proj = ws.project(project_url)
            versions = proj.versions()
        return [
            {
                "version": getattr(v, "version", None),
                "id": getattr(v, "id", None),
                "images": getattr(v, "images", None),
            }
            for v in versions
        ]
    except Exception as e:
        return [{"error": type(e).__name__, "message": str(e)}]


@mcp.tool()
def upload_image(
    project_url: str,
    image_path: str,
    annotation_path: str = "",
    split: str = "train",
    batch_name: str = "",
    workspace_url: str = "",
) -> dict:
    """Upload an image (and optional annotation) to a Roboflow project.
    image_path: absolute local path to the image file.
    annotation_path: optional absolute path to annotation file (PASCAL VOC XML, COCO JSON, etc).
    split: 'train', 'valid', or 'test' (default: 'train').
    batch_name: optional batch label for grouping uploads."""
    try:
        ws = get_ws(workspace_url)
        with _suppress_stdout():
            proj = ws.project(project_url)
        kwargs = {"split": split, "num_retry_uploads": 2}
        if annotation_path:
            kwargs["annotation_path"] = annotation_path
        if batch_name:
            kwargs["batch_name"] = batch_name
        with _suppress_stdout():
            result = proj.upload(image_path, **kwargs)
        return {"success": True, "image_path": image_path, "result": str(result)}
    except Exception as e:
        return {"error": type(e).__name__, "message": str(e)}


@mcp.tool()
def create_version(
    project_url: str,
    preprocessing: dict = {},
    augmentation: dict = {},
    workspace_url: str = "",
) -> dict:
    """Generate a new dataset version with preprocessing and augmentation settings.
    preprocessing example: {"auto-orient": True, "resize": {"width": 640, "height": 640, "format": "Stretch to"}}
    augmentation example: {"flip": {"horizontal": True}, "rotation": {"degrees": 15}}
    Leave both empty to generate a version with no modifications."""
    try:
        ws = get_ws(workspace_url)
        with _suppress_stdout():
            proj = ws.project(project_url)
        settings = {}
        if preprocessing:
            settings["preprocessing"] = preprocessing
        if augmentation:
            settings["augmentation"] = augmentation
        with _suppress_stdout():
            version = proj.generate_version(settings)
        return {"success": True, "version": str(version)}
    except Exception as e:
        return {"error": type(e).__name__, "message": str(e)}


@mcp.tool()
def download_dataset(
    project_url: str,
    version_number: int,
    format: str = "yolov8",
    location: str = "",
    workspace_url: str = "",
) -> dict:
    """Download a dataset version to a local directory.
    Supported formats: yolov8, yolov5, coco, voc, tensorflow, darknet, createml, multiclass.
    Returns the local path where the dataset was saved.
    Leave location empty to download to the current directory."""
    try:
        ws = get_ws(workspace_url)
        with _suppress_stdout():
            proj = ws.project(project_url)
            ver = proj.version(version_number)
        kwargs = {"model_format": format, "overwrite": True}
        if location:
            kwargs["location"] = location
        with _suppress_stdout():
            dataset = ver.download(**kwargs)
        return {
            "location": getattr(dataset, "location", location),
            "format": format,
            "version": version_number,
        }
    except Exception as e:
        return {"error": type(e).__name__, "message": str(e)}


@mcp.tool()
def search_universe(query: str, page: int = 1, results_per_page: int = 10) -> list:
    """Search Roboflow Universe for public datasets and pre-trained models.
    Returns matching projects with name, URL, type, image count, and class list.
    Great for finding existing datasets before starting annotation from scratch."""
    try:
        api_key = os.environ.get("ROBOFLOW_API_KEY", "")
        resp = requests.get(
            "https://api.roboflow.com/search",
            params={"q": query, "page": page, "resultsPerPage": results_per_page},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        return [
            {
                "name": r.get("name"),
                "url": r.get("url"),
                "type": r.get("type"),
                "images": r.get("images"),
                "classes": r.get("classes"),
                "universe_url": f"https://universe.roboflow.com/{r.get('url', '')}",
            }
            for r in results
        ]
    except Exception as e:
        return [{"error": type(e).__name__, "message": str(e)}]


@mcp.tool()
def run_inference(
    project_url: str,
    version_number: int,
    image_path: str,
    confidence: int = 50,
    overlap: int = 50,
    workspace_url: str = "",
) -> dict:
    """Run inference on an image using a deployed Roboflow model.
    image_path: absolute local file path or a public URL.
    confidence: minimum confidence threshold 0-100 (default 50).
    overlap: maximum bounding box overlap 0-100 (default 50).
    Returns predictions with bounding boxes, classes, and confidence scores."""
    try:
        ws = get_ws(workspace_url)
        with _suppress_stdout():
            proj = ws.project(project_url)
            ver = proj.version(version_number)
            model = ver.model
        with _suppress_stdout():
            predictions = model.predict(image_path, confidence=confidence, overlap=overlap)
        return predictions.json()
    except Exception as e:
        return {"error": type(e).__name__, "message": str(e)}


@mcp.tool()
def get_model_metrics(
    project_url: str,
    version_number: int,
    workspace_url: str = "",
) -> dict:
    """Get model training metrics (mAP, precision, recall) for a trained version.
    Returns performance metrics if the model has been trained on Roboflow."""
    try:
        rf = get_rf()
        api_key = os.environ.get("ROBOFLOW_API_KEY", "")
        ws_url = workspace_url or getattr(get_ws(), "url", "")
        resp = requests.get(
            f"https://api.roboflow.com/{ws_url}/{project_url}/{version_number}",
            params={"api_key": api_key},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        version_data = data.get("version", {})
        model_data = version_data.get("model", {})
        return {
            "version": version_number,
            "mAP": model_data.get("map"),
            "precision": model_data.get("precision"),
            "recall": model_data.get("recall"),
            "training_status": version_data.get("training", {}).get("status"),
        }
    except Exception as e:
        return {"error": type(e).__name__, "message": str(e)}


@mcp.tool()
def download_universe_dataset(
    universe_workspace: str,
    universe_project: str,
    version_number: int,
    format: str = "yolov8",
    location: str = "",
) -> dict:
    """Download a public dataset from Roboflow Universe directly.
    universe_workspace: workspace slug on Universe (e.g. 'brad-dwyer').
    universe_project: project slug (e.g. 'chess-pieces-new').
    version_number: dataset version to download.
    Supported formats: yolov8, yolov5, coco, voc, tensorflow, darknet."""
    try:
        rf = get_rf()
        api_key = os.environ.get("ROBOFLOW_API_KEY", "")
        with _suppress_stdout():
            proj = rf.workspace(universe_workspace).project(universe_project)
            ver = proj.version(version_number)
        kwargs = {"model_format": format, "overwrite": True}
        if location:
            kwargs["location"] = location
        with _suppress_stdout():
            dataset = ver.download(**kwargs)
        return {
            "location": getattr(dataset, "location", location),
            "format": format,
            "version": version_number,
            "source": f"https://universe.roboflow.com/{universe_workspace}/{universe_project}",
        }
    except Exception as e:
        return {"error": type(e).__name__, "message": str(e)}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
