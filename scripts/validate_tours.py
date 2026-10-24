"""Validate this repository's CodeTour navigation without third-party packages."""

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve_path(value, label):
    require(isinstance(value, str) and value, f"{label}: expected a path")
    require(not Path(value).is_absolute(), f"{label}: path must be relative")
    path = (ROOT / value).resolve()
    require(path.is_relative_to(ROOT), f"{label}: path escapes repository")
    return path


def validate():
    files = sorted((ROOT / ".tours").rglob("*.tour"))
    require(files, "No .tours/*.tour files found")
    tours = {}
    total_steps = 0
    for file in files:
        tour = json.loads(file.read_text(encoding="utf-8"))
        require(isinstance(tour, dict), f"{file.name}: expected an object")
        title = tour.get("title")
        require(isinstance(title, str) and title.strip(), f"{file.name}: missing title")
        require(title not in tours, f"Duplicate tour title: {title}")
        require(tour.get("$schema") == "https://aka.ms/codetour-schema",
                f"{file.name}: missing official schema reference")
        require(isinstance(tour.get("isPrimary", False), bool), f"{title}: invalid isPrimary")
        steps = tour.get("steps")
        require(isinstance(steps, list) and steps, f"{title}: no steps")
        tours[title] = tour
        for number, step in enumerate(steps, 1):
            label = f"{file.name}, step {number}"
            require(isinstance(step, dict), f"{label}: expected an object")
            require(isinstance(step.get("description"), str) and step["description"].strip(),
                    f"{label}: missing description")
            if "file" in step:
                path = resolve_path(step["file"], label)
                require(path.is_file(), f"{label}: file does not exist: {path}")
                source = path.read_text(encoding="utf-8")
                require(not ("line" in step and "pattern" in step),
                        f"{label}: line overrides pattern; use only one")
                if "line" in step:
                    line = step["line"]
                    require(type(line) is int and 1 <= line <= len(source.splitlines()),
                            f"{label}: invalid 1-based line")
                if "pattern" in step:
                    pattern = step["pattern"]
                    require(isinstance(pattern, str) and pattern, f"{label}: invalid pattern")
                    matches = list(re.finditer(pattern, source, re.MULTILINE))
                    require(len(matches) == 1,
                            f"{label}: pattern matched {len(matches)} times in {step['file']}")
            else:
                require("line" not in step and "pattern" not in step,
                        f"{label}: source anchor requires a file")
            if "directory" in step:
                require(resolve_path(step["directory"], label).is_dir(),
                        f"{label}: directory does not exist")
            total_steps += 1
    require(sum(tour.get("isPrimary", False) for tour in tours.values()) == 1,
            "Expected exactly one primary tour")
    for title in tours:
        seen = set()
        current = title
        while current is not None:
            require(current in tours, f"{title}: nextTour target does not exist: {current}")
            require(current not in seen, f"{title}: nextTour cycle")
            seen.add(current)
            current = tours[current].get("nextTour")
    print(f"Validated {len(tours)} tours, {total_steps} steps, source anchors, and tour links.")


if __name__ == "__main__":
    try:
        validate()
    except (ValueError, OSError, TypeError) as exc:
        raise SystemExit(f"Tour validation failed: {exc}") from exc
