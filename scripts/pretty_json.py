import json
from pathlib import Path

import fire


def write_pretty_json(path: str, out: str) -> None:
    path, out = Path(path), Path(out)
    with open(path, "r") as file:
        content = json.load(file)
    out.parent.mkdir(exist_ok=True, parents=True)
    with open(out, "w") as file:
        json.dump(content, file, indent=2, sort_keys=True)


if __name__ == "__main__":
    fire.Fire(write_pretty_json)
