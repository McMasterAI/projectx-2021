import gzip
import json
import os
import pathlib
import urllib.request

from tqdm import tqdm


def string_db_pull_process(out_dir):
    # API has limits on the number of requests. Need to download directly to get all the interactions.
    print("Downloading string-db...")
    addr = "https://stringdb-static.org/download/protein.links.full.v11.5/9606.protein.links.full.v11.5.txt.gz"
    urllib.request.urlretrieve(addr, "temp.txt.gz")
    print("done.")

    with gzip.open("temp.txt.gz") as f:
        lines = f.read().decode().split("\n")

    string_db = {}
    header = lines[0].split(" ")

    # Dump into json
    for line in tqdm(lines[1:]):
        line = line.split(" ")
        string_db[line[0]] = dict(zip(header[1:], line[1:]))

    os.remove("temp.txt.gz")

    pathlib.Path(out_dir).mkdir(exist_ok=True, parents=True)
    json.dump(string_db, open(os.path.join(out_dir, "string_db.json"), "w"))


if __name__ == "__main__":
    string_db_pull_process("./Corpuses")
