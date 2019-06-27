import sys
from pathlib import Path

import dask.dataframe as dd
import tqdm

if __name__ == "__main__":
    location = sys.argv[1]
    if sys.argv[2]:
        start = int(sys.argv[2])
    else:
        start = 0
    print(start)

    last_generation = 2000
    groupbys = ["opponent"] + ["gene_{}".format(i) for i in range(205)]

    for i, path in enumerate(tqdm.tqdm(Path(location).glob("*"))):
        if i > start:
            strategy = str(path).split("/")[-1]

            df = dd.read_csv("%s/main.csv" % path)
            df = df[(df["generation"] == last_generation)]

            df = (
                df[df["score"] == df["score"].max()][groupbys]
                .drop_duplicates()
                .reset_index()
            )
            df = df.compute(num_workers=4)
            df.to_csv("raw_data/%s_top_individuals.csv" % strategy)