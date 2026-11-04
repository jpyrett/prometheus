"""Quick test of DataToolbox conversion: take the first N rows of a real trades_v1 and quotes_v1 file,
convert them with DataToolbox._convert_file, and check rows, schema and values. Takes seconds.
Run: python test_convert.py [rows]   (default 1,000,000)"""
import gzip
import itertools
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from data_toolbox import ARROW_TYPES, SCHEMAS, DataToolbox

DAY = "2026-10-01"


def sample(src: Path, dst: Path, rows: int) -> None:
    with gzip.open(src, "rt") as f, gzip.open(dst, "wt") as out:
        out.writelines(itertools.islice(f, rows + 1))  # header + rows


def main(rows: int = 1_000_000) -> int:
    agent = DataToolbox()
    failed = False
    with tempfile.TemporaryDirectory() as tmp:
        for data_type in ("trades_v1", "quotes_v1"):
            src = agent.root_data_dir / data_type / "2026" / "10" / f"{DAY}.csv.gz"
            csv_path, out_path = Path(tmp) / f"{data_type}.csv.gz", Path(tmp) / f"{data_type}.parquet"
            sample(src, csv_path, rows)
            n, skipped = agent._convert_file(csv_path, out_path, data_type, data_type)
            print()
            table = pq.read_table(out_path)
            expected = pa.schema([(c, ARROW_TYPES[t]) for c, t in SCHEMAS[data_type].items()])
            problems = []
            if table.num_rows != n or n + skipped != rows:
                problems.append(f"rows: parquet {table.num_rows}, converted {n}, skipped {skipped}, sampled {rows}")
            if table.schema.names != expected.names:
                problems.append(f"columns {table.schema.names} != {expected.names}")
            else:
                for field in expected:
                    if table.schema.field(field.name).type != field.type:
                        problems.append(f"{field.name}: {table.schema.field(field.name).type} != {field.type}")
            nulls = {c: table[c].null_count for c in table.schema.names if table[c].null_count == table.num_rows}
            if nulls:
                problems.append(f"all-null columns: {list(nulls)}")
            print(f"{data_type}: {n:,} rows, {skipped} skipped -> {'FAIL' if problems else 'OK'}")
            for p in problems:
                print("  -", p)
            print(table.slice(0, 2).to_pylist())
            failed |= bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000))
