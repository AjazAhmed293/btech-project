from src.data.loaders import parse_cycle_number
from src.data.splits import split_cycles


def test_split_cycles_basic():
    files = [f"data/processed/B0005_cycle_{n}.csv" for n in [2, 4, 6, 8, 10]]
    train_files, val_files = split_cycles(files, parse_cycle_number, val_ratio=0.4)
    assert len(val_files) >= 1
    assert len(train_files) + len(val_files) == len(files)


def test_parse_cycle_number():
    assert parse_cycle_number("data/processed/B0005_cycle_106.csv") == 106
