def split_cycles(files, parse_cycle_number_fn, val_ratio=0.2):
    cycles = [parse_cycle_number_fn(p) for p in files]
    unique = sorted([c for c in cycles if c is not None])
    if not unique:
        return files, []

    val_count = max(1, int(len(unique) * val_ratio))
    val_cycles = set(unique[-val_count:])

    train_files = [p for p in files if parse_cycle_number_fn(p) not in val_cycles]
    val_files = [p for p in files if parse_cycle_number_fn(p) in val_cycles]
    return train_files, val_files
