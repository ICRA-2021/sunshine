import sys
import os
import re

default_pattern = r"/(\d+)_(\d+)_(.+)\.bin"


def cast_numeric(value):
    try:
        return int(value)
    except ValueError:
        return float(value)


def pad(value, length=3):
    return value if len(value) >= length else '0' * (length - len(value)) + value


def change_to_delta(name, pattern=None, starts=None):
    """
    :type name: str
    :type pattern: str
    """
    if pattern is None:
        pattern = default_pattern
    matches = re.findall(pattern, name)
    if len(matches) != 1:
        raise ValueError("Input {} is not uniquely matched by pattern {}".format(name, pattern))
    groups = matches[0]
    assert len(groups) == 3
    if len(groups[1]) != 3:
        name = name.replace("_" + groups[1] + "_", "_" + pad(groups[1], 3) + "_")
        return name, starts, False
    if starts is None:
        starts = {groups[2]: [cast_numeric(m) for m in groups[:-1]]}
    elif groups[2] not in starts:
        starts[groups[2]] = [cast_numeric(m) for m in groups[:-1]]
    # print(name, starts)
    delta_ms = cast_numeric(groups[1]) - starts[groups[2]][1]
    delta_s = cast_numeric(groups[0]) - starts[groups[2]][0]
    if delta_ms < 0:
        delta_ms += 1000
        delta_s -= 1
    if delta_s < 0 or delta_ms < 0:
        raise NameError("Deltas must not be negative!")
    name = name.replace("/" + groups[0] + "_", "/" + pad(str(delta_s), 10) + "_")
    name = name.replace("_" + groups[1] + "_", "_" + pad(str(delta_ms), 3) + "_")
    return name, starts, True


if len(sys.argv) < 2:
    raise UserWarning("Usage: timestamps_to_deltas.py <dir> [pattern]")
elif os.path.isdir(sys.argv[1]):
    while True:
        files = sorted(os.listdir(sys.argv[1]))
        starts = None
        renames = []
        success = True
        for file in files:
            filepath = os.path.join(sys.argv[1], file)
            if not os.path.isfile(filepath):
                raise ValueError("Failed to rename {} which doesn't exist!".format(filepath))
            new_filepath, starts, success = change_to_delta(filepath, starts=starts)
            if os.path.isfile(new_filepath):
                raise ValueError("Refusing to rename {} to {}, which already exists!".format(filepath, new_filepath))
            renames.append((filepath, new_filepath))
            if not success:
                renames = [renames[-1]]
                break
        for old, new in renames:
            os.rename(old, new)
            print(old, new)
        if success:
            break
else:
    raise ValueError(sys.argv[1] + " is not a directory!")
