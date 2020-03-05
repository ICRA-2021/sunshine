import seaborn as sns
import pandas as pd
import numpy as np
import argparse
from csv import DictReader
import os
from math import ceil, sqrt
import ast
from PIL import Image
from scipy.optimize import linear_sum_assignment
from itertools import combinations


def parse_args():
    parser = argparse.ArgumentParser(
        prog="extract_path_csv.py",
        description="Extracts the last path message from a rosbag topic and converts it to a CSV.",
    )

    parser.add_argument("lifting", type=str, help="mat file with permutation correspondences", metavar="mat")
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--full", default=None, type=str)
    parser.add_argument("--name", default=None, type=str)
    args = parser.parse_args()
    return args


def aggregate_cells(cell_file):
    dists_by_pose = {}
    min_x, max_x = np.inf, -np.inf
    min_y, max_y = np.inf, -np.inf
    outofrange_count = 0
    old_count = 0
    with open(cell_file, "r") as infile:
        reader = DictReader(infile)
        for row in reader:
            pose = (int(row["pose_dim_1"]), int(row["pose_dim_2"]))
            if not (-1000 <= pose[0] <= 1000 and -1000 <= pose[1] <= 1000):
                outofrange_count += 1
                continue
            min_x, max_x = min(min_x, pose[0]), max(max_x, pose[0])
            min_y, max_y = min(min_y, pose[1]), max(max_y, pose[1])
            dist = np.array([int(row[k]) for k in sorted(row.keys()) if "k_" in k])
            dist = np.divide(dist, dist.sum())
            if pose in dists_by_pose and int(row["pose_dim_0"]) == dists_by_pose[pose][0]:
                dists_by_pose[pose][1] += dist
            elif pose not in dists_by_pose or int(row["pose_dim_0"]) > dists_by_pose[pose][0]:
                dists_by_pose[pose] = [int(row["pose_dim_0"]), dist]
            else:
                old_count += 1
        for key in dists_by_pose.keys():
            dists_by_pose[key][1] /= dists_by_pose[key][1].sum()
    print("Skipped {} out of range cells and {} outdated cells".format(outofrange_count, old_count))
    return dists_by_pose, (min_x, max_x), (min_y, max_y)


def cells_to_image(dists_by_pose, range_x, range_y, colors, lifting=None, layers=False, filename=None):
    sx = range_x[1] - range_x[0] + 1
    sy = range_y[1] - range_y[0] + 1
    red, green, blue = (
        np.zeros((sx, sy), dtype=np.float),
        np.zeros((sx, sy), dtype=np.float),
        np.zeros((sx, sy), dtype=np.float),
    )

    for key, val in dists_by_pose.items():
        x = key[0] - range_x[0]
        y = key[1] - range_y[0]
        # print(x, y, sx, sy)
        assert 0 <= x < sx and 0 <= y < sy

        k = np.argmax(val[1])
        assert float(val[1][k]) > 0.0
        if lifting is not None:
            assert k < len(lifting)
            k = lifting[k]
        red[x, y] = colors[k][0] * 255.999
        green[x, y] = colors[k][1] * 255.999
        blue[x, y] = colors[k][2] * 255.999
    img = Image.fromarray((np.dstack((red, green, blue))).astype(np.uint8), mode="RGB")
    if filename is not None:
        img.save(filename)
    if layers:
        layers_img = cells_to_layers(
            dists_by_pose,
            range_x,
            range_y,
            colors,
            lifting=lifting,
            filename=filename.replace(os.path.splitext(filename)[0], os.path.splitext(filename)[0] + "-layers"),
        )
        return img, layers_img
    return img


def cells_to_layers(dists_by_pose, range_x, range_y, colors, lifting=None, filename=None):
    sx = range_x[1] - range_x[0] + 1
    sy = range_y[1] - range_y[0] + 1
    red, green, blue = (
        np.zeros((len(colors), sx, sy), dtype=np.float),
        np.zeros((len(colors), sx, sy), dtype=np.float),
        np.zeros((len(colors), sx, sy), dtype=np.float),
    )

    if lifting is not None:
        scaling = [lifting.count(k) for k in range(len(colors))]
    else:
        scaling = [1 for k in range(len(colors))]
    for key, val in dists_by_pose.items():
        x = key[0] - range_x[0]
        y = key[1] - range_y[0]
        # print(x, y, sx, sy)
        assert 0 <= x < sx and 0 <= y < sy

        for k in range(len(colors)):
            if lifting is None:
                ku = k
            else:
                assert k < len(lifting)
                ku = lifting[k]
            assert 1 >= float(val[1][ku]) >= 0.0
            assert scaling[ku] >= 1
            intensity = val[1][ku]
            red[ku, x, y] = colors[ku][0] * 255.999 * intensity / scaling[ku]
            green[ku, x, y] = colors[ku][1] * 255.999 * intensity / scaling[ku]
            blue[ku, x, y] = colors[ku][2] * 255.999 * intensity / scaling[ku]
    canvas_width = ceil(sqrt(len(colors))) * (sx + 1) - 1
    canvas_height = ceil(sqrt(len(colors))) * (sy + 1) - 1
    canvas = np.ones((canvas_width, canvas_height, 3), dtype=np.uint8) * 255
    for k in range(len(colors)):
        canvas_x = k % ceil(sqrt(len(colors)))
        canvas_y = k // ceil(sqrt(len(colors)))
        canvas[
            canvas_x * (sx + 1) : (canvas_x + 1) * (sx + 1) - 1, canvas_y * (sy + 1) : (canvas_y + 1) * (sy + 1) - 1, :
        ] = np.dstack((red[k, :, :], green[k, :, :], blue[k, :, :]))
    img = Image.fromarray(canvas, mode="RGB")
    if filename is not None:
        img.save(filename)
    return img


def permute(dist, lifting):
    out_dist = np.zeros((max(len(dist), max(lifting)),), dist.dtype)
    for i in range(len(dist)):
        out_dist[lifting[i]] += dist[i]
    return out_dist


def merge_cells(dists_by_pose_list, liftings):
    dists_by_pose = {}
    for i in range(len(dists_by_pose_list)):
        for key, val in dists_by_pose_list[i].items():
            if key in dists_by_pose:
                dists_by_pose[key][1] += permute(val[1], liftings[i])
            else:
                dists_by_pose[key] = [val[0], permute(val[1], liftings[i])]
    for key in dists_by_pose.keys():
        dists_by_pose[key][1] /= dists_by_pose[key][1].sum()
    return dists_by_pose


def cell_distance(dists_by_pose_a, dists_by_pose_b, metric=None):
    if metric is None:
        metric = lambda x, y: np.sum(np.abs(y - x))
    distance = 0.0
    size = 0
    for k, v in dists_by_pose_a.items():
        if k in dists_by_pose_b:
            distance += metric(v[1], dists_by_pose_b[k][1])
            size += 1
    return distance / size


def match_distance(dists_by_pose_a, dists_by_pose_b):
    assert len(dists_by_pose_a) > 0 and len(dists_by_pose_b) > 0
    dists = None
    total = 0
    for k1, v1 in dists_by_pose_a.items():
        if k1 in dists_by_pose_b:
            total = total + 1
            v2 = dists_by_pose_b[k1]
            if dists is None:
                dists = np.zeros((v1[1].size, v2[1].size))
            dists += np.abs(v1[1].reshape((-1, 1)) - v2[1].reshape((1, -1)))
    rows, cols = linear_sum_assignment(dists)
    with np.printoptions(precision=3, linewidth=200):
        print(dists)
        print(dists[rows, cols])
    return dists[rows, cols].sum() / (2*total), total


if __name__ == "__main__":
    args = parse_args()
    try:
        os.makedirs(args.output_dir)
    except OSError:
        pass

    K = 0
    lifting_dict = {}
    with open(args.lifting, "r") as infile:
        reader = DictReader(infile)
        for row in reader:
            lifting_dict[row["file"]] = ast.literal_eval(row["lifting"])
            K = max(K, len(lifting_dict[row["file"]]))

    colors = sns.color_palette("Set3", n_colors=K)
    if args.full is not None:
        aggregate = aggregate_cells(args.full)
        full_img = cells_to_image(
            *aggregate,
            colors,
            layers=True,
            filename=args.output_dir + "/{}reference-img.png".format(args.name + "-" if args.name is not None else "")
        )
        partial_match_results = []
        min_x, max_x = aggregate[1]
        min_y, max_y = aggregate[2]
    else:
        min_x, max_x, min_y, max_y = np.inf, -np.inf, np.inf, -np.inf
        partial_match_results = None
        aggregate = None

    aggregates = []
    liftings = []
    for file in lifting_dict.keys():
        cells_file = file.replace("modelweights.bin", "cells.csv")
        if not os.path.isfile(cells_file):
            raise ValueError("Failed to find " + cells_file)
        aggregates.append(aggregate_cells(cells_file))
        if args.full is not None:
            partial_match_results.append(match_distance(aggregates[-1][0], aggregate[0]))
        min_x = min(min_x, aggregates[-1][1][0])
        min_y = min(min_y, aggregates[-1][2][0])
        max_x = max(max_x, aggregates[-1][1][1])
        max_y = max(max_y, aggregates[-1][2][1])
        liftings.append(lifting_dict[file])
        img = cells_to_image(
            aggregates[-1][0],
            (min_x, max_x),
            (min_y, max_y),
            colors,
            layers=False,
            filename=file.replace("modelweights.bin", "unaligned-img.png"),
        )
        aligned_img = cells_to_image(
            aggregates[-1][0],
            (min_x, max_x),
            (min_y, max_y),
            colors,
            lifting_dict[file],
            layers=False,
            filename=file.replace("modelweights.bin", "aligned-img.png"),
        )

    merged_cells = merge_cells([a[0] for a in aggregates], liftings)
    aligned_img = cells_to_image(
        merged_cells,
        (min_x, max_x),
        (min_y, max_y),
        colors,
        layers=True,
        filename=args.output_dir + "/{}merged-img.png".format(args.name + "-" if args.name is not None else ""),
    )

    naive_cells = aggregates[0][0]
    for a in aggregates[1:]:
        naive_cells.update(a[0])
    aligned_img = cells_to_image(
        naive_cells,
        (min_x, max_x),
        (min_y, max_y),
        colors,
        layers=True,
        filename=args.output_dir + "/{}naive-merged-img.png".format(args.name + "-" if args.name is not None else ""),
    )

    if args.full is not None:
        print("Merged distance: {}".format(match_distance(merged_cells, aggregate[0])[0]))
        print("Naive distance: {}".format(match_distance(naive_cells, aggregate[0])[0]))
        print("Min Distance: {}".format(sum([a[0] * a[1] for a in partial_match_results]) / sum([a[1] for a in partial_match_results])))
