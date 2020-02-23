import seaborn as sns
import pandas as pd
import numpy as np
import argparse
from csv import DictReader
import os
import ast
from PIL import Image
from scipy.io import loadmat
from scipy.linalg import null_space
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


def save_color_csv(fname, colors):
    with open(fname, "w") as outfile:
        color_writer = writer(outfile, delimiter=",")

        def to_rgb(color):
            return int(color * 255)

        for topic in range(len(colors)):
            color_writer.writerow([topic, to_rgb(colors[topic][0]), to_rgb(colors[topic][1]), to_rgb(colors[topic][2])])

def convert_to_color(image, colors):
    red, green, blue = (
        np.zeros(image.shape, dtype=np.float),
        np.zeros(image.shape, dtype=np.float),
        np.zeros(image.shape, dtype=np.float),
    )
    for j in range(len(colors)):
        red[image == (j + 1)] = colors[j][0] * 255.999
        green[image == (j + 1)] = colors[j][1] * 255.999
        blue[image == (j + 1)] = colors[j][2] * 255.999
    return Image.fromarray((np.dstack((red, green, blue))).astype(np.uint8), mode="RGB")


def compute_image_match_score(left, right):
    mask = np.ones_like(left)
    mask[left == 0] = 0
    mask[right == 0] = 0
    intersection = mask.sum()
    # print(intersection)
    diff = right == left
    # print(diff.min(), diff.max())
    correct = diff[mask > 0].sum()
    # print(intersection - wrong)
    return correct, intersection, correct / float(intersection) * 100


def aggregate_cells(cell_file):
    dists_by_pose = {}
    min_x, max_x = np.inf, -np.inf
    min_y, max_y = np.inf, -np.inf
    with open(cell_file, "r") as infile:
        reader = DictReader(infile)
        for row in reader:
            pose = (int(row["pose_dim_1"]), int(row["pose_dim_2"]))
            if not (-1000 <= pose[0] <= 1000 and -1000 <= pose[1] <= 1000):
                print("Skipping out of range pose " + str(pose))
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
                print("Skipping old pose " + str(pose))
        for key in dists_by_pose.keys():
            dists_by_pose[key][1] /= dists_by_pose[key][1].sum()
    return dists_by_pose, (min_x, max_x), (min_y, max_y)

def cells_to_image(dists_by_pose, range_x, range_y, colors, lifting=None):
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
        assert(float(val[1][k]) > 0.)
        if lifting is not None:
            assert(k < len(lifting))
            k = lifting[k]
        red[x, y] = colors[k][0] * 255.999
        green[x, y] = colors[k][1] * 255.999
        blue[x, y] = colors[k][2] * 255.999
    return Image.fromarray((np.dstack((red, green, blue))).astype(np.uint8), mode="RGB")

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
    return dists_by_pose

def cell_distance(dists_by_pose_a, dists_by_pose_b, metric=None):
    if metric is None:
        metric = lambda x, y: np.sum(np.abs(y - x))
    distance = 0.
    size = 0
    for k, v in dists_by_pose_a.items():
        if k in dists_by_pose_b:
            distance += metric(v[1], dists_by_pose_b[k][1])
            size += 1
    return distance / size

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
        full_img = cells_to_image(*aggregate, colors)
        full_img.save(args.output_dir + "/{}reference-img.png".format(args.name + "-" if args.name is not None else ""))
        min_x, max_x = aggregate[1]
        min_y, max_y = aggregate[2]
    else:
        min_x, max_x, min_y, max_y = np.inf, -np.inf, np.inf, -np.inf

    aggregates = []
    liftings = []
    for file in lifting_dict.keys():
        cells_file = file.replace("modelweights.bin", "cells.csv")
        if not os.path.isfile(cells_file):
            raise ValueError("Failed to find " + cells_file)
        aggregates.append(aggregate_cells(cells_file))
        min_x = min(min_x, aggregates[-1][1][0])
        min_y = min(min_y, aggregates[-1][2][0])
        max_x = max(max_x, aggregates[-1][1][1])
        max_y = max(max_y, aggregates[-1][2][1])
        liftings.append(lifting_dict[file])
        img = cells_to_image(aggregates[-1][0], (min_x, max_x), (min_y, max_y), colors)
        unaligned_file = file.replace("modelweights.bin", "unaligned-img.png")
        img.save(unaligned_file)
        aligned_img = cells_to_image(aggregates[-1][0], (min_x, max_x), (min_y, max_y), colors, lifting_dict[file])
        aligned_file = file.replace("modelweights.bin", "aligned-img.png")
        aligned_img.save(aligned_file)

    merged_cells = merge_cells([aggregate[0] for aggregate in aggregates], liftings)
    aligned_img = cells_to_image(merged_cells, (min_x, max_x), (min_y, max_y), colors)
    aligned_img.save(args.output_dir + "/{}merged-img.png".format(args.name + "-" if args.name is not None else ""))


    naive_cells = aggregates[0][0]
    for a in aggregates[1:]:
        naive_cells.update(a[0])
    aligned_img = cells_to_image(naive_cells, (min_x, max_x), (min_y, max_y), colors)
    aligned_img.save(args.output_dir + "/{}naive-merged-img.png".format(args.name + "-" if args.name is not None else ""))

    if args.full is not None:
        print("Merged distance: {}".format(cell_distance(merged_cells, aggregate[0])))
        print("Naive distance: {}".format(cell_distance(naive_cells, aggregate[0])))
