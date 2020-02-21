import seaborn as sns
import pandas as pd
import numpy as np
import cv2
import argparse
from csv import writer
import os
from PIL import Image
from scipy.io import loadmat
from scipy.linalg import null_space
from itertools import combinations


def parse_args():
    parser = argparse.ArgumentParser(
        prog="extract_path_csv.py",
        description="Extracts the last path message from a rosbag topic and converts it to a CSV.",
    )

    def Box(arg):
        return [int(d) for d in arg.split("x")]

    parser.add_argument("mat", type=str, help="mat file with permutation correspondences", metavar="mat")
    parser.add_argument("inputs", type=str, nargs="+", help="input files", metavar="inputs...")
    parser.add_argument(
        "--warmup", type=int, default=10, help="For timeseries data, ignore this many rows from the beginning"
    )
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--crop", default=None, type=Box)
    args = parser.parse_args()
    return args


def save_color_csv(fname, colors):
    with open(fname, "w") as outfile:
        color_writer = writer(outfile, delimiter=",")

        def to_rgb(color):
            return int(color * 255)

        for topic in range(len(colors)):
            color_writer.writerow([topic, to_rgb(colors[topic][0]), to_rgb(colors[topic][1]), to_rgb(colors[topic][2])])


def process_timeseries(args, colors, match_dict):
    for i, timeseries in enumerate(args.inputs):
        fname = "%s/output%d-colors.csv" % (args.output_dir, i)
        save_color_csv(fname, colors)
        # copyfile(timeseries, "%s/run%d-timeseries.csv" % (args.output_dir, i))
        fname = "%s/output%d-aligned-colors.csv" % (args.output_dir, i)
        save_color_csv(fname, colors)

    # print("Parsing paths")
    num_unique_topics = len(colors)
    timeseries_data = [pd.read_csv(timeseries) for timeseries in args.inputs]
    topic_id = 1
    for i, timeseries_df in enumerate(timeseries_data):
        timeseries_df.iloc[args.warmup :].to_csv("%s/output%d-timeseries.csv" % (args.output_dir, i), index=False)
        aligned_df = timeseries_df.copy(deep=True)  # type: pd.DataFrame
        aligned_df["time"] = timeseries_df["time"]
        for column_name in ["topic_%d_count" % id for id in range(num_unique_topics)]:
            aligned_df[column_name] = 0
        for col in range(1, len(timeseries_df.columns)):
            matched_col = match_dict[topic_id] - 1
            column_name = "topic_%d_count" % matched_col
            aligned_df[column_name] = timeseries_df[timeseries_df.columns[col]]
            topic_id += 1
        aligned_df.iloc[args.warmup :].to_csv("%s/output%d-aligned-timeseries.csv" % (args.output_dir, i), index=False)
        # print(aligned_df)


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


def process_maps(args, colors, match_dict):
    map_images = [cv2.imread(image_file, flags=cv2.IMREAD_UNCHANGED).astype(np.int) for image_file in args.inputs]
    if args.crop is not None:
        x, y, w, h = args.crop
        map_images = [image[y : y + h, x : x + w] for image in map_images]
    merged_images = []
    img_names = []
    merged_names = []
    topic_id = 1
    for i, image in enumerate(map_images):
        img_names.append("%s/output%d-color-topics.png" % (args.output_dir, i))
        convert_to_color(image, colors).save(img_names[-1], format="PNG")
        merged_image = image.copy()
        for j in range(20):  # TODO: Fix this to get the number of topics dynamically (perhaps from corresponding CSV?)
            merged_image[image == (j + 1)] = match_dict[topic_id]
            topic_id += 1
        merged_images.append(merged_image)
        merged_names.append("%s/output%d-corrected-color-topics.png" % (args.output_dir, i))
        convert_to_color(merged_image, colors).save(merged_names[-1], format="PNG")

    match_results = []
    for left, right in combinations(range(len(map_images)), 2):
        init_correct, init_match, init_score = compute_image_match_score(map_images[left], map_images[right])
        final_correct, final_match, final_score = compute_image_match_score(merged_images[left], merged_images[right])
        print("Initial matches: %d / %d (%.1f%%)" % (init_correct, init_match, init_score))
        print("Result matches:  %d / %d (%.1f%%)" % (final_correct, final_match, final_score))
        match_results.append(
            {
                "Left": img_names[left],
                "Right": img_names[right],
                "Left-Merged": merged_names[left],
                "Right-Merged": merged_names[right],
                "Overlap": init_match,
                "Initial Score": init_score,
                "Final Score": final_score,
            }
        )
    pd.DataFrame(match_results).to_csv("%s/match-scores.csv" % args.output_dir, index=False)


if __name__ == "__main__":
    args = parse_args()
    try:
        os.makedirs(args.output_dir)
    except OSError:
        pass
    matches = loadmat(args.mat)["Pout"]
    match_dict = {}
    id_count = 1
    for row in range(1, 1 + matches.shape[0]):
        for col in range(1, row):
            if matches[row - 1, col - 1] > 0:
                match_dict[row] = col
                break
        else:
            match_dict[row] = id_count
            id_count += 1

    # print(matches)
    num_unique_topics = matches.shape[0] - null_space(matches).shape[1]
    # print(num_unique_topics, "unique topics (expected", id_count - 1, ")")
    assert num_unique_topics == (id_count - 1)
    colors = sns.color_palette("Set3", n_colors=num_unique_topics)
    if args.inputs[0].endswith("csv"):
        process_timeseries(args, colors, match_dict)
    elif args.inputs[0].endswith("png"):
        process_maps(args, colors, match_dict)
