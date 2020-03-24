#!python

import numpy as np
from PIL import Image
import sys
import math
import random
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
import argparse


#parser =argparse.ArgumentParser(description="Computes Mutual Information between two images")

file1 = sys.argv[1]
file2 = sys.argv[2]
# numpoints = int(sys.argv[3])

im1 = Image.open(file1)
im2 = Image.open(file2)

def to_ids(img):
    ids1 = []
    im1_colors = {}
    im1_unique = 0
    im1_pixels = img.load()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            c = im1_pixels[i, j]
            if c not in im1_colors:
                im1_colors[c] = im1_unique
                ids1.append(im1_unique)
                im1_unique += 1
            else:
                ids1.append(im1_colors[c])
    return ids1

ids1, ids2 = to_ids(im1), to_ids(im2)
print(normalized_mutual_info_score(ids1, ids2))
#print(normalized_mutual_info_score(ids1, ids3))
