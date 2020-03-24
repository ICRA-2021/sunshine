import os
import sys

ref_files = os.listdir(sys.argv[1])
for filename in os.listdir(sys.argv[1]):
    if filename.endswith("images.png"):
        if filename.replace("images", "topics") not in ref_files:
            os.rename(filename, filename.replace("images", "extra"))
