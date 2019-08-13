#!/bin/bash
ffmpeg -framerate 10 -pattern_type glob -i 'sentry-topics-*-timeseries.png' -vcodec libx264 -preset slow -crf 20 timeseries.mp4
ffmpeg -framerate 10 -pattern_type glob -i 'sentry-topics-*-topics.png' -vf scale=672:658:flags=neighbor -vcodec libx264 -pix_fmt yuv420p -crf 23 topics.mp4
ffmpeg -framerate 10 -pattern_type glob -i 'sentry-topics-*-images.png' -vcodec libx264 -preset slow -crf 20 images.mp4
ffmpeg -i images.mp4 -i topics.mp4 -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' -map [vid] -c:v libx264 -crf 23 -preset slow images-topics.mp4
ffmpeg -i timeseries.mp4 -i images-topics.mp4 -filter_complex '[0:v]scale=2004:-1[v0];[1:v][v0]vstack=inputs=2' complete.mp4
