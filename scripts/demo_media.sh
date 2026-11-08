#!/usr/bin/env bash
# The README's and the documentation's demo (docs/assets/demo.webp),
# cut from the paper's three-camera demo video (zephyr-manuscript's scripts/demo_video.py,
# rendered at 1920x1080): each camera's crop (the picture in picture of its view, the
# snout only) over the scrolling traces, without the full views of the animal.
#
#   bash scripts/demo_media.sh <demo.mp4> [start s]
set -euo pipefail
in=$1
start=${2:-5}
in=$(realpath "$in")
out=$(realpath docs/assets)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
# matplotlib's DejaVu Sans Bold, the demo video's font; copied beside the graph, since a
# drive letter's colon breaks a filter argument on Windows.
cp "$(python -c "import matplotlib, pathlib; print(pathlib.Path(matplotlib.get_data_path(), 'fonts/ttf/DejaVuSans-Bold.ttf'))")" "$work/font.ttf"
graph="$work/graph.txt"
label() { echo "drawtext=fontfile=font.ttf:text='$1':fontcolor=$2:fontsize=46:x=$3-text_w/2:y=14"; }
cat > "$graph" <<EOF
color=c=black:s=1920x960:r=60:d=10[bg];
[0:v]split=4[a][b][c][d];
[a]crop=198:198:34:294,scale=300:300:flags=lanczos[face];
[b]crop=198:198:1059:294,scale=300:300:flags=lanczos[side];
[c]crop=198:198:1294:294,scale=300:300:flags=lanczos[front];
[d]crop=1920:580:0:500[lanes];
[bg][face]overlay=390:70:shortest=1[t1];
[t1][side]overlay=810:70:shortest=1[t2];
[t2][front]overlay=1230:70:shortest=1[t3];
[t3][lanes]overlay=0:380:shortest=1[t4];
[t4]$(label 'Face camera' 0x00E5FF 540),$(label 'Side camera' 0xFF3EA5 960),$(label 'Front camera' 0xA6FF3D 1380),scale=720:360:flags=lanczos,fps=20,trim=duration=8[webp]
EOF
(cd "$work" && ffmpeg -v error -y -ss "$start" -t 10 -i "$in" -/filter_complex graph.txt \
\
  -map "[webp]" -c:v libwebp_anim -quality 35 -compression_level 6 -loop 0 -an "$out/demo.webp")
ls -l "$out/demo.webp"
