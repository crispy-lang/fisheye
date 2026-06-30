#!/usr/bin/env bash
set -u

PROJECT_DIR="${PROJECT_DIR:-/mnt/hgfs/lab_final/PRJ3_fisheye_depth_to_normal_fixed/prj3_fixed}"
DATASET_DIR="${DATASET_DIR:-/mnt/hgfs/lab_final/fisheye_normal_dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/output/report_all}"
BIN="${BIN:-$PROJECT_DIR/build/fisheye_normal}"

CSV="$OUTPUT_ROOT/run_all_metrics.csv"
FAILED="$OUTPUT_ROOT/failed.txt"
SUMMARY="$OUTPUT_ROOT/scene_summary.csv"

mkdir -p "$OUTPUT_ROOT"
: > "$FAILED"

export OPENCV_IO_ENABLE_OPENEXR=1

if [[ ! -x "$BIN" ]]; then
  echo "[ERROR] executable not found: $BIN"
  echo "Build first:"
  echo "  cd \"$PROJECT_DIR\""
  echo "  cmake -S . -B build"
  echo "  cmake --build build -j"
  exit 1
fi

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "[ERROR] dataset not found: $DATASET_DIR"
  exit 1
fi

if [[ ! -f "$CSV" ]]; then
  echo "scene,frame,valid_count,mean,median,rmse,p11_25,p22_5,p30" > "$CSV"
fi

total=$(find "$DATASET_DIR" -path "*/camera_l_depth/*.exr" -type f | wc -l)
done_count=0
run_count=0
skip_count=0
fail_count=0

echo "============================================================"
echo "Full dataset fisheye normal evaluation"
echo "Project : $PROJECT_DIR"
echo "Dataset : $DATASET_DIR"
echo "Output  : $OUTPUT_ROOT"
echo "Total   : $total images"
echo "============================================================"

while IFS= read -r -d '' depth_file; do
  scene="$(basename "$(dirname "$(dirname "$depth_file")")")"
  base="$(basename "$depth_file")"
  frame="${base#camera_l_depth_}"
  frame="${frame%.exr}"
  gt_file="$DATASET_DIR/$scene/camera_l_normal/camera_l_normal_${frame}.npy"
  out_dir="$OUTPUT_ROOT/$scene/$frame"
  metrics="$out_dir/metrics.txt"
  log_file="$out_dir/run.log"

  done_count=$((done_count + 1))

  if [[ ! -f "$gt_file" ]]; then
    echo "[MISS GT] $scene/$frame"
    echo "$scene/$frame missing_gt" >> "$FAILED"
    fail_count=$((fail_count + 1))
    continue
  fi

  if [[ -f "$metrics" ]]; then
    echo "[$done_count/$total] SKIP $scene/$frame"
    skip_count=$((skip_count + 1))
    continue
  fi

  mkdir -p "$out_dir"
  echo "[$done_count/$total] RUN  $scene/$frame"

  "$BIN" \
    --input="$depth_file" \
    --gt="$gt_file" \
    --gt-format=xyz \
    --output="$out_dir" \
    --f=254.6479 \
    --cx=399.5 \
    --cy=399.5 \
    --depth-mode=ray \
    --depth-scale=1.0 \
    --max-theta=90 \
    --valid-radius=399 \
    --normal-orientation=away \
    > "$log_file" 2>&1

  status=$?
  if [[ $status -ne 0 || ! -f "$metrics" ]]; then
    echo "[FAIL] $scene/$frame, see $log_file"
    echo "$scene/$frame status=$status" >> "$FAILED"
    fail_count=$((fail_count + 1))
    continue
  fi

  valid=$(awk -F= '/valid_count/ {print $2}' "$metrics")
  mean=$(awk -F= '/mean_angular_error_deg/ {print $2}' "$metrics")
  median=$(awk -F= '/median_angular_error_deg/ {print $2}' "$metrics")
  rmse=$(awk -F= '/rmse_angular_error_deg/ {print $2}' "$metrics")
  p11=$(awk -F= '/percent_below_11.25_deg/ {print $2}' "$metrics")
  p22=$(awk -F= '/percent_below_22.5_deg/ {print $2}' "$metrics")
  p30=$(awk -F= '/percent_below_30_deg/ {print $2}' "$metrics")

  echo "$scene,$frame,$valid,$mean,$median,$rmse,$p11,$p22,$p30" >> "$CSV"
  run_count=$((run_count + 1))
done < <(find "$DATASET_DIR" -path "*/camera_l_depth/*.exr" -type f -print0 | sort -z)

awk -F, '
NR > 1 {
  scene=$1
  valid=$3 + 0
  count[scene] += 1
  valid_sum[scene] += valid
  mean_sum[scene] += ($4 + 0) * valid
  median_sum[scene] += ($5 + 0)
  rmse_sum[scene] += ($6 + 0) * valid
  p11_sum[scene] += ($7 + 0) * valid
  p22_sum[scene] += ($8 + 0) * valid
  p30_sum[scene] += ($9 + 0) * valid
  total_count += 1
  total_valid += valid
  total_mean += ($4 + 0) * valid
  total_median += ($5 + 0)
  total_rmse += ($6 + 0) * valid
  total_p11 += ($7 + 0) * valid
  total_p22 += ($8 + 0) * valid
  total_p30 += ($9 + 0) * valid
}
END {
  print "scene,count,valid_count,weighted_mean,avg_median,weighted_rmse,weighted_p11_25,weighted_p22_5,weighted_p30"
  for (scene in count) {
    printf "%s,%d,%.0f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
      scene, count[scene], valid_sum[scene],
      mean_sum[scene]/valid_sum[scene],
      median_sum[scene]/count[scene],
      rmse_sum[scene]/valid_sum[scene],
      p11_sum[scene]/valid_sum[scene],
      p22_sum[scene]/valid_sum[scene],
      p30_sum[scene]/valid_sum[scene]
  }
  if (total_count > 0) {
    printf "Overall,%d,%.0f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
      total_count, total_valid,
      total_mean/total_valid,
      total_median/total_count,
      total_rmse/total_valid,
      total_p11/total_valid,
      total_p22/total_valid,
      total_p30/total_valid
  }
}' "$CSV" > "$SUMMARY"

echo "============================================================"
echo "Finished."
echo "Ran     : $run_count"
echo "Skipped : $skip_count"
echo "Failed  : $fail_count"
echo "CSV     : $CSV"
echo "Summary : $SUMMARY"
echo "Failed list: $FAILED"
echo "============================================================"
