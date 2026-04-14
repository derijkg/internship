# seperate terminals

#calc12
CUDA_VISIBLE_DEVICES=0 nohup python src/marker_distributed.py --total_workers 3 --worker_index 0 > worker0.log 2>&1 &

#calc11
# This targets the FIRST GPU (index 0)
CUDA_VISIBLE_DEVICES=0 nohup python src/marker_distributed.py --total_workers 3 --worker_index 1 > worker1.log 2>&1 &

# This targets the SECOND GPU (index 1)
CUDA_VISIBLE_DEVICES=0 nohup python src/marker_distributed.py --total_workers 3 --worker_index 2 > worker2.log 2>&1 &


# final zip
# Optional: merge zips later
zip -g output_marker_part_0.zip output_marker_part_1.zip output_marker_part_2.zip