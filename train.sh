export PYTHONPATH=$(pwd):$PYTHONPATH
python train.py \
--weights /storage_server/disk5/wuweihang/project/yolo_gas/yolov5n.pt \
--cfg /storage_server/disk5/wuweihang/project/yolo_gas/models/yolov5n.yaml \
--data /storage_server/disk5/wuweihang/project/yolo_gas/data/gasdata_wwh.yaml \
--hyp /storage_server/disk5/wuweihang/project/yolo_gas/data/hyps/hyp.gas-wwh.yaml \
--epochs 40 \
--batch-size 128 \
--device 2 \
--optimizer AdamW \
--workers 8 \
--project /storage_server/disk5/wuweihang/work_dir/yolo_gas \
--name rep_cos_adamw \
--entity wuweihang \
--cos-lr \
--K 12 \
--save_period 10