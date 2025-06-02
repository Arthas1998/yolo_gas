export PYTHONPATH=$(pwd):$PYTHONPATH
python train.py \
--weights='yolov5n.pt' \
--cfg='models/yolov5n.yaml' \
--data='data/gasdata_wwh.yaml' \
--hyp='data/hyps/hyp.gas-wwh.yaml' \
--epochs=25 \
--batch-size=1 \
--device='0' \
--optimizer=AdamW \
--workers=16 \
--project='/data/wuweihang/data/work_dir/yolo_gas/' \
--name='rep_1' \
--entity='wuweihang' \
--K=12 \
--save_period=10