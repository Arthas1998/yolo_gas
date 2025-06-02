export PYTHONPATH=$(pwd):$PYTHONPATH
python train.py \
--weights D:\\data\\PythonProject\\HITProject\\Code_HIT\\yolov5n.pt \
--cfg D:\\data\\PythonProject\\HITProject\\Code_HIT\\models\\yolov5n.yaml \
--data D:\\data\\PythonProject\\HITProject\\Code_HIT\\data\\gasdata_wwh.yaml \
--hyp D:\\data\\PythonProject\\HITProject\\Code_HIT\\data\\hyps\\hyp.gas-wwh.yaml \
--epochs 25 \
--batch-size 1 \
--device 0 \
--optimizer AdamW \
--workers 16 \
--project D:\\data\\work_dir\\yolo_gas\\ \
--name rep_1 \
--entity wuweihang \
--K 12 \
--save_period 10


--weights D:\\data\\PythonProject\\HITProject\\Code_HIT\\yolov5n.pt
--cfg D:\\data\\PythonProject\\HITProject\\Code_HIT\\models\\yolov5n.yaml
--data D:\\data\\PythonProject\\HITProject\\Code_HIT\\data\\gasdata_wwh.yaml
--hyp D:\\data\\PythonProject\\HITProject\\Code_HIT\\data\\hyps\\hyp.gas-wwh.yaml
--epochs 25
--batch-size 1
--device cpu
--optimizer AdamW
--workers 16
--project D:\\data\\work_dir\\yolo_gas\\
--name rep_1
--entity wuweihang
--K 12
--save_period 10