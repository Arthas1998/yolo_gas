import onnx

model1 = onnx.load('last.onnx')
model2 = onnx.load('yolov5n_qa_infer_baseline.onnx')

# 比较模型的图结构
graph1 = model1.graph
graph2 = model2.graph

# 比较节点
nodes1 = {node.name: node for node in graph1.node}
nodes2 = {node.name: node for node in graph2.node}

common_nodes = set(nodes1.keys()) & set(nodes2.keys())
diff_nodes = set(nodes1.keys()) ^ set(nodes2.keys())

print(f"Common nodes: {len(common_nodes)}")
print(f"Different nodes: {len(diff_nodes)}")

for node in diff_nodes:
    if node in nodes1:
        print(f"Node '{node}' only exists in Model 1")
    else:
        print(f"Node '{node}' only exists in Model 2")