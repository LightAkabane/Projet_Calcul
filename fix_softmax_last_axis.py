# fix_softmax_last_axis.py
import onnx
from onnx import helper, shape_inference, checker

def move_axis_last_perm(rank, axis):
    a = axis if axis >= 0 else rank + axis
    return [i for i in range(rank) if i != a] + [a]

def invert_perm(perm):
    inv = [0]*len(perm)
    for i,p in enumerate(perm):
        inv[p] = i
    return inv

model_in  = "yolov8m.onnx"          # <- ton fichier exporté
model_out = "yolov8m_webgpu.onnx"   # <- sortie patchée

print(f"Loading {model_in} ...")
model = onnx.load(model_in)

print("Inferring shapes ...")
model = shape_inference.infer_shapes(model)
g = model.graph
vis = {vi.name: vi for vi in list(g.value_info) + list(g.input) + list(g.output)}

new_nodes = []
patched = 0

for n in g.node:
    if n.op_type == "Softmax":
        axis_attr = next((a for a in n.attribute if a.name == "axis"), None)
        axis = axis_attr.i if axis_attr else 1  # défaut ONNX = 1
        if axis != -1:
            inp, out = n.input[0], n.output[0]
            vi = vis.get(inp)
            if not vi or not vi.type.tensor_type.shape.dim:
                # pas d'info de rang -> on ne modifie pas
                new_nodes.append(n)
                continue
            rank = len(vi.type.tensor_type.shape.dim)
            perm = move_axis_last_perm(rank, axis)
            inv = invert_perm(perm)

            pre  = helper.make_node("Transpose", [inp], [inp+"_to_last"], perm=perm, name=n.name+"_preT")
            smx  = helper.make_node("Softmax",   [inp+"_to_last"], [out+"_soft"], axis=-1, name=n.name+"_axis-1")
            post = helper.make_node("Transpose", [out+"_soft"], [out], perm=inv, name=n.name+"_postT")

            new_nodes += [pre, smx, post]
            patched += 1
            continue
    new_nodes.append(n)

del g.node[:]
g.node.extend(new_nodes)

print(f"Patched Softmax nodes: {patched}")
checker.check_model(model)
onnx.save(model, model_out)
print(f"Saved -> {model_out}")
