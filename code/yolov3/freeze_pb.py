from tensorflow.python.tools import freeze_graph

ckpt_filepath = '../../data/pb/pb.ckpt'
pbtxt_filename = 'model.pbtxt'
pbtxt_filepath = '../../data/pb/model.pbtxt'
pb_filepath = '../../data/pb/freeze.pb'

freeze_graph.freeze_graph(input_graph=pbtxt_filepath, input_saver='', input_binary=False, input_checkpoint=ckpt_filepath, output_node_names='yolov3/yolov3_head/feature_map_1,yolov3/yolov3_head/feature_map_2,yolov3/yolov3_head/feature_map_3', restore_op_name='save/restore_all', filename_tensor_name='save/Const:0', output_graph=pb_filepath, clear_devices=True, initializer_nodes='')
