cfg = {}

cfg["embedding"] = {}

cfg["embedding"]["vocab_size"] = {}
cfg["embedding"]["vocab_size"]["train"] = 9447
cfg["embedding"]["vocab_size"]["val"] = 6835
cfg["embedding"]["vocab_size"]["tests"] = 6818

cfg["w1"] = {}
cfg["w1"]["input_dim"] = 512 + 2048
cfg["w1"]["output_dim"] = 512

cfg["image_attention"] = {}
cfg["image_attention"]["w2_input_size"] = 512
cfg["image_attention"]["w2_output_size"] = 1

cfg["i_w1"] = {}
cfg["i_w1"]["input_dim"] = 2048
cfg["i_w1"]["output_dim"] = 512
cfg["i_w2"] = {}
cfg["i_w2"]["input_dim"] = 512
cfg["i_w2"]["output_dim"] = 2048
cfg["i_w3"] = {}
cfg["i_w3"]["input_dim"] = 2048
cfg["i_w3"]["output_dim"] = 3129

cfg["q_w1"] = {}
cfg["q_w1"]["input_dim"] = 512
cfg["q_w1"]["output_dim"] = 512
cfg["q_w2"] = {}
cfg["q_w2"]["input_dim"] = 512
cfg["q_w2"]["output_dim"] = 300
cfg["q_w3"] = {}
cfg["q_w3"]["input_dim"] = 300
cfg["q_w3"]["output_dim"] = 3129

cfg["gru"] = {}
cfg["gru"]["input_size"] = 14 * 300
cfg["gru"]["hidden_sizes"] = 512

cfg["epoch_num"] = 10
cfg["learning_rate"] = 0.1
cfg["momentum"] = 0.9