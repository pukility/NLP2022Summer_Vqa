import os.path as osp

cfg = {}

# in ./utils/tokenizer.py
cfg["que_path"] = "vqa/questions"
cfg["ans_path"] = "vqa/annotations"
cfg["img_path"] = "vqa/image_features.pkl"
cfg["glove_path"] = "vqa/glove.6B"
cfg["embd_path"] = "vqa/preprocess"
cfg["maxlen"] = 14

# in ./model/question_embedding
cfg["embedding"] = {}
cfg["embedding"]["vocab_size"] = 9448
#cfg["embedding"]["vocab_size"]["train"] = 9447
#cfg["embedding"]["vocab_size"]["val"] = 6835
#cfg["embedding"]["vocab_size"]["tests"] = 6818
cfg["embedding"]["embed_size"] = 300

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
cfg["i_w3"]["output_dim"] = 457

cfg["q_w1"] = {}
cfg["q_w1"]["input_dim"] = 512
cfg["q_w1"]["output_dim"] = 512
cfg["q_w2"] = {}
cfg["q_w2"]["input_dim"] = 512
cfg["q_w2"]["output_dim"] = 300
cfg["q_w3"] = {}
cfg["q_w3"]["input_dim"] = 300
cfg["q_w3"]["output_dim"] = 457

cfg["gru"] = {}
cfg["gru"]["input_size"] = 300
cfg["gru"]["hidden_size"] = 512

cfg["batch_size"] = 10
cfg["epoch_num"] = 5
cfg["learning_rate"] = 0.002
