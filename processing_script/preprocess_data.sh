MODEL=java-small
TRAINING=0 # 0 for inferring, 1 for testing
INPUT_DATA_DIRECTORY=../OJ_raw_pkl_small
OUTPUT_PATH=../OJ_raw_pkl_small/OJ_raw_pkl_small.pkl
SUBTREE_DIRECTORY=../java-small-subtrees # Matter if TRAINING = 1
NODE_TYPE_VOCAB_PATH=../vocab/type_vocab.csv
NODE_TOKEN_VOCAB_PATH=../vocab/${MODEL}/token_vocab.csv
SUBTREE_VOCAB_PATH=../subtrees_vocab/${MODEL}_subtrees_vocab.csv # Matter if TRAINING = 1
PYTHON=python3
${PYTHON} preprocess_data.py \
--input_data_directory ${INPUT_DATA_DIRECTORY} --output_path ${OUTPUT_PATH} \
--subtree_vocabulary_path ${SUBTREE_VOCAB_PATH} \
--node_type_vocabulary_path ${NODE_TYPE_VOCAB_PATH} \
--token_vocabulary_path ${NODE_TOKEN_VOCAB_PATH} \
--subtree_directory ${SUBTREE_DIRECTORY} \
--model ${MODEL} --training ${TRAINING} \
