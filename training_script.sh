PARSER=treesitter
TRAIN_TREE_PATH=java-small-test/trees-training.pkl
TRAIN_BUCKET_PATH=java-small-test/buckets-all-training.pkl
NODE_TYPE_VOCABULARY_PATH=vocab/${PARSER}/node_type/node_types_c_java_cpp_c-sharp_rust.txt
TOKEN_VOCABULARY_PATH=vocab/${PARSER}/node_token/token.txt
SUBTREE_VOCABULARY_PATH=vocab/${PARSER}/node_token/token.txt
BATCH_SIZE=2
CHECKPOINT_EVERY=500
TREE_SIZE_THRESHOLD_UPPER=1500
TREE_SIZE_THRESHOLD_LOWER=0
CUDA=-1
VALIDATING=1
NODE_TYPE_DIM=100
NODE_TOKEN_DIM=100
CONV_OUTPUT_DIM=100
NUM_CONV=2
EPOCH=120
PYTHON=python3
NODE_INIT=2
BEST_F1=0.0
${PYTHON} training_script.py \
--train_tree_path ${TRAIN_TREE_PATH} --train_bucket_path ${TRAIN_BUCKET_PATH} --batch_size ${BATCH_SIZE} \
--checkpoint_every ${CHECKPOINT_EVERY} --cuda ${CUDA} --validating ${VALIDATING} \
--tree_size_threshold_upper ${TREE_SIZE_THRESHOLD_UPPER} \
--tree_size_threshold_lower ${TREE_SIZE_THRESHOLD_LOWER} \
--node_type_dim ${NODE_TYPE_DIM} --node_token_dim ${NODE_TOKEN_DIM} \
--node_type_vocabulary_path ${NODE_TYPE_VOCABULARY_PATH} \
--token_vocabulary_path ${TOKEN_VOCABULARY_PATH} --subtree_vocabulary_path ${SUBTREE_VOCABULARY_PATH} \
--epochs ${EPOCH} --parser ${PARSER} \
--node_init ${NODE_INIT} --num_conv ${NUM_CONV} --conv_output_dim ${CONV_OUTPUT_DIM} \
--best_f1 ${BEST_F1}