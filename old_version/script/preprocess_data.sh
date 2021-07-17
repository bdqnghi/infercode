DATASET=java-small
DATA_DIRECTORY=../java-small-pkl/training
SUBTREE_DIRECTORY=../java-small-subtrees
NODE_TYPE_VOCAB_PATH=../vocab/${DATASET}/type_vocab.csv
NODE_TOKEN_VOCAB_PATH=../vocab/${DATASET}/token_vocab.csv
SUBTREE_VOCAB_PATH=../subtrees_vocab/${DATASET}_subtrees_vocab_train.csv
PYTHON=python3
${PYTHON} preprocess_data.py \
--data_directory ${DATA_DIRECTORY} \
--subtree_vocabulary_path ${SUBTREE_VOCAB_PATH} \
--node_type_vocabulary_path ${NODE_TYPE_VOCAB_PATH} \
--token_vocabulary_path ${NODE_TOKEN_VOCAB_PATH} \
--subtree_directory ${SUBTREE_DIRECTORY} \
--dataset ${DATASET} \
