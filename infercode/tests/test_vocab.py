import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
from data_utils.vocabulary import Vocabulary
import os
from dpu_utils.codeutils import identifiersplitting


vocab = Vocabulary(1000000, "../sentencepiece_vocab/tokens/universal_token_subword.model")
# vocab.create_vocabulary(tokens=data, model_filename="universal_subtrees", model_type="word")

text = "void bubbleSort(int arr[], int n) { int i, j; for (i = 0; i < n-1; i++) // Last i elements are already in place for (j = 0; j < n-i-1; j++) if (arr[j] > arr[j+1]) swap(&arr[j], &arr[j+1]); }"
text = " ".join(identifiersplitting.split_identifier_into_parts(text))
a = vocab.get_id_or_unk_for_text(text)
print(a)

b = vocab.tokenize(text)
print(b)

# y = vocab.get_id_or_unk_for_text("do_statement")
# print(vocab.get_vocabulary())
