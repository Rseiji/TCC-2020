import pandas as pd
import numpy as np
import bert
import math

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers

## TO DO: UNDER CONSTRUCTION
## Needed to be optimized!
# def compile_results(test_data, df_dataset, y_hat):
#     tokenized_examples = []
#     for data, label in test_data:
#         tokenized_examples += data.numpy().tolist()
# 
#     assert len(y_hat) == len(tokenized_examples)
# 
#     y_hat = [1 if x[0] >= 0.5 else 0 for x in y_hat]
# 
#     for i in range(len(tokenized_examples)):
#         print(i)
#         tokenized_examples[i] = _remove_padding(tokenized_examples[i], i)
# 
#     df_results = _concat_results_to_df(df_dataset, tokenized_examples, y_hat)
#     return df_results
# 
# 
# def _remove_padding(tokenized_example: list, i: int):
#     while tokenized_example[-1] == 0:
#         if len(tokenized_example) > 1:
#             tokenized_example.pop(-1)
#         else:
#             return tokenized_example
#     return tokenized_example
# 
# 
# def _concat_results_to_df(df_dataset, tokenized_examples, predicted_labels):
#     print(len(tokenized_examples))
#     print(len(predicted_labels))
#     df_pred_results = pd.DataFrame({'tokenized_commentary': tokenized_examples,
#                                'label_pred': predicted_labels})
#     for i in range(df_pred_results.shape[0]):
#         print(i)
#         mask = [x == df_pred_results.loc[i, 'tokenized_commentary']
#                                 for x in df_dataset['tokenized_commentary']]
#         df_dataset.loc[mask, 'label_pred'] = df_pred_results.loc[i, 'label_pred']
#     return df_dataset[df_dataset['label_pred'].notnull()]
# 
