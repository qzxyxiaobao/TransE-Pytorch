# TransE-Pytorch
An implementation of TransE[1] in Pytorch.

Test results with default parameter:

        -----Result of Link Prediction (Raw)-----
      |  Mean Rank  |  Filter@10  |
      |  tensor(353.5773, device='cuda:0')  |  0.101488039816  |
        -----Result of Link Prediction (Filter)-----
      |  Mean Rank  |  Filter@10  |
      |  tensor(276.4282, device='cuda:0')  |  0.171166900848  |

Better performance can be achieved by tunning the parameters.

[1] Bordes A, Usunier N, Garcia-Duran A, et al. Translating embeddings for modeling multi-relational data[C]//Advances in neural information processing systems. 2013: 2787-2795.
