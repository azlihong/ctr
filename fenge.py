import pandas as pd
chunks=pd.read_csv('./dataProcess/train.csv',header=None,iterator=True)
2chunk=chunks.get_chunk(2000)
chunk.to_csv('./dataProcess/train.csv')
