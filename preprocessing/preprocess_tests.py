import pandas as pd
from testbook import testbook


@testbook('preprocessing.ipynb', execute=True)
def test_func(tb):
   func = tb.get("filter_negative_price")

   events_df = pd.DataFrame()
   events_df['price'] = [1,-1,10,-10,3]

   assert len(func(events_df)) == 3
