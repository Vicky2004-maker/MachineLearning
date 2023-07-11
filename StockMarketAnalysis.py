from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from twelvedata import TDClient

api_key = 'dc079599235b4000aa2bea861faaf52b'

# Initialize client - apikey parameter is requiered
td = TDClient(apikey=api_key)

# Construct the necessary time series
ts = td.time_series(
    symbol="AAPL",
    interval="1min",
    outputsize=1000,
    timezone="America/New_York",
)

# Returns pandas.DataFrame
df = ts.as_pandas()

# %%

_df = pd.DataFrame(df, columns=['high'])
#_df = _df.reset_index().rename(columns={'datetime': 'datetime'})

# Set the Date as Index
_df['datetime'] = pd.to_datetime(_df['datetime'])
#_df.index = df['datetime']
#del _df['datetime']

_df.plot(figsize=(15, 6))
plt.show()

#%%
