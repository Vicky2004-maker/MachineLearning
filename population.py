import pandas as pd
from sklearn.linear_model import LinearRegression

# %%
world_population = pd.read_excel("E:/Dataset/World Population/world_bank_population.xlsx")

# %%
country_name = 'Brazil'
country = world_population.loc[world_population['Country Name'] == country_name]
country = country.drop(['Country Name', 'Country Code'], axis=1)
country = country.T
country.dropna(inplace=True)
country = country.reset_index().rename(columns={29: 'population', 'index': 'year'})
# %%
linear_regression = LinearRegression()
x = country.iloc[:, 0].values.reshape(-1, 1)
y = country.iloc[:, 1].values.reshape(-1, 1)
model = linear_regression.fit(x, y)
y_pred = model.predict([[i for i in range(2022, 2051)]])
y = y_pred[0]
print(*y)

# %%
