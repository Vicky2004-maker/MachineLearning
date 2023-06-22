import pandas as pd
from sklearn.linear_model import LinearRegression

# %%
world_population = pd.read_excel("E:/Dataset/World Population/world_bank_population.xlsx")

# %%
country_name = 'India'
country = world_population.loc[world_population['Country Name'] == country_name]
country = country.drop(['Country Name', 'Country Code'], axis=1)
country = country.T
country.dropna(inplace=True)
country = country.reset_index().rename(columns={109: 'population', 'index': 'year'})
# %%
year = 2023
linear_regression = LinearRegression()
x = country.iloc[:, 0].values.reshape(-1, 1)
y = country.iloc[:, 1].values.reshape(-1, 1)
model = linear_regression.fit(x, y)
y_pred = model.predict([[year]])
y = y_pred[0]
print(f'The population of {country_name} in the year {year} will be {round(y[0])}')

# %%
