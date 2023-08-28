import pandas as pd
from sklearn.metrics import mean_absolute_error # model validity
from sklearn.model_selection import train_test_split # breaks data into pieces
from sklearn.tree import DecisionTreeRegressor

melbourne_file_path = 'melb_data/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) # load data with pandas

# dropna drops missing values
filtered_melbourne_data = melbourne_data.dropna(axis=0)

y = filtered_melbourne_data.Price # create prediction target object, stored in Series

# columns inputted into model called "features"
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
melbourne_model = DecisionTreeRegressor() # define model

melbourne_model.fit(train_X, train_y) # fit model


val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

# compare MAE with differing values of max_leaf_nodes
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return (mae)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max lead nodes: %d  \t\t Mean Absolute Error: %d"  %(max_leaf_nodes, my_mae))

# # print the top few validation predictions
# print(melbourne_model.predict(val_X.head()))
# # print the top few actual prices from validation data
# print(y.head().tolist())

