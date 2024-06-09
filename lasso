from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

lasso = Lasso(alpha = 0.1)
lasso.fit(x_train, y_train)

y_train_pred = lasso.predict(x_train)
y_test_pred = lasso.predict(x_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
cv_rmse = np.mean(np.sqrt(-cross_val_score(lasso, x_train, y_train, scoring = 'neg_mean_squared_error', cv = 10)))

print(f'Train RMSE is {train_rmse}')
print(f'Test RMSE is {test_rmse}')
print(f"Cross Validation score's RMSE is {train_rmse}")

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f'Training R2 score is {train_r2}')
print(f'Test R2 score is {test_r2}')
