import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

sns.set(style='whitegrid', context='notebook')

df = pd.read_csv('Vanilla Arma 3 Data.csv')
fmle = LabelEncoder()
stle = LabelEncoder()

sc = StandardScaler()

df['Firing Mode'] = fmle.fit_transform(df['Firing Mode'].values)

df['Distance'] = [euclidean((i, j), (k, l)) for i, j, k, l in df[['Firing Position X', 'Firing Position Y', 'Target Position X', 'Target Position Y']].values]
df['Alt. Difference'] = [i - j for i, j in df[['Firing Elevation', 'Target Elevation']].values]

ex = pd.DataFrame()
cols = ['Distance', 'Alt. Difference', 'Firing Angle']
sns.pairplot(df[cols], size=2.5)
plt.show()

X = pd.DataFrame()

X['Distance'] = df['Distance'].values
X['Alt. Difference'] = df['Alt. Difference'].values

y = pd.DataFrame()
y['Firing Angle'] = df['Firing Angle'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

# Fit Random Forest and meausre its performance
forest = RandomForestRegressor(n_estimators=1000, criterion='mse')
forest.fit(X_train, np.ravel(y_train))

y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))

resid_train = y_train_pred - y_train['Firing Angle']
resid_test = y_test_pred - y_test['Firing Angle']

plt.scatter(y_train_pred, resid_train, c='black', marker='o', s=35, alpha=0.5, label='Training data')
plt.scatter(y_test_pred, resid_test, c='lightgreen', marker='s', s=35, alpha=0.7, label='Test data')

plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.show()

# Fit Polynomial Regresser and measure its performance
quadratic = PolynomialFeatures(degree=2)
pr = LinearRegression()
X_quad_train = quadratic.fit_transform(X_train, np.ravel(y_train))
X_quad_test = quadratic.fit_transform(X_test, np.ravel(y_test))
pr.fit(X_quad_train, y_train)

pr_train_pred = pr.predict(X_quad_train)
pr_test_pred = pr.predict(X_quad_test)

print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, pr_train_pred), mean_squared_error(y_test, pr_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, pr_train_pred), r2_score(y_test, pr_test_pred)))


# Take inputs from user
while False:
    pos_x_coor = int(input('Please enter your X coordinate:\n'))
    pos_y_coor = int(input('Please enter your Y coordinate:\n'))

    tar_x_coor = int(input('Please enter X coordinate of target:\n'))
    tar_y_coor = int(input('Please enter Y coordinate of target:\n'))

    elev = int(input('Please enter the difference in elevation:\n'))
    dist = euclidean((pos_x_coor, pos_y_coor), (tar_x_coor, tar_y_coor))

    type = input('Please enter 1 for forest, 2 for polynomial:\n')
    if type == '1':
        req = pd.DataFrame()
        req['Distance'] = [dist]
        req['Alt. Difference'] = [elev]
        print(forest.predict(req))
    elif type == '2':
        req = pd.DataFrame()
        req['Distance'] = [dist]
        req['Alt. Difference'] = [elev]
        print(pr.predict(quadratic.fit_transform(req)))
    else:
        print('ERROR: Wrong type selected\n')


# Plot predictions across range of values for forest regresser
disp_X = pd.DataFrame()
disp_X['Distance'] = [i for i in range(100, 250)]
disp_X['Alt. Difference'] = [0 for i in range(100, 250)]

disp_y = forest.predict(disp_X)

plt.scatter(disp_X['Distance'], disp_y)
plt.title('Random Forest Regresser')
plt.xlabel('Distance (10m)')
plt.ylabel('Firing Angle (deg)')
plt.show()

# Plot prediction across range of values for polynomial regresser
disp_X = pd.DataFrame()
disp_X['Distance'] = [i for i in range(100, 250)]
disp_X['Alt. Difference'] = [0 for i in range(100, 250)]

disp_y = pr.predict(quadratic.fit_transform(disp_X))

plt.scatter(disp_X['Distance'], disp_y)
plt.title('Polynomial Regresser (deg=2)')
plt.xlabel('Distance (10m)')
plt.ylabel('Firing Angel (deg)')
plt.show()