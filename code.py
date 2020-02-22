import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor
from xgboost import plot_importance, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings  
warnings.filterwarnings('ignore')
plt.style.use('ggplot')




pjme_df = pd.read_csv('PJME_hourly.csv', parse_dates=[0], index_col=[0])
pjme_df = pjme_df.loc[~pjme_df.index.duplicated(keep='first')].sort_index().dropna()



city = 'New York'
humidity = pd.read_csv('humidity.csv', parse_dates=[0], index_col=[0])[city].dropna().rename('humidity')
pressure = pd.read_csv('pressure.csv', parse_dates=[0], index_col=[0])[city].dropna().rename('pressure')
temperature = pd.read_csv('temperature.csv', parse_dates=[0], index_col=[0])[city].dropna().rename('temperature')
wind_direction = pd.read_csv('wind_direction.csv', parse_dates=[0], index_col=[0])[city].dropna().rename('wind_direction')
wind_speed = pd.read_csv('wind_speed.csv', parse_dates=[0], index_col=[0])[city].dropna().rename('wind_speed')


weather_df = pd.concat([temperature, humidity, pressure, wind_direction, wind_speed], axis=1).sort_index()
weather_df = weather_df.loc[~weather_df.index.duplicated(keep='first')].sort_index().dropna()
weather_df = weather_df.assign(pressure_log = weather_df.pressure.apply(np.log))



comb_df = pd.concat([pjme_df.loc[weather_df.index[0]:weather_df.index[-1]], weather_df], axis=1).sort_index().dropna()



fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(20,15))
axes[0].set_title('PJME Power Consumption')
axes[0].set_ylabel('Power(MW)')
comb_df.PJME_MW.plot(ax=axes[0])

axes[1].set_title('Temperature')
axes[1].set_ylabel('Temperature(K)')
comb_df.temperature.plot(ax=axes[1])

axes[2].set_title('Pressure')
axes[2].set_ylabel('Pressure')
comb_df.pressure.plot(ax=axes[2])

axes[3].set_title('Pressure_log')
axes[3].set_ylabel('Pressure_log')
comb_df.pressure_log.plot(ax=axes[3]                      
                      
                     )

plt.tight_layout()
plt.show()

final_df = (comb_df.assign( day_of_week = comb_df.index.dayofweek
                            ,year = comb_df.index.year
                            ,month = comb_df.index.month
                            ,day = comb_df.index.day
                            ,day_of_year = comb_df.index.dayofyear

                            ,week = comb_df.index.week
                            ,week_day = comb_df.index.weekday_name 
                            ,quarter = comb_df.index.quarter
                            ,hour = comb_df.index.hour
                            ,hour_x = np.sin(2.*np.pi*comb_df.index.hour/24.)
                            ,hour_y = np.cos(2*np.pi*comb_df.index.hour/24.)
                            ,day_of_year_x = np.sin(2.*np.pi*comb_df.index.dayofyear/365.)
                            ,day_of_year_y = np.cos(2.*np.pi*comb_df.index.dayofyear/365.)

                          )
           )

# df['hourfloat']=df.hour+df.minute/60.0
# df['x']=np.sin(2.*np.pi*df.hourfloat/24.)
# df['y']=np.cos(2.*np.pi*df.hourfloat/24.)

lagged_df = final_df.copy()

# Next day's load values.
lagged_df['load_tomorrow'] = lagged_df['PJME_MW'].shift(-24)    

for day in range(8):
    lagged_df['temperature_d' + str(day)] = lagged_df.temperature.shift(24*day)
    lagged_df['wind_speed_d' + str(day)] = lagged_df.wind_speed.shift(24*day)
    lagged_df['humidity_d' + str(day)] = lagged_df.humidity.shift(24*day)
    lagged_df['pressure_log_d' + str(day)] = lagged_df.pressure_log.shift(24*day)

    
    
    lagged_df['load_d' + str(day)] = lagged_df.PJME_MW.shift(24*day)

     
lagged_df = lagged_df.dropna()
    

lagged_df = lagged_df.drop(columns=['temperature', 'wind_speed', 'humidity', 'pressure', 'wind_direction', 'week_day','PJME_MW'])

X = lagged_df.drop(columns=['load_tomorrow'])
y = lagged_df['load_tomorrow']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)


def plot_prediction(actual, prediction, start_date, end_date, title, prediction_label):
    plt.figure(figsize=(20,5))
    plt.title(title)
    plt.plot(y_test.index, y_test, label='Actual')
    plt.plot(y_test.index, prediction, label=prediction_label)
    plt.ylabel('Power(MW)')
    plt.xlabel('Datetime')
    plt.legend()
    plt.xlim(left= start_date, right=end_date)
    plt.show()
    
def subplot_prediction(actual, prediction,prediction_label):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 15))
    
    con_df = pd.concat([actual.rename('Actual'),pd.DataFrame(prediction, index=actual.index, columns=[prediction_label])], axis=1)
    axes[0].set_title('Actual vs Prediction - One day')
    axes[0].set_ylabel('Power(MW)')
    axes[0].set_xlabel('Datetime')
    con_df.plot(ax=axes[0])
    axes[0].set_xlim(left=con_df.index[-24*1] , right=con_df.index[-1])
    
    axes[1].set_title('Actual vs Prediction - One week')
    axes[1].set_ylabel('Power(MW)')
    axes[1].set_xlabel('Datetime')
    con_df.plot(ax=axes[1])
    axes[1].set_xlim(left=actual.index[-24*7] , right=actual.index[-1])
    
    axes[2].set_title('Actual vs Prediction - One month')
    axes[2].set_ylabel('Power(MW)')
    axes[2].set_xlabel('Datetime')
    con_df.plot(ax=axes[2])
    axes[2].set_xlim(left=actual.index[-24*7*4] , right=actual.index[-1])
    
    plt.tight_layout()
    plt.show()
    
def plot_feature_importances( clf, X_train, y_train=None
                             ,top_n=10, figsize=(10,18), print_table=False, title="Feature Importances"):
    feat_imp = pd.DataFrame({'importance':clf.feature_importances_})    
    feat_imp['feature'] = X_train.columns
    feat_imp.sort_values(by='importance', ascending=False, inplace=True)
    feat_imp = feat_imp.iloc[:top_n]
    
    feat_imp.sort_values(by='importance', inplace=True)
    feat_imp = feat_imp.set_index('feature', drop=True)
    feat_imp.plot.barh(title=title, figsize=figsize)
    plt.xlabel('Feature Importance Score')
    plt.show()
    
    if print_table:
        from IPython.display import display
        print("Top {} features in descending order of importance".format(top_n))
        display(feat_imp.sort_values(by='importance', ascending=False))
        
    return feat_imp


reg = xgb.XGBRegressor(silent=True)

tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(reg, X.values, y.values, cv=tscv
                         ,scoring='explained_variance'
                        )
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() ))
print(scores)


reg.fit(X_train,y_train)
prediction = reg.predict(X_test)

#_ = plot_feature_importances(reg, X_train, y_train, top_n=X_train.shape[1], title=reg.__class__.__name__, print_table=True)


def get_features(date, comb_df):
    features = comb_df.loc[date]
    features = (features.assign(
                                day_of_week = features.index.dayofweek
                                ,year = features.index.year
                                ,month = features.index.month
                                ,day = features.index.day
                                ,day_of_year = features.index.dayofyear
                                ,week = features.index.week
#                                             ,week_day = features.index.weekday_name 
                                ,quarter = features.index.quarter
                                ,hour = features.index.hour
                                ,hour_x = np.sin(2.*np.pi*features.index.hour/24.)
                                ,hour_y = np.cos(2*np.pi*features.index.hour/24.)
                                ,day_of_year_x = np.sin(2.*np.pi*features.index.dayofyear/365.)
                                ,day_of_year_y = np.cos(2.*np.pi*features.index.dayofyear/365.)
                                
                                ))
    
    for day in range(8):
        features['temperature_d' + str(day)] = comb_df.temperature.shift(24*day)
        features['wind_speed_d' + str(day)] = comb_df.wind_speed.shift(24*day)
        features['humidity_d' + str(day)] = comb_df.humidity.shift(24*day)
        features['pressure_log_d' + str(day)] = comb_df.pressure_log.shift(24*day)



        features['load_d' + str(day)] = comb_df.PJME_MW.shift(24*day)

    features = features.dropna()
    
    features = features.drop(columns=['temperature', 'wind_speed', 'humidity', 'pressure', 'wind_direction','PJME_MW'])

    return features



date = input("Give a date between 2012-10-01 and 2017-10-27: ")


prediction = reg.predict(get_features(date, comb_df))
idx = comb_df.PJME_MW.loc[date].index 


def plot_prediction_multistep(actual, prediction, start_date, title, prediction_label):
    date_rng = pd.date_range(start=start_date, periods=24, freq='H')
    plt.figure(figsize=(20,5))
    plt.title(title)
    plt.plot(actual.index, actual, label='Actual')
    plt.plot(actual.index, prediction, label=prediction_label)
    plt.ylabel('Power(MW)')
    plt.xlabel('Datetime')
    plt.legend()
    plt.show()
    
plot_prediction_multistep(actual=comb_df.PJME_MW.loc[date],prediction=prediction, start_date=date, title='Multistep prediction - 24 hours a head',
                prediction_label='ExtraTrees Regressor model prediction') 



