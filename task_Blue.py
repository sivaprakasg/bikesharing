# importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_squared_log_error
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import copy
#%%  # reading the dataset
bikes_hourly = pd.read_csv('hour.csv')
bike_daily = pd.read_csv('day.csv')
bikes_hour_df = bikes_hourly.drop(['dteday','instant','casual' , 'registered'], axis=1)
print('total dataset \n:', bikes_hour_df)
#Getting the shape of dataset with rows and columns
print(bikes_hour_df.shape)
#check for count of missing values in each column.
bikes_hour_df.isnull().sum()

#%% 
class BikeRentalDataVisualizer:
    def __init__(self, data):
        self.data = data
        self.results_table = []
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.results = None
        self.best_model = None
        
        
    def data_dist(self, col): # distribution of total customers data in the dataset
        fig,  (ax1, ax2)  =  plt.subplots(nrows=1, ncols=2, figsize=(13, 6))
        ax1 = sns.boxplot(bikes_hour_df['cnt'], ax = ax1)
        ax1.set_title('Boxplot for variable cnt')
        ax2 = sns.distplot(bikes_hour_df['cnt'], ax = ax2)
        ax2.set_title('Distribution curve of cnt')
        fig.tight_layout()
        plt.show()
    
        
    def remove_outliers_iqr(self, col): #removing outliers in the dataset
        Q1 = self.data['cnt'].quantile(0.25)
        Q3 = self.data['cnt'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR     
        self.data = self.data[(self.data['cnt'] >= lower_bound) & (self.data['cnt'] <= upper_bound)]
        return self.data

    def point_plot_hr(self, x, y, hue, title): #plotting count of all bikes rented with respect to days
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.pointplot(data=self.data, x='hr', y='cnt', hue='weekday', ax=ax)
        ax.set(title=title)
        plt.suptitle(f'Numerical Feature: {y} vs {x}')
        plt.xlabel(x)
        plt.ylabel('Count of all Bikes Rented')
        plt.show()

    def point_plot_season(self, x, y,hue, title): #plotting count of bikes rented with respect to seasons
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.pointplot(data=self.data, x='hr', y='cnt', hue='season', ax=ax)
        ax.set(title=title)
        plt.suptitle(f'Numerical Feature: {y} vs {x}')
        plt.xlabel(x)
        plt.show()

    def create_correlation_heatmap(self):#correlation map to show relationship between features
        corr = self.data.corr()
        plt.figure(figsize=(15, 10))
        sns.heatmap(corr, annot=True, annot_kws={'size': 15})
        plt.show()
        

    def plot_box_plots(self): #boxplots to see vaiours feature distribution
        sns.set(font_scale=1.0)
        fig, axes = plt.subplots(nrows=2, ncols=2 , figsize=(30, 30))           
        sns.boxplot(data=data, y="cnt", x="mnth", orient="v", ax=axes[0][0])
        sns.boxplot(data=data, y="cnt", x="weathersit", orient="v", ax=axes[0][1])
        sns.boxplot(data=data, y="cnt", x="hr", orient="v", ax=axes[1][0])
        sns.boxplot(data=data, y="cnt", x="temp", orient="v", ax=axes[1][1])
    
        axes[0][0].set(xlabel='Month', ylabel='Count', title="Box Plot for Count vs Months")
        axes[0][1].set(xlabel='Weather Situation', ylabel='Count', title="Box Plot for Count vs Weather Situations")
        axes[1][0].set(xlabel='Hour Of The Day', ylabel='Count', title="Box Plot for Count vs Hours")
        axes[1][1].set(xlabel='Temperature', ylabel='Count', title="Box Plot On Count vs Temperature")    
        plt.show()


 #%%      
    def create_grouped_bar_plots(self, group_col, count_col, title, x_labels): #bike rentals based on seasons and weather conditions
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(13, 6))
        grouped_data_season = self.data[['season','cnt']].groupby(['season']).sum().reset_index()
        ax1 = grouped_data_season.plot(kind='bar', legend=False, title=f'Counts of Bike Rentals by {group_col}', 
                                       stacked=True, fontsize=12, ax=ax1, color='r')
        ax1.set_xlabel("season", fontsize=12)
        ax1.set_ylabel("Count", fontsize=12)
        ax1.set_xticklabels(['spring','sumer','fall','winter'])

        grouped_data_weathersit = self.data[['weathersit', 'cnt']].groupby(['weathersit']).sum().reset_index()
        ax2 = grouped_data_weathersit.plot(kind='bar', legend=False, title=f'Counts of Bike Rentals by {group_col}', 
                                           stacked=True, fontsize=12, ax=ax2, color='b')
        ax2.set_xlabel("weathersit", fontsize=12)
        ax2.set_ylabel("Count", fontsize=12)
        ax2.set_xticklabels(['1: Clear','2: Mist','3: Light Snow','4: Heavy Rain'])

        fig.tight_layout()
        
    def one_hot_encoding(self, column):
        self.data = pd.get_dummies(self.data, columns=[column], prefix=column, drop_first=True)
    
      
#%% 
    def split_data(self): # Split data into training and testing sets
        X = self.data.drop('cnt', axis=1)
        y = self.data['cnt']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.25, random_state=42)
#%%           
    def train_models(self, X, y): #training the models
        models = [LinearRegression(), Ridge(), HuberRegressor(), ElasticNetCV(), DecisionTreeRegressor(),
                  RandomForestRegressor()]
        results = []
       
        
        for model in models:
            
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test
            model.fit(x_train,y_train)
            y_pred = model.predict(x_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            if model.__class__.__name__ == "LinearRegression":
                
                linear = model
                
            elif model.__class__.__name__ == "Ridge":
                
                ridge = model
                
            elif model.__class__.__name__ == "HuberRegressor":
                
                huber = model
                
            elif model.__class__.__name__ == "ElasticNetCV":
               
                cv = model
                
            elif model.__class__.__name__ == "DecisionTreeRegressor":
                decision_tree = model
                
            elif model.__class__.__name__ == "RandomForestRegressor":
                random_forest = model
                
            results.append([type(model).__name__, f'{mae:.2f}', f'{r2:.2f}'])
            
        
        model_fit = {"LinearRegression" : linear,
                     "Ridge" : ridge,
                     "HuberRegressor" : huber,
                     "ElasticNetCV" : cv,
                     "DecisionTreeRegressor" : decision_tree,
                     "RandomForestRegressor" : random_forest}
        
        self.results = results
        print(tabulate(results, headers=['Model', 'Mean Absolute Error (MAE)', 'R2 Score'], tablefmt='pretty'))
        #tabulating results of the models
        
        min_value = float(results[0][1])
        for row in results:
            
            if float(row[1]) < min_value:
                min_value = float(row[1])
                model_name = row[0]
                best_model = model_fit["{}".format(row[0])]
        print("The best model is {0} with MAE:{1} ".format(model_name, min_value))   
        self.best_model = best_model #selecting best model based on MAE score
#%% We chose Random forest model and now we can see how various error measures during training or testing

    def evaluate(self, model, x,y, data_type = None):
       
        y_pred = model.predict(x)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        score = model.score(x, y)    
        rmsle = np.sqrt(mean_squared_log_error(y, y_pred))
        results = [[model.__class__.__name__, mae, mse, score, rmsle, data_type]]
        #self.results_table.append([type(model).__name__, dataset, format(mse, '.2f'), format(mae, '.2f'), format(rmsle, '.2f'), format(score, '.2f')])
        print(tabulate(results, headers=['Model', 'Mean Absolute Error (MAE)', 'MSE', 'R2 Score', 'RMSLE', '{}'.format(data_type) ], tablefmt='pretty'))    
                
#%%
# actual dataset
data = bikes_hour_df

# Create an instance of the BikeRentalDataVisualizer
visualizer = BikeRentalDataVisualizer(data)
# Select relevant features from the dataset
category_features = data[['season', 'holiday', 'mnth', 'hr', 'weekday', 'workingday', 'weathersit']]
number_features = data[['temp', 'atemp', 'hum', 'windspeed']]
X= pd.concat([category_features, number_features], axis=1)
y= data['cnt']

# Visualize the distribution of the 'cnt' variable
visualizer.data_dist('cnt')

#Remove outliers from the 'cnt' variable using the Interquartile Range (IQR) method
c = visualizer.remove_outliers_iqr('cnt')
c= visualizer.remove_outliers_iqr('cnt')
print ('Dataset after removing outliers:\n', c)

# # Plot box plots for various features
visualizer.plot_box_plots()
# # Plot point plots to visualize bike counts by hour and season
visualizer.point_plot_hr('hr', 'cnt', 'weekday', 'Count of bikes during weekdays and weekends')
visualizer.point_plot_season('hr', 'cnt', 'season', 'Count of bikes during different seasons')
# Create a correlation heatmap to visualize feature relationships
visualizer.create_correlation_heatmap()
# Create grouped bar plots to show bike counts by season and weather conditions
visualizer.create_grouped_bar_plots('season', 'cnt', 'Counts of Bike Rentals by season', ['Spring', 'Summer', 'Fall', 'Winter'])
#Train machine learning models using the selected features and target variable
visualizer.train_models(X , y)
# Evaluate the model with training data
visualizer.evaluate(visualizer.best_model, visualizer.x_train, visualizer.y_train, 'training')
# Evaluate the model with testing data
visualizer.evaluate(visualizer.best_model, visualizer.x_test, visualizer.y_test, 'testing')





