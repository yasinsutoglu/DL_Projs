import pandas as pd
from datetime import datetime
import seaborn as sns
from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)


df = pd.read_csv("tweets_labeled.csv")


df.columns
df.shape
df.head()
df.info()

# Converting the time zone of the date variable to the Istanbul time zone
df["date"] = pd.to_datetime(df["date"])
df['date'] = df['date'].dt.tz_convert('Europe/Istanbul')
df['date'] = df['date'].dt.tz_localize(None)


# Creating and editing the "month" variable
df['month'] = df['date'].dt.month_name()
df['tweet'] = df['tweet'].str.lower()
df["month"] = df['month'].replace({'December': 'Aralık',
                                   'January': 'Ocak',
                                   'February': 'Şubat',
                                   'March': 'Mart',
                                   'April': 'Nisan',
                                   'May': 'Mayıs',
                                   'June': 'Haziran',
                                   'July': 'Temmuz',
                                   'August': 'Ağustos',
                                   'September': 'Eylül',
                                   'October': 'Ekim',
                                   'November': 'Kasım'
                                   })

# Creating the "seasons" variable
seasons = {'Ocak': 'Kış',
           'Şubat': 'Kış',
           'Mart': 'İlkbahar',
           'Nisan': 'İlkbahar',
           'Mayıs': 'İlkbahar',
           'Haziran': 'Yaz',
           'Temmuz': 'Yaz',
           'Ağustos': 'Yaz',
           'Eylül': 'Sonbahar',
           'Ekim': 'Sonbahar',
           'Kasım': 'Sonbahar',
           'Aralık': 'Kış'}

df['seasons'] = df['month'].map(seasons)

# Creating the "day" variable
df["days"] = [date.strftime('%A') for date in df["date"]]
df["days"] = df["days"].replace({"Monday" : "Pazartesi",
                                 "Tuesday" : "Salı",
                                 "Wednesday" : "Çarşamba",
                                 "Thursday": "Perşembe",
                                 "Friday" : "Cuma",
                                 "Saturday" : "Cumartesi",
                                 "Sunday": "Pazar"})

# Dividing the day into six parts with 4-hour intervals
df['hour'] = df['date'].dt.hour
df['4hour_interval'] = (df['hour'] // 2) * 2
interval = {0: '0-2',
            2: '2-4',
            4: '4-6',
            6: '6-8',
            8: '8-10',
            10: '10-12',
            12: '12-14',
            14: '14-16',
            16: '16-18',
            18: '18-20',
            20: '20-22',
            22: '22-24'
            }
df['4hour_interval'] = df['4hour_interval'].map(interval)
df["time_interval"] = df["4hour_interval"].replace({"0-2": "22-02",
                                                   "22-24": "22-02",
                                                   "2-4": "02-06",
                                                   "4-6": "02-06",
                                                   "6-8": "06-10",
                                                   "8-10": "06-10",
                                                   "10-12": "10-14",
                                                   "12-14": "10-14",
                                                   "14-16": "14-18",
                                                   "16-18": "14-18",
                                                   "18-20": "18-22",
                                                   "20-22": "18-22"})

df.drop(["4hour_interval", "hour"], axis=1, inplace=True)

cols = ["time_interval", "days", "seasons"]

def summary(dataframe, col_name, plot=False):
    # target variable analysis for negative tweets
    dataframe = dataframe.loc[df["Durum"] == -1]
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("---------------------------------------------")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cols:
    summary(df, col, plot=True)


