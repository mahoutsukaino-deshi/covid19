import pandas as pd
from matplotlib import pylab as plt
from fbprophet import Prophet
import japanize_matplotlib

MAX_Y = 10000


def get_data():
    url = 'https://www.mhlw.go.jp/content/pcr_positive_daily.csv'
    df = pd.read_csv(url,
                     usecols=[0, 1],
                     names=['ds', 'y'],
                     skiprows=1,
                     parse_dates=['ds'],
                     )
    return df


def main():
    plt.xkcd()

    df = get_data()
    df['cap'] = MAX_Y

    model = Prophet(growth='logistic')
    model.fit(df)
    df_future = model.make_future_dataframe(periods=30)
    df_future['cap'] = MAX_Y
    predicts = model.predict(df_future)
    model.plot(predicts)
    plt.tight_layout()
    plt.title('Prediction COVID-19 by Prophet Model')
    plt.savefig('covid19-prophet2.png')
    plt.show()


if __name__ == '__main__':
    main()
