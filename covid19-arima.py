import datetime
import pandas as pd
from matplotlib import pylab as plt
import japanize_matplotlib
import statsmodels.api as sm


def get_data():
    url = 'https://www.mhlw.go.jp/content/pcr_positive_daily.csv'
    df = pd.read_csv(url,
                     usecols=[0, 1],
                     names=['date', 'positives'],
                     skiprows=1,
                     parse_dates=['date'],
                     index_col='date'
                     )
    return df


def main():
    plt.xkcd()

    df = get_data()
    diff = df['positives'].diff()
    diff = diff.dropna()

    params = sm.tsa.arma_order_select_ic(diff, ic='aic', trend='nc')
    aic_order = params['aic_min_order']

    '''
    orderはarma_order_select_icで求めた値を指定。
    seasonal_orderは1週間周期なので4番目に7を指定。
    '''
    model = sm.tsa.SARIMAX(
        df,
        order=(aic_order[0], 1, aic_order[1]),
        seasonal_order=(1, 1, 1, 7)
    ).fit()

    '''
    30日分の予想をしてみる
    '''
    predict_period_from = df.index.max()
    predict_period_to = df.index.max() + datetime.timedelta(days=30)

    predict = model.predict(predict_period_from, predict_period_to)
    plt.plot(df, label='real')
    plt.plot(predict, label='predict')
    plt.title('Prediction COVID-19 by SARIMA Model')
    plt.savefig('covid19-arima2.png')
    plt.show()


if __name__ == '__main__':
    main()
