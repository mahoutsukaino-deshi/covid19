import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib


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
    df = get_data()

    df.plot()
    plt.title('COVID19 日別感染者数')
    plt.show()


if __name__ == '__main__':
    main()
