from matplotlib import pyplot
from pandas import Series

series = Series.from_csv('E:\\Mex\\Data\\1\\act\\03_01-07_1_t.csv', header=0)
series.plot()
pyplot.show()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_pacf(series, lags=500)
pyplot.show()