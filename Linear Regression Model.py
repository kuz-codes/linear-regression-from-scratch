import pandas as pd


class LinearRegression():
    def __init__(self, x: pd.Series, y: pd.Series):
        # Computing for intercept
        n = x.size
        numerator = sum(y)*sum(x**2) - sum(x)*sum(x*y)
        denominator = n*sum(x**2) - sum(x)**2
        self.intercept = numerator/denominator

        # Computing for slope
        numerator = (n*sum(x*y)) - (sum(x)*sum(y))
        denominator = (n*sum(x**2)) - (sum(x)**2)
        self.slope = numerator/denominator

    def get_intercept(self):
        return self.intercept

    def get_slope(self):
        return self.slope

    def get_equation(self):
        return f"{self.intercept} + {self.slope}x"

    def predict(self, x: float):
        return self.intercept + (self.slope * x)
