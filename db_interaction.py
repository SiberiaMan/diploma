import pandas as pd

class RespositoryDB():
    def __init__(
            self,
            col1: str,
            col2: str,
            col3: str,
            col4: str,
    ):
        self.df = pd.DataFrame(columns=[col1, col2, col3, col4])
        self.col1 = col1
        self.col2 = col2
        self.col3 = col3
        self.col4 = col4

    def add(self, val1: int, val2: float, val3: float, val4: str):
        row = pd.DataFrame({
            self.col1: [val1],
            self.col2: [val2],
            self.col3: [val3],
            self.col4: [val4],
        })
        self.df = self.df.append(row, ignore_index=True)

    def save_xls(self):
        self.df.to_excel('results.xls')

    def save_csv(self):
        self.df.to_csv('results.csv')
