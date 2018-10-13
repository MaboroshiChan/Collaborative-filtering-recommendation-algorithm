import numpy as np
import pandas as pd
class CF:

    def __init__(self, numMovies, numUsers, topK):
        self.numMovies = numMovies
        self.numUsers = numUsers
        self.table = np.full((numMovies, numUsers), -1.0)
        self.similarity_table = np.full((numMovies, numUsers), 0.0)
        self.user_average = np.zeros(numUsers)
        self.topK = topK

    def loadData(self, ratings_location):
        f = pd.read_csv(ratings_location)
        f = f.values
        for each in f:
            self.table[each[1] - 1][each[0] - 1] = each[2]
        for usr in np.arange(0, self.numUsers):
            r = self.table[:, usr - 1]
            r = r[r != -1.0]
            self.user_average[usr - 1] = np.average(r)

        for i in range(self.numMovies):
            for j in range(self.numMovies):
                self.similarity_table[i][j] = self.sim(i, j)
                print(self.sim(i, j), "sim")
        print(self.table)
    def sim(self, itemA, itemB):
        x = self.table[itemA - 1] - self.user_average
        y = self.table[itemB - 1] - self.user_average
        ans = np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))
        return ans*ans

    def pred(self, userID, item):
        numer = 0.0
        denom = 0.0
        for N in range(0, self.numMovies):
            sim = self.similarity_table[item - 1][N]
            numer += sim*self.table[N][userID - 1]
            denom += sim
        return numer/denom

    def displayTopK(self, userID):
        highest_rated_itemID = self.table[:userID - 1].argmax()
        simple = self.similarity_table[highest_rated_itemID]
        simple = simple.argsort()[-3:][::-1]
        return simple


cf = CF(5, 5, 5)

cf.loadData(r"ratings.csv")
print(cf.pred(1, 1))




