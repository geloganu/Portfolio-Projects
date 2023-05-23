import pandas as pd
import numpy as np

class cost_dataframe:
    def __init__(self, members):
        self.members = members
        self.num_members = len(members)

        self.cost_table=pd.DataFrame(index=[], columns=self.members)
        self.food_index=np.array([])
        return
    
    def print(self):
        return self.num_members, 'people including', self.members

    def add_item(self, item_name, cost, participants):
        individual_cost= cost/len(participants)

        ind_cost_row=np.zeros(self.num_members)

        index=-1
        for x in self.members:
            index =index + 1 
            for y in participants:
                if x == y:
                    ind_cost_row[index]=individual_cost


        self.cost_table.loc[item_name,:]=ind_cost_row

    def add_item_all(self, item_name,cost):
        
        individual_cost=cost/self.num_members
        ind_cost_row=np.ones(self.num_members)*individual_cost

        self.cost_table.loc[item_name,:]=ind_cost_row

    def check_out(self, tip):

        ind_cost_row=np.ones(self.num_members)*tip/self.num_members
        self.cost_table.loc['Tip/Tax',:]=ind_cost_row

        self.cost_table.loc['Sum (MXN)',:] = self.cost_table.sum()
        self.cost_table.loc['Sum (USD)',:] = self.cost_table.loc['Sum (MXN)']/18.91

        self.cost_table.style



