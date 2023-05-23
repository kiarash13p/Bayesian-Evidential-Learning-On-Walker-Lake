import pandas as pd
import numpy as np
import re,csv
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

data_path   =   r"Data/"
img_path    =   r"Images/"
num_sims    =   60
# Reading Data
data = pd.read_csv(r'Data/Data for BEL.csv')
print( f"***\nShape of the data is {data.shape} with columns:\n{data.columns.values}")

reg_v = r".*_backtransformed.*"
reg_t = r".*T___real.*"
compil_v = re.compile(reg_v)
compil_t = re.compile(reg_t)
realization_v_columns = list(filter(compil_v.match,data.columns))
realization_t_columns = list(filter(compil_t.match,data.columns))

def plot_V():
    for xsection in range(1,261):
        plt.style.use('bmh')
        fig , ax = plt.subplots(figsize=(15,5))
        selection = data[ data['X'] == xsection ]
        ax.plot(selection['V'] , color = 'blue' , label = 'V True value')
        for col in realization_v_columns:
            ax.plot(selection[col] , color = 'palegreen' , alpha = 0.08 , label = "Simulated Values" , linestyle = ':')
        ax.plot( selection['V_Etype'] , color = 'olivedrab' , label = 'Etype of V Realizations')
        ax2 = ax.twinx()
        ax2.plot(selection['H'],color='red', label = 'H' )
        ax_h , ax_l = ax.get_legend_handles_labels()
        ax_plots = dict( zip(ax_l , ax_h) )
        ax.legend( handles = ax_plots.values() , labels = ax_plots.keys() , loc = 'upper left' , fontsize = 'x-small')
        ax2.legend( loc = 'upper right' , fontsize = 'x-small')

        ax.set_ylabel( "V" , fontsize = 'large')
        ax2.set_ylabel( "H" , fontsize = 'large')
        fig.suptitle( f"Section view of section {xsection}")
        fig.savefig( img_path + f"Sections for V/v_plot_section_{xsection}.png")
        plt.close(fig)

def plot_T():
    for xsection in range(0,260):
        # plt.style.use('bmh')
        fig , ax = plt.subplots(figsize=(20,5))
        selection = data[ data['X'] == xsection ]
        selection.reset_index( inplace = True )
        for idx,row in selection.iterrows():
            count_cat_1   =   list(row[realization_t_columns].value_counts())[0]
            counts = [count_cat_1 , 60-count_cat_1 ]
            starts = [0 , counts[0] ]
            labels = ['Type1','Type2']
            #ax.barh( y = [''] , width = widths , align='edge' , color = 'red')
            bar1 = ax.bar( idx , counts[0] , width = 1 , color = 'palegreen' , label='Rock Type 1' , bottom=starts[0] )
            bar2 = ax.bar( idx , counts[1] , width = 1 , color = 'olivedrab' , label='Rock Type 2' , bottom=starts[1] )
        ax2 = ax.twinx()
        ax2.plot(selection['H'],color='red', label = 'H' )
        ax_h , ax_l = ax.get_legend_handles_labels()
        ax_plots = dict( zip(ax_l , ax_h) )
        ax.legend( handles = ax_plots.values() , labels = ax_plots.keys() , loc = 'upper left' , fontsize = 'x-small')
        ax2.legend( loc = 'upper right' , fontsize = 'x-small')

        ax.set_ylabel( "T" , fontsize = 'large')
        ax2.set_ylabel( "H" , fontsize = 'large')
        fig.suptitle( f"Section view of section {xsection}")
        fig.savefig( img_path + f"Sections for T/v_plot_section_{xsection}.png")
        plt.close(fig)
plot_T()
