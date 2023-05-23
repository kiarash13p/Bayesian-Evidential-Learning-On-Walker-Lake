import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os.path,re

data_path 	= 	r"Data/"
img_path  	=	r"Images/"

# Never used, just as a draft
def weathered( t_var ):
    return t_var >= 0.15

# Never used, just as a draft
def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

# Never used, just as a draft
def random_weathering( ratio ):
    true_w = translate( ratio , 0.15 , 0.25 , 0 , 1 )
    d = random.gauss( true_w , 0.1 )
    return min( max( 0 , d ) , 1 )

def weathering_rate( t_var:float ) -> float:
    if not Weathered(t_var): return 0.0
    return Translator(t_var)

# Making a random function to create a distribution based on input,W
# We can assume uniform distribution can be applied on W -> [0,1]
W_MAX = 1.0
W_MIN = 0.001
NUM_SIM = 60
def random_weathering( w:float , range_:float=0.15 , min_:float=W_MIN , max_:float=W_MAX) -> list:
    return [0.0] * NUM_SIM if w == 0.0 else [random.uniform( max( min_ , w-range_) , min( max_ , w+range_) )for t in range(0,NUM_SIM)]


# Reading Data, as the size is quire large, we use constant address
main = pd.read_csv( "C:/Users/kpashna/surfdrive/Projects/Aafje Houben/SGeMS Project/Exhaustive 14-05-2023" )
print( f"Head of data is as follows:\n{main.head()}\n-----\n	Shape of the data is {main.shape}")

# Applying trim to column names to prevent syntax errors
main.rename(columns=str.strip , inplace=True)

# Making some plots based on variance of T simulations
fig = plt.figure( figsize = (12,8) )
ax = fig.subplots(2,2)
ax[0,0].hist(main['T_Variance'] , bins = 40)
ax[0,0].set_title( 'Histogram' )
ax[0,1].scatter( main['T_Etype'] , main["T_Variance"] ) # 0.15 for threshold of W is fine :D
ax[0,1].set_title( 'Etype vs Variance of T' )
ax[1,0].scatter( main['U'] , main['T_Variance'] )
ax[1,0].set_title( 'U vs Variance of T' )
ax[1,1].scatter( main['V_Etype'] , main['T_Variance'] )
ax[1,1].set_title( 'Etype of V vs Variance of T' )
fig.suptitle("Charts of T_Variance")
fig.savefig( img_path + "T_Variance plots.png" , dpi = 300 )

fig_2 = plt.figure( figsize = (max(main['X'])/30,max(main['Y'])/30) )
ax_2 = fig_2.subplots(1,1)
ax_2.scatter( main['X'] , main['Y'] , s=1 , c=main['T_Variance'] , cmap='jet' )
ax_2.set_title("Main Exhaustive Grid with Variance of T")
fig_2.savefig( img_path + "T_Variance in Grid")

# Making a function to create variable W based on T_Variance
T_VAR_MAX = max(main['T_Variance'])
T_VAR_MIN = 0.15
Weathered = lambda W:W >= T_VAR_MIN
Translator = lambda val: (val - T_VAR_MIN)/(T_VAR_MAX-T_VAR_MIN)*(W_MAX-W_MIN) + W_MIN if val else None

# Make a synthetic example to check translation
T = np.linspace(0.0,T_VAR_MAX)
W = list( map(weathering_rate,T) )
fig_3 = plt.figure( figsize = (12,8) )
ax_3 = fig_3.subplots(1,1)
ax_3.plot(T,W)
ax_3.set_title("T vs W")
fig_3.savefig( img_path + "Sample conversion of T to W")

# Assigning simulation to every point
columns = [f"W__sim{i}" for i in range(1,NUM_SIM+1)]
Weathers = list()
Weathers_etype = list()
for row in main.itertuples():
    w = random_weathering( weathering_rate( row.T_Variance ) )
    wmean = np.mean(w)
    Weathers.append(w)
    Weathers.append(wmean)
    for i in range(0,NUM_SIM):
        main.loc[row.Index,columns[i]] = w[i]
    main.loc[row.Index,'W_etype'] = wmean
fig_w = plt.figure( figsize = (15,10) )
fig_w.suptitle("Plotting of new variable W based on Variance of T")
ax_w = fig_w.subplots(1,2)
ax_w[0].scatter( main['X'] , main['Y'] , s=1 , c=main['T_Variance'] , cmap='jet' )
ax_w[0].set_title("Variance of T")
ax_w[1].scatter( main['X'] , main['Y'] , s=1 , c=main['W_etype'] , cmap='jet' )
ax_w[1].set_title("Etype of W")
fig_w.savefig( img_path + "W and T")

# Lets have a review on W
fig_4 = plt.figure( figsize = (15,10) )
ax_4 = fig_4.subplots(2,2)
ax_4[0,0].hist(main['W_etype'] , bins = 40)
ax_4[0,1].scatter( main['W_etype'] , main["T_Variance"] )
ax_4[1,0].scatter( main['W_etype'] , main['T_Etype'] )
ax_4[1,1].scatter( main['W_etype'] , main['U'] )
fig_4.suptitle( "Plots for W")
fig_4.savefig( img_path + "Plots of W")

# Now we want to create H based on U
# Simple transformation from [0-max(U)] upto [20,55]
U_VAR_MAX = max(main['U'])
U_VAR_MIN = min(main['U'])
H_MAX = 55.0
H_MIN = 20.0
Translator_H = lambda val: (val - U_VAR_MIN)/(U_VAR_MAX-U_VAR_MIN)*(H_MAX-H_MIN) + H_MIN
main['H'] = Translator_H( main['U'] )

# Making some plots
fig_h = plt.figure( figsize = (15,10) )
ax_h = fig_h.subplots(2,2)
ax_h[0,0].hist(main['H'] , bins = 40)
ax_h[0,1].scatter( main['H'] , main["W_etype"] )
ax_h[1,0].scatter( main['H'] , main['T_Etype'] )
ax_h[1,1].scatter( main['H'] , main['V_Etype'] )
fig_h.suptitle( "Plots of H")
fig_h.savefig( img_path + "Plots of H" )

print(f"columns are:\n{main.columns.values}")
# Saving dataframe for further use in BEL
# Dropping some columns that we wouldnt use them anymore
reg_v_old = r".*sgs___real.*"
compil_v_old = re.compile(reg_v_old)
realization_v_old_columns = list(filter(compil_v_old.match,main.columns))
# reg_v_normvals = r".*sgs___real.*"
# compil_v_normvals = re.compile(reg_v_normvals)
# realization_v_normvals_columns = list(filter(compil_v_normvals.match,main.columns))
drops = ['Z','T_IK__real0','T_Etype','V_Etype_sgs','V_Etype_sgs_backtrans','V_Etype_sgs_backtrans_cons_ex','is1_tr1','W_etype']
main.drop([*drops,*realization_v_old_columns],axis = 1).to_csv(data_path + "Data for BEL.csv")

print("---\nProcessing and saving has been completed!" )