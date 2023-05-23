# Guessing true stochastic distribution of parameters
import random
import numpy 
import matplotlib.pyplot as plt
import statsmodels.api as sm # quite complicated for us, maybe later
from scipy import stats
from scipy.stats import beta

data_path 	= 	r"Data/"
img_path  	=	r"Images/"

SIZE 	= 1000
rows 	= list(range(0,SIZE,1))
x 		= [i/SIZE for i in rows]
norm_dist = [ random.gauss() for i in range(SIZE) ] 
def Beta( alpha_  , beta_ ):
	values_beta = list()
	for i in range(SIZE):
		values_beta.append( random.betavariate(alpha_,beta_) )
	fig , ax = plt.subplots(2,3,figsize=(16,10))
	mean_beta , var_beta , skew_beta , kurt_beta = beta.stats( alpha_ , beta_ , moments = 'mvsk' )

	ax[0,0].scatter(rows,values_beta)
	ax[0,0].set_title("Scatter Plot by index")

	ax[0,1].hist(values_beta , bins=50 )
	ax[0,1].set_title( "Histogram" )

	ax[0,2].plot(x,beta.pdf(x,alpha_,beta_))
	ax[0,2].set_title( "PDF")

	ax[1,0].boxplot( values_beta , vert = True )
	ax[1,0].set_title( "Box plot of random variables")

	
	ax[1,1].scatter( norm_dist , values_beta )
	const_max = max( [*norm_dist , *values_beta ] )
	const_min = min( [*norm_dist , *values_beta ] )
	ax[1,1].set_xlim(const_min,const_max)
	ax[1,1].set_ylim(const_min,const_max)
	ax[1,1].set_title( "Scatter Plot of Random Beta vs Random Normal")
	ax[1,1].set_xlabel( "normal" )
	ax[1,1].set_ylabel( "beta" )

	stats.probplot( sorted( values_beta ) , dist = 'norm' , plot=ax[1,2])

	fig.suptitle( f"Beta distribution with alpha: {alpha_} , beta: {beta_}" )
	fig.savefig( img_path + f"beta a - {alpha_} , b - {beta_}.png")

	stats.probplot
Beta( 0.1 , 10 )