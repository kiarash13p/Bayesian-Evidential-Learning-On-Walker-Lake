import csv,re,random
import numpy
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import tempfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, normalize
#print( az.style.available )
az.style.use("arviz-royish")

data_path 	= 	r"Data/"
img_path  	=	r"Images/"
temp_path	=	r"Temp/"

# Create a temporary file and directory to record all important milestones
temp 		=	tempfile.TemporaryFile( mode='w' , newline = '\n' , suffix = "-suff" , prefix = "pref-" , dir = temp_path )
print( f"Directory of temp: {tempfile.gettempdir()}" )
temp.write( "Running BEL" )
# Reading and arranging main frame of data
# No more use of pandas!
data_read = list()
with open(r'Data/Data for BEL.csv' , newline = '') as datafile:
    reader = csv.reader( datafile , delimiter = ',' )
    for row in reader:
        data_read.append(row)
columns = data_read[0][1:]
index   = list( i[0] for i in data_read )
index.pop(0)
index 	=	list(map(int,index))
data    = dict()
for idx,col in enumerate(columns):
	try:
		data[col] = [ float(i[idx+1]) for i in data_read[1:] ]
	except Exception as e:
		data[col] = [ i[idx+1] for i in data_read[1:] ]
assert( columns == list( data.keys() ) )
#print(f"Features we are working with are as follow:\n{data.keys()}")

# Finding columns
reg_v = r".*V__real.*_back.*"
compile_v       =   re.compile( reg_v )
list_v          =   list( filter( compile_v.match , columns ) )

reg_t = r".*T___real.*"
compile_t       =   re.compile( reg_t )
list_t          =   list( filter( compile_t.match , columns ) )

reg_w = r".*W__sim.*"
compile_w       =   re.compile( reg_w )
list_w          =   list( filter( compile_w.match , columns ) )

# Function for sampling
#print( data['H'] , numpy.mean( data['H'] ) , numpy.var( data['H'] ) )
data['H'] = list(numpy.log10( data['H']))
print( "***H:\n" , data['H'] , "\n*** Mean and Var:", numpy.mean( data['H'] ) , numpy.var( data['H'] ) , " ***\n")
def sampling( num:int , percent:float ) -> [list,list] :
	rep 		= num // 78000
	remains 	= num % 78000
	finallist = list()
	for i in range(rep):
		indexes		=	random.sample( index , 78000 )
		for idx in indexes:
			temp_ = list()
			for colv in list_v:
				temp_.append(data[colv][idx]) 
			v_selected = random.choice( temp_ )
			temp_ = list()
			for colw in list_w:
				temp_.append(data[colw][idx]) 
			w_selected = random.choice( temp_ )
			temp_ = list()
			for colt in list_t:
				temp_.append(data[colt][idx]) 
			t_selected = random.choice( temp_ )
			h_selected = data['H'][idx]
			#print( f"For index {idx}, we selected \tV:{v_selected}\tW:{w_selected}\tT:{t_selected}\tH:{h_selected}\tamong simulations")
			finallist.append( (v_selected , w_selected , t_selected , h_selected ) )
			del temp_
	indexes		=	random.sample( index , remains )
	for idx in indexes:
		temp_ = list()
		for colv in list_v:
			temp_.append(data[colv][idx]) 
		v_selected = random.choice( temp_ )
		temp_ = list()
		for colw in list_w:
			temp_.append(data[colw][idx]) 
		w_selected = random.choice( temp_ )
		temp_ = list()
		for colt in list_t:
			temp_.append(data[colt][idx]) 
		t_selected = random.choice( temp_ )
		h_selected = data['H'][idx]
		#print( f"For index {idx}, we selected \tV:{v_selected}\tW:{w_selected}\tT:{t_selected}\tH:{h_selected}\tamong simulations")
		finallist.append( (v_selected , w_selected , t_selected , h_selected ) )
		del temp_
	train,test 	= train_test_split( finallist , test_size=percent , random_state=42 )
	return train , test 
temp.write( "Start Sampling")
train,test 	= 	sampling( 110000 , 0.01 )

V_train , V_test 			=	[train[i][0] for i in range(len(train))] , [test[i][0] for i in range(len(test))]
W_train , W_test 			=	[train[i][1] for i in range(len(train))] , [test[i][1] for i in range(len(test))]
T_train , T_test 			=	[0 if train[i][2] =='category_0' else 1 for i in range(len(train))] , [0 if test[i][2] =='category_0' else 1 for i in range(len(test))]
H_train , H_test 			=	[train[i][3] for i in range(len(train))] , [test[i][3] for i in range(len(test))]

_MAX = 1.0
_MIN = 0.0
H_MAX = 55.0
H_MIN = 20.0
Translator 		= lambda val: (val - _MIN)/(_MAX-_MIN)*(H_MAX-H_MIN) + H_MIN # 0-1 to 20-55
Translator_r	= lambda val: (val - H_MIN)/(H_MAX-H_MIN)*(_MAX-_MIN) + _MIN # 20-55 to 0-1
H_train_scaled  = [Translator_r(i) for i in H_train]
H_train_scaled_mean , H_train_scaled_var = numpy.mean(H_train_scaled) , numpy.var( H_train_scaled)

#normalized values
quantile_V 		=	QuantileTransformer( output_distribution='normal' , n_quantiles=10000)
V_normalized	=	quantile_V.fit( numpy.reshape( V_train , (len(V_train),1) )).transform( numpy.reshape( V_train , (len(V_train),1) ) )
V_normalized 	=	numpy.reshape(V_normalized,len(V_normalized))
quantile_W 		=	QuantileTransformer( output_distribution='normal' , n_quantiles=10000)
W_normalized	=	quantile_W.fit( numpy.reshape( W_train , (len(W_train),1) )).transform( numpy.reshape( W_train , (len(W_train),1) ) )
W_normalized 	=	numpy.reshape(W_normalized,len(W_normalized))
#T_normalized	=	quantile.fit_transform( numpy.reshape( T_train , (len(T_train,1)) ) )
quantile_H 		=	QuantileTransformer( output_distribution='normal' , n_quantiles=10000)
H_normalized	=	quantile_H.fit( numpy.reshape( data['H'] , (len(data['H']),1) )).transform( numpy.reshape( H_train , (len(H_train),1) ) )
H_normalized 	=	numpy.reshape(H_normalized,len(H_normalized))
H_normalized_mean , H_normalized_var = numpy.mean( H_normalized ) , numpy.var( H_normalized )
fig_h 			=	plt.figure( figsize = (22,10) , label = "Histogram of Normalized H")
ax_h 			=	fig_h.subplots(1,1)
ax_h.hist( H_normalized , bins=50 )
fig_h.savefig( img_path + "Histogram of Normalized H.png")

print( H_normalized_mean , H_normalized_var )
temp.write( "Sampling Completed")
# Sample run of BEL
temp.write( "Running BEL" )
if __name__ == '__main__' : # we are using this statement due to the error of freezing in pymc package

	with pm.Model() as model:
		# Considering linear correlation
		# Define priors
		v_coeff		=	pm.Normal( 'V' , mu = 0 , sigma = 40 )
		w_coeff 	=	pm.Normal( 'W' , mu = 0 , sigma = 40 )
		t_coeff		=	pm.Normal( 'T' , mu = 0 , sigma = 40 )
		intercept	=	pm.Normal( 'Intercept' , mu = 0 , sigma = 40 )
		h_estimate 	=	v_coeff * V_normalized + w_coeff * W_normalized + t_coeff * T_train + intercept

		#h_mean 		=	H_train_scaled_mean#Translator_r(pm.math.maximum( pm.math.minimum( h_estimate , 55.0 ) , 20.0 ))
		#h_sigma 	=	pm.math.minimum( pm.math.maximum( 0.0001 , pm.Exponential( 'σH' , lam = 1.0  ) ) , pm.math.sqrt( h_mean/(1-h_mean) )-0.01 )
		#outcome	=	pm.Normal( "H" , mu = mu , sigma = h_sigma , observed = H_train )
		h_sigma		=	pm.Uniform( 'σH' , lower = 0.3 , upper = 0.8)
		outcome		=	pm.Normal( "H" , mu = h_estimate , sigma = h_sigma , observed= H_normalized)
		print( f"-\tModel Deterministics are as follow:\n{model.deterministics}")
		# draw n posterior train
		#prior 		=	pm.sample_prior_predictive( 2000 )
		trace 		=	pm.sample(6000 , tune=3000 , chains = 8 , return_inferencedata= True) # instead of 4 chains
		posterior 	=	pm.sample_posterior_predictive( trace  )
	fig 		=	plt.figure( figsize = (22,10) , layout = 'constrained' , label = "Results")
	subfigs		=	fig.subfigures( 1 , 2 )
	axleft 		=	subfigs[0].subplots(5,1)
	subfigs[0].suptitle( "Posterior Plot")
	axright		=	subfigs[1].subplots(5,2)
	subfigs[1].suptitle( "Trace Plot")
	az.plot_posterior( trace  , ax = axleft  )
	az.plot_trace( trace  , axes = axright )#, legend=True )
	fig.savefig( img_path + "BEL results.png")

	fig_ppc 	=	plt.figure( figsize = (22,10) , layout = 'constrained' , label = "Results")
	ax_ppc		=	fig_ppc.subplots(1,1)
	print(posterior.posterior_predictive)
	az.plot_ppc( posterior , num_pp_samples= 4000 , ax=[ax_ppc] )
	fig_ppc.savefig( img_path + "BEL ppc plot.png")
	print("\n**** Modelling is completed! ****\n")
temp.close()