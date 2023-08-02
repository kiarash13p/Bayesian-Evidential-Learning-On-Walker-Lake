import csv,re,random
import numpy
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import tempfile

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

def sampling( num:int ) -> list :
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
	return finallist
temp.write( "Start Sampling")
samples 	= 	sampling( 300 )
temp.write( "Sampling Completed")
V 			=	[samples[i][0] for i in range(len(samples))]
W 			=	[samples[i][1] for i in range(len(samples))]
T 			=	[samples[i][2] for i in range(len(samples))]
H 			=	[samples[i][3] for i in range(len(samples))]
# Sample run of BEL
temp.write( "Running BEL")
if __name__ == '__main__' : # we are using this statement due to the error of freezing in pymc package
	with pm.Model() as model:
		# Considering linear correlation
		# Define priors
		h_sigma 	=	pm.HalfCauchy( 'h_sigma' , beta = 10 , testval = 1.0 )
		v_coeff		=	pm.Normal( "v_coeff" , mu = 1 , sigma = 20 )
		w_coeff 	=	pm.Normal( "w_coeff" , mu = 1 , sigma = 20 )
		intercept	=	pm.Normal( 'intercept' , 0 , sigma=30 )
		
		# mu here is the expected outcome
		mu 			=	v_coeff * V + w_coeff * W + intercept
		outcome		=	pm.Normal( "h" , mu = mu , sigma = h_sigma , observed = H )

		# draw n posterior samples
		sample 		=	pm.sample(40)
	fig 		=	plt.figure( figsize = (22,10) , layout = 'constrained' , label = "Results")
	subfigs		=	fig.subfigures( 1 , 2 )
	axleft 		=	subfigs[0].subplots(4,1)
	axright		=	subfigs[1].subplots(4,2)
	az.plot_posterior( sample  , ax = axleft )
	az.plot_trace( sample  , axes = axright )
	fig.savefig( img_path + "BEL results.png")
	print("\n**** Modelling is completed! ****\n")
temp.close()