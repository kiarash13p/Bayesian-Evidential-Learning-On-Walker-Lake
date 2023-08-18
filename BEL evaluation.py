import csv,re,random
import numpy
from sklearn.preprocessing import QuantileTransformer , normalize , Normalizer
import matplotlib.pyplot as plt
import tempfile
from sklearn.model_selection import train_test_split
#print( az.style.available )

data_path 	= 	r"Data/"
img_path  	=	r"Images/"

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
train,test 	= 	sampling( 75000 , 0.2 )

V_train , V_test 			=	[train[i][0] for i in range(len(train))] , [test[i][0] for i in range(len(test))]
W_train , W_test 			=	[train[i][1] for i in range(len(train))] , [test[i][1] for i in range(len(test))]
T_train , T_test 			=	[0 if train[i][2] =='category_0' else 1 for i in range(len(train))] , [0 if test[i][2] =='category_0' else 1 for i in range(len(test))]
H_train , H_test 			=	[train[i][3] for i in range(len(train))] , [test[i][3] for i in range(len(test))]

fig = plt.figure()
ax		=	fig.subplots(1,1)
# quantile 	=	QuantileTransformer( output_distribution='normal' , n_quantiles=160 )
# H_train 	=	numpy.reshape( H_train , (len(H_train),1))

normalizer_h 		=	QuantileTransformer( output_distribution='normal' , random_state=60 , n_quantiles=10000).fit( numpy.reshape( data['H'] , (len(data['H']),1) ))

data_transformed	=	normalizer_h.transform( numpy.reshape( H_train , (len(H_train),1) ) )
data_transformed 	=	numpy.reshape(data_transformed,len(data_transformed))
print( data_transformed , numpy.mean( data_transformed ) , numpy.var( data_transformed ))
#print( numpy.reshape(data_transformed,len(data_transformed)) )
ax.hist( data_transformed , bins=50)
plt.show()