import numpy as np

class ANN:
    
    
    def __init__(self, X_train, X_test, y_train, y_test):

        self.X_train= X_train
        self.X_test= X_test
        self.y_train= y_train
        self.y_test= y_test
    
    
    def initlazition_w_b(self,node_num_list:list=None):
        # example:
        # y =x.w +b = (1,N) * (N,M) +(1,M)
        weights={}
        
        N=self.y_test[0].size#Getting the size of the input
        
        for i in range(node_num_list.size):

            M=node_num_list[i]
            
            weights['w'+str(i+1)] = np.random.random_sample((N,M))
            
            weights['b'+str(i+1)] = np.full((1,M),0)              
            
            N=M
    
        return weights
                     
        
    def activation_function(self,x,name='Sigmoid',derivative=False):#default: Sigmoid Function,derivative =False
        
        if name =='Threshold' and derivative==False:
            
            output = int(x>=0) # Threshold(x)
         
        elif name =='Threshold' and derivative==True:
            
            output = 0 # Threshold'(x)
            
        elif name =='Sigmoid' and derivative==False:
            
            output = 1 / (1 + np.exp(-x)) # s(x)
        
        elif name =='Sigmoid' and derivative==True:
            
            output = (1 / (1 + np.exp(-x))) * (1-(1 / (1 + np.exp(-x)))) # s'(x)=s(x) * (1 - s(x))
        
        elif name =='Tanh' and derivative==False:# tanh is sometimes called hyperbolic tangent
            
            output = np.sinh(x)/np.cosh(x)  # tanh(x)
        
        elif name =='Tanh' and derivative==True:
            
            output = 1 -np.power(np.sinh(x)/np.cosh(x),2) # tanh'(x) = 1- (tanh(x)^2)
            
        elif name =='ReLU' and derivative==False:# ReLU = Rectified Linear Unit
            
            output = np.maximum(0,x) #ReLU(x)
        
        elif name =='ReLU' and derivative==True:
            
            output = int(x >= 0 ) # ReLU'(x) = { 0  if x < 0,
                                  #             1  if x >= 0  }
                
        elif name =='leakyRELU' and derivative==False:
            
            output = np.maximum(0.01 * x, x) #leaky_ReLU'(x) 
            
        elif name =='leakyRELU'  and derivative==True:
            
            output = 0.01+(int(x >= 0 )*0.99) # leaky_ReLU'(x) = {  0.01   if x < 0,
                                              #                     1      if x >= 0  }
        elif name =='Swish' and derivative ==False:
            
            output = np.dot(x, 1 / (1 + np.exp(-x))) # Swish(x)
        
        elif name =='Swish' and derivative ==True:
            
            output = (1 + np.exp(-x)) + ((np.exp(-x)*x) / np.power(1 + np.exp(-x),2)) # Swish'(x)
        
        elif name =='Softmax' and derivative ==False:
            
            output ='Yapılmadı Araştır'  # Swish(x)
        
        elif name =='Softmax' and derivative ==True:
            
            output ='Yapılmadı Araştır'  # Swish'(x)
                
        return output
    

    def layer_architecture(self,node_num_list:np.array=None,activation_function_list:np.array=None): # [node_1]
        
        return  self.initlazition_w_b(node_num_list) , activation_function_list
    

    # w_b, activation_function_list = ANN.layer_architecture(node_num_list,activation_function_list)
    def feedforwad(self,w_b,activation_function_list):#bu fonksiyonu oluşturma amacım elinizdeki önemli ağırlıklar var ise bunarı direk kullanın diye
        
        a = self.X_train#input
        
        for i in range(activation_function_list.size):
            #print( a.shape,w_b['w'+str(i+1)].shape) #size check 
            
            z = np.matrix.dot(a , w_b['w'+str(i+1)] )  +  w_b['b'+str(i+1)] # Indexes for weights and bias numbers start at 1
            
            a = self.activation_function(z,name =activation_function_list[i])
            
        
        return a
