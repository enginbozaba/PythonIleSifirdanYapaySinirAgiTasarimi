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
        
        N=self.y_test[0].size #Getting the size of the input
        
        for i in range(node_num_list.size):

            M=node_num_list[i]
            
            weights['w'+str(i+1)] = np.random.random_sample((N,M))
            
            weights['b'+str(i+1)] = np.full((1,M),0)              
            
            N=M
    
        return weights
                     
        
    def activation_function(self,x,name='Sigmoid',derivative=False):#default: Sigmoid Function,derivative =False
        
        if name =='Threshold' and derivative==False:
            
            output = (x>=0).astype(np.int) # Threshold(x)
         
        elif name =='Threshold' and derivative==True:
            
            output = np.full((x.shape),0) # Threshold'(x)
            
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
            
            output = (x >= 0 ).astype(np.int) # ReLU'(x) = { 0  if x < 0,
                                              #             1  if x >= 0  }
                
        elif name =='leakyRELU' and derivative==False:
            
            output = np.maximum(0.01 * x, x) #leaky_ReLU'(x) 
            
        elif name =='leakyRELU'  and derivative==True:
            
            output = 0.01+((x >= 0 ).astype(np.int)*0.99) # leaky_ReLU'(x) = {  0.01   if x < 0,
                                                          #                     1      if x >= 0  }
        elif name =='Swish' and derivative ==False:
            
            output = x * 1 / (1 + np.exp(-x)) # Swish(x)
        
        elif name =='Swish' and derivative ==True:
            
            output = (1 + np.exp(-x)) + ((np.exp(-x)*x) / np.power(1 + np.exp(-x),2)) # Swish'(x)
        
        elif name =='Softmax' and derivative ==False:
            
            output = np.exp(x)/np.sum(np.exp(x))  # Swish(x)
        
        elif name =='Softmax' and derivative ==True: 
            
            #https://medium.com/@enginbozaba/softmax-fonksi%CC%87yonun-t%C3%BCrevi%CC%87-ve-matri%CC%87s-%C3%A7%C3%B6z%C3%BCm%C3%BC-b0197d1d019e            
            
            x_full=np.full((x.size,x.size),x).T
            e_x_full = np.exp(x_full)
            np.fill_diagonal(e_x_full, 0)

            numerator=np.matrix.dot(e_x_full , np.exp(x.T) )
            denominator = np.power(np.sum(np.exp(x.T)),2)
            output = numerator /denominator


                
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
    
    def cost_function(self,y_test,y_pred,name='mean_squared_error',derivative=False):
        
        if name == 'mean_squared_error' and derivative==False:

            output = np.mean(0.5*np.power((y_test - y_pred),2),axis=0) # Boyut : 1 x En_Son_Katmandaki_Nöron
                #http://www.derinogrenme.com/2018/06/28/geri-yayilim-algoritmasina-matematiksel-yaklasim/

        elif name == 'mean_squared_error' and derivative==True:
            
            output = np.mean(-(y_test - y_pred),axis=0) # Boyut : 1 x En_Son_Katmandaki_Nöron
            
        elif name == 'binary_cross_entropy' and derivative==False:
            
            output = -1* np.mean(y_test*np.log(y_pred) + (1-y_test)*np.log(1-y_pred)) # -(1/n) ((y_test*log(y_pred) + (1 - y_test)*log(1 - y_pred)))
        
        elif name == 'binary_cross_entropy' and derivative==False:
            
             output = - 1* ((y_test/y_pred)+(1-y_test/1-y_pred))
                #https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c
                #Notation : https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
                #Derivative :https://math.stackexchange.com/questions/2503428/derivative-of-binary-cross-entropy-why-are-my-signs-not-right
            
        # ilerleyen zamanlarda burayı zenginleştir.
        
        return output
