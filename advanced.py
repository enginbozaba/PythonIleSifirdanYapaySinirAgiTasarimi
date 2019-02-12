# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

# %%

def deneme(*args):    # deneme(1,2,3,"ali","engin")
    print(args)       # (1, 2, 3, 'ali', 'engin')

def deneme2(**kwargs):# deneme2(a=1,b=11)
    print(kwargs)     #{'b': 11, 'a': 1}

# %% PracticalFunctions:lambda,map,filter,reduce

arr1=np.arange(0,10,2)#array([0, 2, 4, 6, 8])

arr2=np.linspace(0,5,num=np.size(arr1))#array([ 0.,1.25,2.5,3.75,5.])

func = lambda x,y:x*y

result=list(map(func,arr1,arr2)) 

print(result)#[0.0, 2.5, 10.0, 22.5, 40.0]   

filtre = lambda x:x<=10

result=list(filter(filtre,result)) 

print(result)#[0.0, 2.5, 10.0]   


tek_satır=list(list(filter(filtre,list(map(func,arr1,arr2)))))

print(tek_satır)


from functools import reduce

product = reduce((lambda x, y: x * y), [1, 2, 3, 4])

print(product) # 24

# %% Scope: global,enclosing,local

my_string = "Global"
#Global

def my_func():
    
    my_string = "Enclosing"
    #Enclosing
    
    def my_func_2():
        
        #Local
        my_string = "Local"
        print(my_string)
        
    my_func_2()

# %% Decorators
import time
def zaman_hesapla(func_name):
    
    def wrapper(arg1,arg2):
        
        start = time.time()
        result =  func_name(arg1,arg2)
        finish =  time.time()
        print(func_name.__name__ + " " + str(finish-start) + " saniye sürdü.")
        
        return result
    
    return wrapper

@zaman_hesapla
def kare_hesapla(arg1,arg2):
    print("Sonuç : ",arg1*arg2)
        
kare_hesapla(10,10)
kare_hesapla(10000,10000)
kare_hesapla(1000000,1000000)
kare_hesapla(1008756000000,100000000000000000)

# example 2-----------------------------------------------------------------

def decorator_function(func):
    
    def wrapper_function():
        
        print("wrapper started")
        
        func()
        
        print("wrapper stopped")
    
    return wrapper_function


@decorator_function
def func_new(): 
    print("hello world")

# func_new == example_function = decorator_function(func_new)
func_new()
#wrapper started
#hello world
#wrapper stopped



# %% if __name__ == '__main__'

# ::::: yoda.py ::::
#def func_direct():
#	print("yoda direct")
#
#def func_imported():
#	print("yoda imported")
#
#
#if __name__ == '__main__':
#	func_direct()
#else:
#    func_imported()

# ::::: new.py ::::

#import yoda
#
#print("anakin")


# %%

