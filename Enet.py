import numpy as np 
import pandas as pd 

class Eneter() : 
    #Define the Enet with starting paramaters 
    def __init__( self, rate=0.01, iterr=100, l1_p=0.5, l2_p=0.5) : 
        print("New Model: ",self)
        self.rate = rate 
        self.iterr = iterr
        self.l1_p = l1_p 
        self.l2_p = l2_p
        

    #Fit data to Enet regression model    
    def fit( self, X, Y ) :   
        self.games, self.features = X.shape 
        print("Now fitting",self.games,"games with",self.features,"features.")
        self.weight = np.zeros( self.features ) 
        self.b = 0
        self.X = X 
        self.Y = Y 
        for i in range( self.iterr ) : 
            self.update_weights() 
        return self
    

    #helper function for fit(), adjusts weights within Enet
    def update_weights( self ) : 
        Y_pred = self.predict( self.X ) 
        weight_new = np.zeros( self.features ) 

        for feature in range( self.features ) :   
            if self.weight[feature] > 0 :   
                weight_new[feature] = ( - ( 2 * ( self.X[:,feature] ).dot( self.Y - Y_pred ) ) + self.l1_p + 2 * self.l2_p * self.weight[feature] ) / self.games 
            else : 
                weight_new[feature] = ( - ( 2 * ( self.X[:,feature] ).dot( self.Y - Y_pred ) )  - self.l1_p + 2 * self.l2_p * self.weight[feature] ) / self.games 
  
        b_new = - 2 * np.sum( self.Y - Y_pred ) / self.games  
        self.weight = self.weight - self.rate * weight_new 
        self.b = self.b - self.rate * b_new 

        return self

      
    # Predict input
    def predict( self, X_pred ) : 
         return X_pred.dot( self.weight ) + self.b 