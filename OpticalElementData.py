from sympy import sympify

class OpticalElementData:
    #initialize
    def __init__(self):
        pass
    #Variables
    oe_transmissionfunction="1"
    #Coordinates of Optical Element
    oe_coordinates=[0,0,0.2]
    #Area that should be sampled (geometry) Default only x,y values needed, z for further upgrades
    oe_samplingarea=[[-0.04,0.04],[-0.04,0.04],[0,0]]
    def calculate(self):
        self.oe_transmissionfunction_sympify=sympify(self.oe_transmissionfunction)



