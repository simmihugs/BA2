from sympy import sympify

class SourceData:
    #initialize
    def __init__(self):
        pass
    #Variables
    #Spectral function
    source_spectrum="10000*sin(10000000*x)+1j*10000*cos(10000000*x)" #x means lambda
    #Samplingarea of spectral function (interval)
    source_samplingarea=[1,10]
    #Beam Radius of source
    source_beam_radius=0.04
    #Radius of Curvature
    source_curvature_radius=100
    #coordinates (defaults to 0,0,0), needed for further upgrades
    source_coordinates=[0,0,0]

    def calculate(self):
        #String to calculatable function
        self.source_spectrum_sympify=sympify(self.source_spectrum)







