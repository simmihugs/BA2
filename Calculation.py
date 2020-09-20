import sympy
import numpy
import os #operating system
#time
from time import strftime
#plotting
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
#my classes
from SourceData import SourceData
from OpticalElementData import OpticalElementData
from Settings import Settings





class Calculation:
    def __init__(self, SourceData, OpticalElementData, Settings):
        # initialise the instances
        SourceData.calculate()
        OpticalElementData.calculate()
        #Save the instances
        self.SourceData=SourceData
        self.OpticalElementData=OpticalElementData
        self.Settings=Settings
        #Create Folder to save plots
        self.CreateFolder()
        #Lambda sampling step
        self.d_lambda_step=(SourceData.source_samplingarea[1]-SourceData.source_samplingarea[0])/(Settings.sampling_spectral_N-1)
        #x,y sampling step in accordance with FFT Restrictions
        self.d_x_step=(OpticalElementData.oe_samplingarea[0][1]-OpticalElementData.oe_samplingarea[0][0])/(2**Settings.sampling_FFT_N[0]-1)
        self.d_y_step=(OpticalElementData.oe_samplingarea[1][1]-OpticalElementData.oe_samplingarea[1][0])/(2**Settings.sampling_FFT_N[1]-1)
        #Calculate Gaussian Beams for each Wavelength
        self.calc_sampling_lambda=[] #Wavelengths sampled
        self.calc_sampling_offsets=[]
        self.calc_sampling_waistrad=[]
        self.calc_sampling_E0=[]
        self.calc_Eopt_lambda_xy=numpy.zeros(([Settings.sampling_spectral_N, 2 ** Settings.sampling_FFT_N[0], 2 ** Settings.sampling_FFT_N[1]]), dtype=complex)
        self.calc_transmission_lambda_xy=numpy.zeros(([Settings.sampling_spectral_N, 2 ** Settings.sampling_FFT_N[0], 2 ** Settings.sampling_FFT_N[1]]), dtype=complex)
        self.calc_Eres_lambda_xy = numpy.zeros(([Settings.sampling_spectral_N, 2 ** Settings.sampling_FFT_N[0], 2 ** Settings.sampling_FFT_N[1]]),dtype=complex)
        #Sample the Intensity function
        #loop lambda
        for i in range (0,Settings.sampling_spectral_N):
         #Get Lambda
         self.calc_sampling_lambda.append(SourceData.source_samplingarea[0]+self.d_lambda_step*i)
         #Offsets
         z_off=SourceData.source_curvature_radius/(1+(self.calc_sampling_lambda[i]*SourceData.source_curvature_radius/(numpy.pi*SourceData.source_beam_radius**2))**2)
         self.calc_sampling_offsets.append([0,0,z_off])
         #Waistrad
         w_0=SourceData.source_beam_radius/(1+(numpy.pi*SourceData.source_beam_radius**2/(self.calc_sampling_lambda[i]*SourceData.source_curvature_radius**2))**0.5)
         self.calc_sampling_waistrad.append((w_0))
         #E_0(lambda) Field of Wavelengths
         self.calc_sampling_E0.append(SourceData.source_spectrum_sympify.subs("x",self.calc_sampling_lambda[i]))
         #Precalculation for Gaussian E in Optical Plane
         #Wavenumber
         k=2*numpy.pi/self.calc_sampling_lambda[i]
         #z_optic-z_offset
         z=OpticalElementData.oe_coordinates[2]-self.calc_sampling_offsets[i][2]
         #calculate from x/y independent factors of Field
         F_1=self.calc_sampling_E0[i]*numpy.exp(1j*k*z)/(1+2j*z/(k*self.calc_sampling_waistrad[i]**2))
         #loop x
         for m in range(0,2**Settings.sampling_FFT_N[0]):
             #get x
             x=OpticalElementData.oe_samplingarea[0][0]+self.d_x_step*m
             #loop y
             for n in range(0,2**Settings.sampling_FFT_N[1]):
                 #get y
                 y=OpticalElementData.oe_samplingarea[1][0]+self.d_y_step*n
                 #Calculate E(x,y,z_optic) in optical plane for each wavelength
                 F_3=1*numpy.exp(-(x**2+y**2)/(self.calc_sampling_waistrad[i]**2)*(1+2j*z/(k*self.calc_sampling_waistrad[i]**2)))
                 #Test1=complex(F_1)
                 #Test2=-(x**2+y**2)/(self.calc_sampling_waistrad[i]**2)*(1+2j*z/(k*self.calc_sampling_waistrad[i]**2))
                 #Test3=numpy.exp(0)*Test1
                 F_2 = F_1*F_3
                 self.calc_Eopt_lambda_xy[i, m, n]= F_2
                 #Calculate transmission matrices
                 self.calc_transmission_lambda_xy[i,m,n]=OpticalElementData.oe_transmissionfunction_sympify.subs([["x",x],["y",y]])
         #Cut the Matrices and Calculate for each Wavelength
         cutdim=max(2**Settings.sampling_FFT_N[1],2**Settings.sampling_FFT_N[0]) if Settings.sampling_FFT_N[0] < Settings.sampling_FFT_N[1] else min(2**Settings.sampling_FFT_N[1],2**Settings.sampling_FFT_N[0])
         cutcondition=[1]*cutdim
         E=numpy.compress(cutcondition, self.calc_Eopt_lambda_xy, axis=2)[i] #Field
         T=numpy.compress(cutcondition,self.calc_transmission_lambda_xy,axis=2)[i] #Transmission
         R=E*numpy.fft.fftn(T)
         #Add to Resultplane
         self.calc_Eres_lambda_xy[i,:,:]=R
         #Create all the Intensityplots
         self.PlotIntensity(self.PrepareForIntensityPlot(self.calc_Eopt_lambda_xy,self.d_x_step,self.d_y_step,self.OpticalElementData.oe_samplingarea[0][0],self.OpticalElementData.oe_samplingarea[1][0],i))
         #Create all the Resultplots
         self.PlotIntensity(self.RecalculateCoordinatesFFT(self.PrepareForIntensityPlot(self.calc_Eres_lambda_xy,self.d_x_step,self.d_y_step,self.OpticalElementData.oe_samplingarea[0][0],self.OpticalElementData.oe_samplingarea[1][0],i)))

        # Plot Spectrum
        self.PlotSpectrum()


    def CreateFolder(self):
        self.Directory=self.Settings.parentdir+'/'+strftime("%d_%m_%Y_%H%M")
        try:
            os.mkdir(self.Directory)
            os.mkdir(self.Directory +"/Intensity_OPT/")
            os.mkdir(self.Directory +"/Intensity_RES/")
        except: #For the case that folder already exists
            print("Directory alreay exists")

    def PlotSpectrum(self):
        fig=plt.figure()
        plt.title("Spectrum")
        plt.xlabel("Wavelength(m)")
        plt.ylabel("abs(Field)")
        absE=[abs(i) for i in self.calc_sampling_E0]
        plt.plot(self.calc_sampling_lambda,absE)
        try:
            plt.savefig(self.Directory+"/Spectrum.png")
        except:
            self.CreateFolder()
            plt.savefig(self.Directory)
    def PrepareForIntensityPlot(self,E,dx,dy,x_start,y_start,wavelength):
        x,y,z=[],[],[]
        #loop x
        for i in range(E.shape[1]):
            #loop y
            for j in range(E.shape[2]):
                x.append(x_start+i*dx)
                y.append(y_start+j*dy)
                z.append(numpy.real(E[wavelength,i,j]*numpy.conjugate(E[wavelength,i,j])))
        #x,y,intensity
        return [x,y,z,wavelength,True]

    def PlotIntensity(self,xyzlambda_array):
        x,y,z,l,OptOrRes=xyzlambda_array
        z=numpy.array(z)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_title("Intensity for Wavelenghth"+ str(l))
        ax.set_xlabel('x-axis', fontweight='bold')
        ax.set_ylabel('y-axis', fontweight='bold')
        ax.set_zlabel('Intensity', fontweight='bold')
        ax.plot_trisurf(x, y, z,cmap=cm.jet)
        path=self.Directory +("/Intensity_OPT/" if OptOrRes else "/Intensity_RES/")+str(self.calc_sampling_lambda[l]*10**9)+"nm.png"
        try:
            plt.savefig(path)
        except:
            self.CreateFolder()
            plt.savefig(path)

    def RecalculateCoordinatesFFT(self,xyzlambda_array):
        x,y,z,l,OptOrRes=xyzlambda_array
        dist=self.Settings.image_coordinates[2]-self.OpticalElementData.oe_coordinates[2]
        la=self.calc_sampling_lambda[l]
        for i in range(len(x)):
            x[i] = -dist * la* x[i]
        for k in range(len(y)):
            y[k]= - dist * la * y[k]
        return [x,y,z,l,False]