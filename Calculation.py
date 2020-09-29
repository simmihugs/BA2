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
        #x,y sampling step in accordance with FFT Restrictions in Resultplane
        self.d_x_step_RES=(self.Settings.image_samplingarea[0][1]-self.Settings.image_samplingarea[0][0])/(2**Settings.sampling_FFT_N[0]-1)
        self.d_y_step_RES=(self.Settings.image_samplingarea[1][1]-self.Settings.image_samplingarea[1][0])/(2**Settings.sampling_FFT_N[1]-1)
        self.d_x_step_OPT=(self.OpticalElementData.oe_samplingarea[0][1]-OpticalElementData.oe_samplingarea[0][0])/(2**Settings.sampling_FFT_N[0]-1)
        self.d_y_step_OPT=(self.OpticalElementData.oe_samplingarea[1][1]-self.OpticalElementData.oe_samplingarea[1][0])/(2**Settings.sampling_FFT_N[1]-1)
        #Initiate Arrays
        self.calc_sampling_lambda=[None]*self.Settings.sampling_spectral_N #Wavelengths sampled
        self.calc_sampling_offsets=[None]*self.Settings.sampling_spectral_N #Offsets
        self.calc_sampling_waistrad=[None]*self.Settings.sampling_spectral_N #Waistrads
        self.calc_sampling_E0=[None]*self.Settings.sampling_spectral_N #Field of Wavelength
        self.calc_Eopt_lambda_xy=numpy.zeros(([Settings.sampling_spectral_N, 2 ** Settings.sampling_FFT_N[0], 2 ** Settings.sampling_FFT_N[1]]), dtype=complex) #optical Field
        self.calc_transmission_lambda_xy=numpy.zeros(([Settings.sampling_spectral_N, 2 ** Settings.sampling_FFT_N[0], 2 ** Settings.sampling_FFT_N[1]]), dtype=complex)#transmission Matrix
        self.calc_Eres_lambda_xy = numpy.zeros(([Settings.sampling_spectral_N, 2 ** Settings.sampling_FFT_N[0], 2 ** Settings.sampling_FFT_N[1]]),dtype=complex) #resulting Field
        self.calc_IntensityResult=numpy.zeros([2 ** Settings.sampling_FFT_N[0], 2 ** Settings.sampling_FFT_N[1],2]) #Resulting Intensity after coherence 0=Opt 1=Res
        #Sample the Intensity function
        #loop lambda (Main calculation function)
        for i in range (0,Settings.sampling_spectral_N):
            self.Calculate_for_Wavelength(i,False)
            # Create Intensityplots
            self.Plot_Direction(i,False)
            # Create Beamplots
            self.PlotBeams(i)
        # Plot Spectrum
        self.PlotSpectrum()
        # Save Data
        self.SaveInputData()
        self.Calculate_Coherence(False)


    def CreateFolder(self):
        "Create Folders to save data. Path defined in Settings"
        self.Directory=self.Settings.parentdir+'/'+strftime("%d_%m_%Y_%H%M")
        try:
            os.mkdir(self.Directory)
            os.mkdir(self.Directory +"/Intensity_OPT/")
            os.mkdir(self.Directory +"/Intensity_RES/")
            os.mkdir(self.Directory +"/Beam Propagation/")
        except: #For the case that folder already exists
            print("Directory alreay exists")

    def PlotSpectrum(self):
        "Plot the Spectrum in sampled Interval"
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
        plt.close(fig)
    def PrepareForIntensityPlot(self,E,dx,dy,x_start,y_start,wavelength,OptorRes):
        "Prepares Plotting Data according to plane: (Fieldmatrix,x_step,y_step,x_start,y_start,wavelenght(iterator),Opt=True Res=False"
        x,y,z=[],[],[]
        #loop x
        for i in range(E.shape[1]):
            #loop y
            for j in range(E.shape[2]):
                x.append(x_start+i*dx)
                y.append(y_start+j*dy)
                z.append(numpy.real(numpy.complex(E[wavelength,i,j]*numpy.conjugate(E[wavelength,i,j]))))
        #x,y,intensity
        return [x,y,z,wavelength,OptorRes]

    def PlotIntensity(self,xyzlambda_array):
        x,y,z,l,OptOrRes=xyzlambda_array
        z=numpy.array(z)
        fig = plt.figure()
        ax = Axes3D(fig,elev=self.Settings.plotting_angles[0],azim=self.Settings.plotting_angles[1])
        ax.set_title("Intensity for Wavelenghth"+ str(l))
        ax.set_xlabel('x-axis (cm)', fontweight='bold')
        ax.set_ylabel('y-axis (cm)', fontweight='bold')
        ax.set_zlabel('Intensity', fontweight='bold')
        ax.plot_trisurf([xi*100 for xi in x], [yi*100 for yi in y], z,cmap=cm.jet)
        path=self.Directory +("/Intensity_OPT/" if OptOrRes else "/Intensity_RES/")+str(self.calc_sampling_lambda[l]*10**9)

        try:
            plt.savefig(path+"nm.png")
        except:
            self.CreateFolder()
            plt.savefig(path+"nm.png")
        plt.close(fig)
        #Also save Raw Data to file
        file = open(path+"nm.txt","w+")
        content = []
        content.append("Intensity Vectors for wavelength "+str(self.calc_sampling_lambda[l]*10**9)+"nm in "+("optical plane" if OptOrRes else "result plane"))
        content.append("\nVector Format [x,y,Intensity]")
        for i in range(len(x)):
            content.append("\n"+str([x[i],y[i],z[i]]))
        file.writelines(content)
        file.close()

    def Plot_RecalculateCoordinatesFFT(self,xyzlambda_array):
        "recalculate x,y from opt to res"
        x,y,z,l,OptOrRes=xyzlambda_array
        dist=self.Settings.image_coordinates[2]-self.OpticalElementData.oe_coordinates[2]
        la=self.calc_sampling_lambda[l]
        x=[self.RecalculateCoordinateFFT(xi,l) for xi in x]
        y = [self.RecalculateCoordinateFFT(yi,l)for yi in y]
        return [x,y,z,l,False]

    def Plot_RecalculateCoordinatesFFT_inv(self,xyzlambda_array):
        "recalculate x,y from res to opt"
        x,y,z,l,OptOrRes=xyzlambda_array
        dist=self.Settings.image_coordinates[2]-self.OpticalElementData.oe_coordinates[2]
        la=self.calc_sampling_lambda[l]
        x= [self.RecalculateCoordinateFFT_inv(xi,l) for xi in x]
        y = [self.RecalculateCoordinateFFT_inv(yi,l) for yi in y]
        return [x,y,z,l,True]

    def RecalculateCoordinateFFT(self,coordinate,wavelength):
        "recalculate the coordinate from opt to res, wavelength as iterator"
        dist=self.Settings.image_coordinates[2]-self.OpticalElementData.oe_coordinates[2]
        la=self.calc_sampling_lambda[wavelength]
        return -coordinate*dist*la

    def RecalculateCoordinateFFT_inv(self,coordinate,wavelength):
        "recalculate the coordinate from res to opt, wavelength as iterator"
        dist=self.Settings.image_coordinates[2]-self.OpticalElementData.oe_coordinates[2]
        la=self.calc_sampling_lambda[wavelength]
        return -coordinate/(dist*la)


    def PlotBeams(self,wavelength):
        "Plot beam propagation in profile"
        fig = plt.figure()
        axes= fig.add_axes([0.2,0.1,0.8,0.8]) #x is z axis, y is Radius
        zlim=[min(-0.1,self.SourceData.source_coordinates[2])*1.1,max(self.Settings.image_coordinates[2],self.OpticalElementData.oe_coordinates[2],0)*1.1]
        axes.set_xlim(zlim)
        plt.title("Beam Propagation for \u03BB="+str(round(self.calc_sampling_lambda[wavelength]*10**9,2))+" nm \n \u03C9_0="+str(round(self.calc_sampling_waistrad[wavelength]*1000,2))+"mm @z="+str(round(self.calc_sampling_offsets[wavelength][2]*100,2))+"cm")
        plt.xlabel("z-Axis (m)")
        plt.ylabel("Radius (cm)")
        plt.axvline(self.SourceData.source_coordinates[2] ,label="Source",linewidth=4,color="Red")
        #plt.text(self.SourceData.source_coordinates[2],0,"Source")
        plt.axvline(self.calc_sampling_offsets[wavelength][2], label="Waist of Beam",linestyle=":",linewidth=2)
        #plt.text(self.calc_sampling_offsets[wavelength][2],0,"Waist")
        plt.axvline(self.OpticalElementData.oe_coordinates[2], label="Optical plane",linestyle="--")
        #plt.text(self.OpticalElementData.oe_coordinates[2], 0, "Optical plane")
        plt.axvline(self.Settings.image_coordinates[2], label="Result plane",linewidth=4)
        #plt.text(self.Settings.image_coordinates[2],0,"Result plane")
        #Add hyperbolic function for the beam
        z_1 = numpy.linspace(self.SourceData.source_coordinates[2], self.OpticalElementData.oe_coordinates[2],50)
        z_2 = numpy.linspace(self.OpticalElementData.oe_coordinates[2], self.Settings.image_coordinates[2],50)
        wz=lambda z:100*(self.calc_sampling_waistrad[wavelength]*(1+((z-self.calc_sampling_offsets[wavelength][2])/((numpy.pi*self.calc_sampling_waistrad[wavelength]**2)/self.calc_sampling_lambda[wavelength]))**2)**0.5)
        radii_1=[wz(k) for k in z_1]
        radii_2=[wz(k) for k in z_2]
        #to optical plane
        plt.plot(z_1,radii_1,color="Black",linewidth=2,label="Envelope")#upper
        plt.plot(z_1,[-k for k in radii_1],color="Black",linewidth=2)#lower
        #to resultplane
        plt.plot(z_2,radii_2,color="Black",linestyle="dashdot",linewidth=2,label="theor. envelope")#upper
        plt.plot(z_2,[-k for k in radii_2],color="Black",linestyle="dashdot",linewidth=2)#lower
        axes.legend()
        #save to Folder
        path=self.Directory +"/Beam Propagation/"+str(self.calc_sampling_lambda[wavelength]*10**9)+"nm.png"
        try:
            plt.savefig(path)
        except:
            self.CreateFolder()
            plt.savefig(path)
        plt.close(fig)

    def SaveInputData(self):
        "Save the input data to file"
        file=open(self.Directory+"/InputData.txt","w+")
        content=[]
        content.append("Source @z="+str(self.SourceData.source_coordinates[2])+" meters")
        content.append("\nSpectrum function: "+self.SourceData.source_spectrum)
        content.append("\nSampled Interval: "+str(self.SourceData.source_samplingarea)+" meters")
        content.append("\nwith "+str(self.Settings.sampling_spectral_N)+" equidistant Points and a step of "+str(self.d_lambda_step)+" meters")
        content.append("\n\nGaussian Beam:")
        content.append("\nBeam radius at Source: "+str(self.SourceData.source_beam_radius) +" meters")
        content.append("\nBeam curvature Radius: " +str(self.SourceData.source_curvature_radius) + " meters")
        content.append("\n\nOptical Element @z="+str(self.OpticalElementData.oe_coordinates[2])+ " meters")
        content.append("\nSampled in x Interval of "+str(self.OpticalElementData.oe_samplingarea[0])+" meters")
        content.append("\nwith "+str(2**self.Settings.sampling_FFT_N[0])+" equidistant Points and a step of "+ str(self.d_x_step_OPT)+" meters")
        content.append("\nSampled in y Interval of "+str(self.OpticalElementData.oe_samplingarea[1])+" meters")
        content.append("\nwith "+str(2 ** self.Settings.sampling_FFT_N[1])+" equidistant Points and a step of "+str(self.d_y_step_OPT)+" meters")
        content.append("\nTransmission function: "+str(self.OpticalElementData.oe_transmissionfunction))
        content.append(("\n\nResult plane @z="+str(self.Settings.image_coordinates[2])+" meters"))
        content.append("\nSampled in x Interval of "+str(self.OpticalElementData.oe_samplingarea[0])+" meters")
        content.append("\nwith "+str(2**self.Settings.sampling_FFT_N[0])+" equidistant Points and a step of "+ str(self.d_x_step_RES)+" meters")
        content.append("\nSampled in y Interval of "+str(self.OpticalElementData.oe_samplingarea[1])+" meters")
        content.append("\nwith "+str(2 ** self.Settings.sampling_FFT_N[1])+" equidistant Points and a step of "+str(self.d_y_step_RES)+" meters")
        file.writelines(content)
        file.close()

    def Calculate_z_offset(self,wavelength):#as iterator
        "wavelength as iterator"
        #get wavelength
        l=self.calc_sampling_lambda[wavelength]
        R=self.SourceData.source_curvature_radius
        W=self.SourceData.source_beam_radius
        res=R/((1+(numpy.pi*(W**2)/(l*R))**2)**0.5)
        return res

    def Calculate_Waistrad(self,wavelength):#as iterator
        "wavelength as iterator"
        #get wavelength
        l=self.calc_sampling_lambda[wavelength]
        R=self.SourceData.source_curvature_radius
        W=self.SourceData.source_beam_radius
        res=W/((1+(l*R/(numpy.pi*(W**2)))**2)**0.5)
        return res

    def Calculate_Field(self,wavelength,x,y):#wavelength as iterator, x,y as value
        "Calculate the Field in optical plane at x,y"
        l=self.calc_sampling_lambda[wavelength]
        E_0=self.calc_sampling_E0[wavelength]
        W_0=self.calc_sampling_waistrad[wavelength]
        z = OpticalElementData.oe_coordinates[2] - self.calc_sampling_offsets[wavelength][2]
        k=(2*numpy.pi)/l
        res=0
        #if within area of optical element, otherwise set field to 0
        if (x>=self.OpticalElementData.oe_samplingarea[0][0] and x <=self.OpticalElementData.oe_samplingarea[0][1] and y>=self.OpticalElementData.oe_samplingarea[1][0] and y<=self.OpticalElementData.oe_samplingarea[1][1]):
            frac=(1/complex(1+2j*z/(k*E_0**2)))
            e1=(E_0*(numpy.exp(1j*k*z))*frac)
            e2=numpy.exp(-((x**2+y**2)*frac)/(W_0**2))
            res= complex(e1*e2)
        return res

    def Calculate_for_Wavelength(self, wavelength, SampledArea):
        "Calculates Field for Wavelength in optical plane wavelength as iterator, Sampled Area: Opt=true Res=False"
        i=wavelength
        #distance of optical and image
        dist = self.Settings.image_coordinates[2] - self.OpticalElementData.oe_coordinates[2]
        # Sample the Intensity function
        # Get Lambda
        self.calc_sampling_lambda[i]=self.SourceData.source_samplingarea[0] + self.d_lambda_step * i
        # Offsets
        self.calc_sampling_offsets[i]=[0, 0, self.Calculate_z_offset(i)]
        # Waistrad
        self.calc_sampling_waistrad[i]=self.Calculate_Waistrad(i)
        # E_0(lambda) Field of Wavelengths
        self.calc_sampling_E0[i]=self.SourceData.source_spectrum_sympify.subs("x", self.calc_sampling_lambda[i])
        # loop x
        for m in range(0, 2 ** self.Settings.sampling_FFT_N[0]):
            # loop y
            for n in range(0, 2 ** self.Settings.sampling_FFT_N[1]):
                # Calculate coordinates in optical plane
                if SampledArea==True: #Case we are already in Optical Plane
                    x,y =self.OpticalElementData.oe_samplingarea[0][0] + self.d_x_step_OPT * m,self.OpticalElementData.oe_samplingarea[1][0] + self.d_y_step_OPT * n
                elif SampledArea==False: #Case we have to recalculate the coordinates
                    x,y=self.RecalculateCoordinateFFT_inv(self.Settings.image_samplingarea[0][0] + self.d_x_step_RES * m,i), self.RecalculateCoordinateFFT_inv(self.Settings.image_samplingarea[1][0] + self.d_y_step_RES * n,i)
                # Calculate E(x,y,z_optic) in optical plane for each wavelength
                self.calc_Eopt_lambda_xy[i, m, n] = self.Calculate_Field(i, x, y)
                #Calculate transmission function
                self.calc_transmission_lambda_xy[i, m, n] = self.OpticalElementData.oe_transmissionfunction_sympify.subs([["x", x], ["y", y]])
                self.calc_Eres_lambda_xy[i,m,n]=self.calc_Eopt_lambda_xy[i, m, n]*numpy.fft.fftn(self.calc_transmission_lambda_xy[i, m, n])
        print(i)
        pass

    def Plot_Direction(self,wavelength,OptorRes):
        "Plots for wavelength (as iterator) according to sampled plane Opt=True Res=False"
        i=wavelength
        if OptorRes: #Optical Case
            # Create all the Intensityplots
            self.PlotIntensity(self.PrepareForIntensityPlot(self.calc_Eopt_lambda_xy, self.d_x_step_OPT, self.d_y_step_OPT,self.OpticalElementData.oe_samplingarea[0][0],self.OpticalElementData.oe_samplingarea[1][0], i,OptorRes))
            # Create all the Resultplots
            self.PlotIntensity(self.Plot_RecalculateCoordinatesFFT(self.PrepareForIntensityPlot(self.calc_Eres_lambda_xy, self.d_x_step_OPT, self.d_y_step_OPT,self.OpticalElementData.oe_samplingarea[0][0],self.OpticalElementData.oe_samplingarea[1][0], i,OptorRes)))
        else: #Resultcase
            # Create all the Intensityplots
            self.PlotIntensity(self.Plot_RecalculateCoordinatesFFT_inv(self.PrepareForIntensityPlot(self.calc_Eopt_lambda_xy, self.d_x_step_RES, self.d_y_step_RES,self.Settings.image_samplingarea[0][0],self.Settings.image_samplingarea[1][0], i ,OptorRes)))
            # Create all the Resultplots
            self.PlotIntensity(self.PrepareForIntensityPlot(self.calc_Eres_lambda_xy, self.d_x_step_RES, self.d_y_step_RES,self.Settings.image_samplingarea[0][0],self.Settings.image_samplingarea[1][0], i,OptorRes))

    def Calculate_Coherence(self,OptorRes):
        "Calculate coherence in given plane Opt=True Res=False "
        #needs 4 cases no coherence, just temporal, just spatial, both
        #Select according field
        E=self.calc_Eopt_lambda_xy if OptorRes else self.calc_Eres_lambda_xy
        #Have a look at temporal and spacial dimensions to make a decision
        x=[]
        y=[]
        z=[]

        for n in range(2 ** self.Settings.sampling_FFT_N[0]):#loop x
            for m in range(2 ** self.Settings.sampling_FFT_N[1]):#loop y
                for i in range(self.Settings.sampling_spectral_N):
                    # no coherence
                    self.calc_IntensityResult[n,m,int(OptorRes)]+= numpy.real(E[i,n,m]*numpy.complex.conjugate(E[i,n,m]))

                # get coordinates
                x.append(self.OpticalElementData.oe_samplingarea[0][0] + self.d_x_step_OPT * n if OptorRes else self.Settings.image_samplingarea[0][0] + self.d_x_step_RES * n)
                y.append(self.OpticalElementData.oe_samplingarea[1][0] + self.d_y_step_OPT * m  if OptorRes else self.Settings.image_samplingarea[1][0] + self.d_y_step_RES * m)
                z.append(self.calc_IntensityResult[n,m,int(OptorRes)])


        #plot and give out
        fig = plt.figure()
        ax = Axes3D(fig,elev=self.Settings.plotting_angles[0],azim=self.Settings.plotting_angles[1])
        ax.set_title("Resulting Intensity in "+"optical plane" if OptorRes else "result plane")
        ax.set_xlabel('x-axis (cm)', fontweight='bold')
        ax.set_ylabel('y-axis (cm)', fontweight='bold')
        ax.set_zlabel('Intensity', fontweight='bold')
        ax.plot_trisurf([xi*100 for xi in x], [yi*100 for yi in y], z,cmap=cm.jet)

        try:
            plt.savefig(self.Directory+"/Result.png")
        except:
            self.CreateFolder()
            plt.savefig(self.Directory+"/Result.png")
        plt.close(fig)
        #Also save Raw Data to file
        file = open(self.Directory+"/Result.txt","w+")
        content = []
        content.append("Resulting Intensity Vectors in "+("optical plane" if OptorRes else "result plane"))
        content.append("\nVector Format [x,y,Intensity]")
        for i in range(len(x)):
            content.append("\n"+str([x[i],y[i],z[i]]))
        file.writelines(content)
        file.close()

        pass