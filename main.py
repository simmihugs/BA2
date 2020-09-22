from Calculation import Calculation
from SourceData import SourceData
from OpticalElementData import OpticalElementData
from Settings import Settings

sd=SourceData()
sd.source_samplingarea=[400e-9,720e-9] #400 to 720 nm
oe=OpticalElementData()
se=Settings()
calc=Calculation(sd,oe,se)
#calc.PlotSpectrum()
#calc.PlotIntensity(calc.PrepareForIntensityPlot(calc.calc_Eopt_lambda_xy,calc.d_x_step,calc.d_y_step,calc.OpticalElementData.oe_samplingarea[0][0],calc.OpticalElementData.oe_samplingarea[1][0],0))
#calc.PlotIntensity(calc.RecalculateCoordinatesFFT(calc.PrepareForIntensityPlot(calc.calc_Eres_lambda_xy,calc.d_x_step,calc.d_y_step,calc.OpticalElementData.oe_samplingarea[0][0],calc.OpticalElementData.oe_samplingarea[1][0],0)))
calc.PlotBeams(0)