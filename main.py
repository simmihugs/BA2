from Calculation import Calculation
from SourceData import SourceData
from OpticalElementData import OpticalElementData
from Settings import Settings

sd=SourceData()
sd.source_samplingarea=[400e-9,720e-9] #400 to 720 nm
oe=OpticalElementData()
se=Settings()
calc=Calculation(sd,oe,se)
calc.Calculation()
calc.Plot_All_SaveAll()