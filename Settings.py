class Settings:
    def __init__(self):
        self.parentdir='C:/Users/Shadow/Desktop/Bachelorarbeit'
    #FFT needs 2^N in each coordinate, Default 0 in Z
        self.sampling_FFT_N=[4,4,0]
        self.sampling_spectral_N=8
        self.image_coordinates=[0,0,0.9]
        self.image_restriction=[[-0.3,0.3],[-0.3,0.3],0]
