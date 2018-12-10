
def ReadTLEData(input_file):
    
    import numpy as np
    
    with open(input_file, 'r') as somefile:
        lines = somefile.readlines()
    
    Nr_of_TLE = int(len(lines)/2)
    line_1_nr = np.zeros(Nr_of_TLE)
    SatNr = np.zeros(Nr_of_TLE)
    InterD_LaunchYear = np.zeros(Nr_of_TLE)
    InterD_NrYear = np.zeros(Nr_of_TLE)
    EpochYear = np.zeros(Nr_of_TLE)
    EpochDay = np.zeros(Nr_of_TLE)
    DmeanMotion_by2 = np.zeros(Nr_of_TLE)
    Inclination = np.zeros(Nr_of_TLE)
    RightAscen = np.zeros(Nr_of_TLE)
    Eccentricity = np.zeros(Nr_of_TLE)
    Perigee = np.zeros(Nr_of_TLE)
    MeanAnomaly = np.zeros(Nr_of_TLE)
    MeanMotion = np.zeros(Nr_of_TLE)
    Revel = np.zeros(Nr_of_TLE)

    Input = np.zeros(Nr_of_TLE)
    print("Reading %d lines" % (len(lines)/2))
    for iter in range (0,Nr_of_TLE):
        
        Line1 = lines[iter*2]   # 1st line
        Line2 = lines[iter*2+1]   # 2nd line
        
        line_1_nr[iter] = float(Line1[0])   # print 6th character onwards
        SatNr[iter] = float(Line1[2:7])
        Classification = Line1[7]
        InterD_LaunchYear[iter] = float(Line1[9:11])
        InterD_NrYear[iter] = float(Line1[11:14])
        InterD_PofLaunch = Line1[14:17]
        EpochYear[iter] = float(Line1[18:20])
        EpochDay[iter] = float(Line1[20:32])
        DmeanMotion_by2[iter] = float(Line1[33:43])
        DDmeanMotion_by6 = Line1[44:52]
        Bstar = Line1[53:61]
        NrZero = Line1[62]
        ElementSet = Line1[63:68]
        CheckSum = Line1[68]
        
        Input[iter] = float(Line1[18:32])
        
        line_2_nr = Line2[0]
        SatNr_2 =  Line2[2:7]
        Inclination[iter] = float(Line2[8:16])
        RightAscen[iter] = float(Line2[17:25])
        Eccentricity[iter] = float(Line2[26:33])
        Perigee[iter] = float(Line2[34:42])
        MeanAnomaly[iter] = float(Line2[43:51])
        MeanMotion[iter] = float(Line2[52:63])
        Revel[iter] = float(Line2[64:68])
        CheckSum2 = Line2[68]

    del Line1; del Line2; del CheckSum; del CheckSum2; del Bstar; del Classification; del DDmeanMotion_by6;
    del ElementSet; del InterD_PofLaunch; del NrZero; del Nr_of_TLE; del SatNr; del SatNr_2;
    del line_1_nr; del line_2_nr; del iter;
    Input = Input.reshape(Input.shape[0],1)
    Inclination = Inclination.reshape(Input.shape[0],1)
    RightAscen = RightAscen.reshape(Input.shape[0],1)
    Eccentricity = Eccentricity.reshape(Input.shape[0],1)
    Perigee = Perigee.reshape(Input.shape[0],1)
    MeanAnomaly = MeanAnomaly.reshape(Input.shape[0],1)
    MeanMotion = MeanMotion.reshape(Input.shape[0],1)
    Revel = Revel.reshape(Input.shape[0],1)


    return(Input,Inclination,RightAscen,Eccentricity,Perigee,MeanAnomaly,MeanMotion,Revel)
