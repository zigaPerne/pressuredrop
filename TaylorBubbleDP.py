# -*- coding: utf-8 -*-
"""
Created on Tue May 24 15:17:09 2022

@author: kren
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


times_all = np.loadtxt("./times.txt", delimiter = ",")
for i in range(0, len(times_all)):
    for j in range(0, len(times_all[i]), 2):
        times_all[i][j] = times_all[i][j] + 5
    for k in range(1, len(times_all[i]), 2):
        times_all[i][k] = times_all[i][k] - 5

Times1a =   times_all[0] 
Times2a = 	times_all[1] 
Times3a = 	times_all[2] 
Times4a = 	times_all[3] 
Times5a = 	times_all[4] 
Times1b = 	times_all[5] 
Times2b = 	times_all[6] 
Times3b = 	times_all[7]
Times4b = 	times_all[8]
Times5b = 	times_all[9]
Times6b = 	times_all[10]
Times7b = 	times_all[11]
Times8b = 	times_all[12]
Times9b = 	times_all[13]
Times23_1 = 	times_all[14]
Times23_2 = 	times_all[15]
Times24_1 = 	times_all[16]
Times24_2 = 	times_all[17]
Times24_3 = 	times_all[18]
Times24_4 = 	times_all[19]
Times24_5 = 	times_all[20]
Times24_6 = 	times_all[21]
Times27_1 = 	times_all[22]
Times27_2 = 	times_all[23]
Times27_3 = 	times_all[24]
Times27_4 = 	times_all[25]
Times27_5 = 	times_all[26]
Times27_6 = 	times_all[27]
Times27_7 = 	times_all[28]
Times30_1 = 	times_all[29]
Times30_2 = 	times_all[30]
Times30_3 = 	times_all[31]
Times31_1 = 	times_all[32]
Times31_2 = 	times_all[33]
Times31_3 = 	times_all[34]
Times31_4 = 	times_all[35]
Times31_5 = 	times_all[36]
Times31_6 = 	times_all[37]
Times68_1 = 	times_all[38]
Times68_2 = 	times_all[39]
Times68_3 = 	times_all[40]
Times68_4 = 	times_all[41]
Times68_5 = 	times_all[42]
Times68_6 = 	times_all[43]
Times69_1 = 	times_all[44]
Times_610_1 = 	times_all[45]
Times_610_2 = 	times_all[46]
Times_610_3 = 	times_all[47]
Times_610_4 = 	times_all[48]
Times_610_5 = 	times_all[49]
Times_610_6 = 	times_all[50]
Times_610_7 = 	times_all[51]
Times_610_8 = 	times_all[52]
Times_610_9 = 	times_all[53]
Times_610_10 = 	times_all[54]
Times_610_11 = 	times_all[55]
Times_610_12 = 	times_all[56]
Times_610_13 = 	times_all[57]
Times_610_14 = 	times_all[58]
Times_611_1 = 	times_all[59]
Times_611_2 = 	times_all[60]
Times_611_3 = 	times_all[61]
Times_611_4 = 	times_all[62]
Times_611_5 = 	times_all[63]
Times_611_6 = 	times_all[64]
Times_611_7 = 	times_all[65]
Times_611_8 = 	times_all[66]
Times_611_9 = 	times_all[67]
Times_611_10 = 	times_all[68]
Times_611_11 = 	times_all[69]
Times_611_12 = 	times_all[70]
Times_611_13 = 	times_all[71]
Times_611_14 = 	times_all[72]
Times_611_15 = 	times_all[73]
Times_611_16 = 	times_all[74]

#corrections
Times24_1[4] = 135
Times24_4[4] = 100
Times24_5[4] = 75
Times68_6[4] = 70
Times_610_4[1] = 155
Times_610_4[2] = 225

Times23 = [Times23_1, Times23_2]
Times24 = [Times24_1, Times24_2, Times24_3, Times24_4, Times24_5, Times24_6]
Times27 = [Times27_1, Times27_2, Times27_3, Times27_4, Times27_5, Times27_6, Times27_7]
Times25 = [Times1a, Times2a, Times3a, Times4a, Times5a]
Times26 = [Times1b, Times2b, Times3b, Times4b, Times5b, Times6b, Times7b, Times8b, Times9b]
Times30 = [Times30_1, Times30_2, Times30_3]
Times31 = [Times31_1, Times31_2, Times31_3, Times31_4, Times31_5, Times31_6]
Times68 = [Times68_1, Times68_2, Times68_3, Times68_4, Times68_5, Times68_6]
Times69 = [Times69_1]
Times610 = [Times_610_1, Times_610_2, Times_610_3, Times_610_4, Times_610_5, Times_610_6, Times_610_7, Times_610_8, Times_610_9, Times_610_10, Times_610_11, Times_610_12, Times_610_13, Times_610_14]
Times611 = [Times_611_1, Times_611_2, Times_611_3, Times_611_4, Times_611_5, Times_611_6, Times_611_7, Times_611_8, Times_611_9, Times_611_10, Times_611_11, Times_611_12, Times_611_13, Times_611_14, Times_611_15, Times_611_16]

StarterA = 5
MultiplierA = 7

StarterB = 6
MultiplierB = 8

MultiplierC = 9

excel_file = "./TB1.xlsx"

df2 = pd.read_excel(excel_file, sheet_name=3)
df23 = np.array(df2)

df2 = pd.read_excel(excel_file, sheet_name=4)
df24 = np.array(df2)

df1 = pd.read_excel(excel_file, sheet_name=5)
df25 = np.array(df1)

df2 = pd.read_excel(excel_file, sheet_name=6)
df26 = np.array(df2)

df2 = pd.read_excel(excel_file, sheet_name=7)
df27 = np.array(df2)

df2 = pd.read_excel(excel_file, sheet_name=8)
df30 = np.array(df2)

df2 = pd.read_excel(excel_file, sheet_name=9)
df31 = np.array(df2)

df2 = pd.read_excel(excel_file, sheet_name=10)
df68 = np.array(df2)

df2 = pd.read_excel(excel_file, sheet_name=11)
df69 = np.array(df2)

df2 = pd.read_excel(excel_file, sheet_name=12)
df610 = np.array(df2)

df2 = pd.read_excel(excel_file, sheet_name=14)
df611 = np.array(df2)

dfa = pd.read_excel('./TB_Exp-AllData.xlsx', sheet_name = 0)
dfa = np.array(dfa)
Flowrates = dfa[1:76, 2]
Length = dfa[1:76, 4]
Length = np.array(Length)
Length_Unc = dfa[1:76, 5]

def find_intersect(MultiplierA, j, c, TimesA, df):
    a = np.where(df[:, MultiplierA*j]>TimesA[c])
    b = np.where(df[:, MultiplierA*j]<TimesA[c+1])
    intersect1 = np.intersect1d(a, b)
    return intersect1
    
def compute_avg_std(MultiplierA, TimesA, df, StarterA, counter, n):
    Means = []
    STDS = []
    Temps = []
    TempSTD = []
    Temps2 = []
    Temp2STD = []
    PresMean = []
    PresSTD = []
    PresMean2 = []
    PresSTD2 = []
    intersects = []
    for j in range(counter):
        print(j)
        MeansA = []
        STDA = []
        TempsA= []
        TempsASTD = []
        intersectsA = []
        Array1 = []
        Array2 = []
        Array3 = []
        Array4 = []
        Array5 = []
        Array6 = []
        for i in range(4):
            c = i*2
            intersect1 = find_intersect(MultiplierA, j, c, TimesA[j], df)
            Mean1 = np.mean(df[np.amin(intersect1):np.amax(intersect1), StarterA+MultiplierA*j])
            std = np.std(df[np.amin(intersect1):np.amax(intersect1), StarterA+MultiplierA*j])
            tempsaa = np.mean(df[np.amin(intersect1):np.amax(intersect1), StarterA-5+n+MultiplierA*j])
            tempsastd = np.std(df[np.amin(intersect1):np.amax(intersect1), StarterA-5+n+MultiplierA*j])
            tempsaa2 = np.mean(df[np.amin(intersect1):np.amax(intersect1), StarterA-4+n+MultiplierA*j])
            tempsastd2 = np.std(df[np.amin(intersect1):np.amax(intersect1), StarterA-4+n+MultiplierA*j])
            pressurea = np.mean(df[np.amin(intersect1):np.amax(intersect1), StarterA-3+n+MultiplierA*j])
            pressurestd = np.std(df[np.amin(intersect1):np.amax(intersect1), StarterA-3+n+MultiplierA*j])
            pressurea2 = np.mean(df[np.amin(intersect1):np.amax(intersect1), StarterA-2+n+MultiplierA*j])
            pressurestd2 = np.std(df[np.amin(intersect1):np.amax(intersect1), StarterA-2+n+MultiplierA*j])
            MeansA.append(Mean1)
            STDA.append(std)
            TempsA.append(tempsaa)
            TempsASTD.append(tempsastd)
            intersectsA.append(intersect1)
            Array1.append(tempsaa2)
            Array2.append(tempsastd2)
            Array3.append(pressurea)
            Array4.append(pressurestd)
            Array5.append(pressurea2)
            Array6.append(pressurestd2)
        intersects.append(intersectsA)
        Means.append(MeansA)
        STDS.append(STDA)
        Temps.append(TempsA)
        TempSTD.append(TempsASTD)
        Temps2.append(Array1)
        Temp2STD.append(Array2)
        PresMean.append(Array3)
        PresSTD.append(Array4)
        PresMean2.append(Array5)
        PresSTD2.append(Array6)
    return np.array(intersects), np.array(Means), np.array(STDS), np.array(Temps), np.array(TempSTD), np.array(Temps2), np.array(Temp2STD), np.array(PresMean), np.array(PresSTD), np.array(PresMean2), np.array(PresSTD2)

Array11 = []
Array22 = []
Array33 = []
Array44 = []
Array55 = []
Array66 = []
Array77 = []
Array88 = []
Array99 = []
Array100 = []

np.array(Array11)
Array22 = np.array(Array22)
Array33 = np.array(Array33)
Array44 = np.array(Array44)
Array55 = np.array(Array55)
Array66 = np.array(Array66)
Array77 = np.array(Array77)
Array88 = np.array(Array88)
Array99 = np.array(Array99)
Array100 = np.array(Array100)

intersects_array = []
intersects, Means, STDS, Temps, TempSTD, Temps2, Temp2STD, PresMean, PresSTD, PresMean2, PresSTD2 = compute_avg_std(MultiplierB, Times23, df23, StarterB, 2, 0)
intersects_array.append(intersects)
Array11 = np.append(Array11, Means)
Array22 = np.append(Array22, STDS)
Array33 = np.append(Array33, np.mean(Temps, 1))
Array44 = np.append(Array44, np.mean(TempSTD, 1))
Array55 = np.append(Array55, np.mean(Temps2, 1))
Array66 = np.append(Array66, np.mean(Temp2STD, 1))
Array77 = np.append(Array77, np.mean(PresMean, 1))
Array88 = np.append(Array88, np.mean(PresSTD, 1))
Array99 = np.append(Array99, np.mean(PresMean2, 1))
Array100 = np.append(Array100, np.mean(PresSTD2, 1))

intersects, Means, STDS, Temps, TempSTD, Temps2, Temp2STD, PresMean, PresSTD, PresMean2, PresSTD2 = compute_avg_std(MultiplierC, Times24, df24, StarterB, 6, 0)
intersects_array.append(intersects)
Array11 = np.append(Array11, Means)
Array22 = np.append(Array22, STDS)
Array33 = np.append(Array33, np.mean(Temps, 1))
Array44 = np.append(Array44, np.mean(TempSTD, 1))
Array55 = np.append(Array55, np.mean(Temps2, 1))
Array66 = np.append(Array66, np.mean(Temp2STD, 1))
Array77 = np.append(Array77, np.mean(PresMean, 1))
Array88 = np.append(Array88, np.mean(PresSTD, 1))
Array99 = np.append(Array99, np.mean(PresMean2, 1))
Array100 = np.append(Array100, np.mean(PresSTD2, 1))

intersects, Means, STDS, Temps, TempSTD, Temps2, Temp2STD, PresMean, PresSTD, PresMean2, PresSTD2 = compute_avg_std(MultiplierA, Times25, df25, StarterA, 5, 1)
intersects_array.append(intersects)
Array11 = np.append(Array11, Means)
Array22 = np.append(Array22, STDS)
Array33 = np.append(Array33, np.mean(Temps, 1))
Array44 = np.append(Array44, np.mean(TempSTD, 1))
Array55 = np.append(Array55, np.mean(Temps2, 1))
Array66 = np.append(Array66, np.mean(Temp2STD, 1))
Array77 = np.append(Array77, np.mean(PresMean, 1))
Array88 = np.append(Array88, np.mean(PresSTD, 1))
Array99 = np.append(Array99, np.mean(PresMean2, 1))
Array100 = np.append(Array100, np.mean(PresSTD2, 1))

intersects, Means, STDS, Temps, TempSTD, Temps2, Temp2STD, PresMean, PresSTD, PresMean2, PresSTD2 = compute_avg_std(MultiplierB, Times26, df26, StarterB, 9, 0)
intersects_array.append(intersects)
Array11 = np.append(Array11, Means)
Array22 = np.append(Array22, STDS)
Array33 = np.append(Array33, np.mean(Temps, 1))
Array44 = np.append(Array44, np.mean(TempSTD, 1))
Array55 = np.append(Array55, np.mean(Temps2, 1))
Array66 = np.append(Array66, np.mean(Temp2STD, 1))
Array77 = np.append(Array77, np.mean(PresMean, 1))
Array88 = np.append(Array88, np.mean(PresSTD, 1))
Array99 = np.append(Array99, np.mean(PresMean2, 1))
Array100 = np.append(Array100, np.mean(PresSTD2, 1))

intersects, Means, STDS, Temps, TempSTD, Temps2, Temp2STD, PresMean, PresSTD, PresMean2, PresSTD2 = compute_avg_std(MultiplierB, Times27, df27, StarterB, 7, 0)
intersects_array.append(intersects)
Array11 = np.append(Array11, Means)
Array22 = np.append(Array22, STDS)
Array33 = np.append(Array33, np.mean(Temps, 1))
Array44 = np.append(Array44, np.mean(TempSTD, 1))
Array55 = np.append(Array55, np.mean(Temps2, 1))
Array66 = np.append(Array66, np.mean(Temp2STD, 1))
Array77 = np.append(Array77, np.mean(PresMean, 1))
Array88 = np.append(Array88, np.mean(PresSTD, 1))
Array99 = np.append(Array99, np.mean(PresMean2, 1))
Array100 = np.append(Array100, np.mean(PresSTD2, 1))

intersects, Means, STDS, Temps, TempSTD, Temps2, Temp2STD, PresMean, PresSTD, PresMean2, PresSTD2 = compute_avg_std(MultiplierB, Times30, df30, StarterB, 3, 0)
intersects_array.append(intersects)
Array11 = np.append(Array11, Means)
Array22 = np.append(Array22, STDS)
Array33 = np.append(Array33, np.mean(Temps, 1))
Array44 = np.append(Array44, np.mean(TempSTD, 1))
Array55 = np.append(Array55, np.mean(Temps2, 1))
Array66 = np.append(Array66, np.mean(Temp2STD, 1))
Array77 = np.append(Array77, np.mean(PresMean, 1))
Array88 = np.append(Array88, np.mean(PresSTD, 1))
Array99 = np.append(Array99, np.mean(PresMean2, 1))
Array100 = np.append(Array100, np.mean(PresSTD2, 1))

intersects, Means, STDS, Temps, TempSTD, Temps2, Temp2STD, PresMean, PresSTD, PresMean2, PresSTD2 = compute_avg_std(MultiplierB, Times31, df31, StarterB, 6, 0)
intersects_array.append(intersects)
Array11 = np.append(Array11, Means)
Array22 = np.append(Array22, STDS)
Array33 = np.append(Array33, np.mean(Temps, 1))
Array44 = np.append(Array44, np.mean(TempSTD, 1))
Array55 = np.append(Array55, np.mean(Temps2, 1))
Array66 = np.append(Array66, np.mean(Temp2STD, 1))
Array77 = np.append(Array77, np.mean(PresMean, 1))
Array88 = np.append(Array88, np.mean(PresSTD, 1))
Array99 = np.append(Array99, np.mean(PresMean2, 1))
Array100 = np.append(Array100, np.mean(PresSTD2, 1))

intersects, Means, STDS, Temps, TempSTD, Temps2, Temp2STD, PresMean, PresSTD, PresMean2, PresSTD2 = compute_avg_std(MultiplierB, Times68, df68, StarterB, 6, 0)
intersects_array.append(intersects)
Array11 = np.append(Array11, Means)
Array22 = np.append(Array22, STDS)
Array33 = np.append(Array33, np.mean(Temps, 1))
Array44 = np.append(Array44, np.mean(TempSTD, 1))
Array55 = np.append(Array55, np.mean(Temps2, 1))
Array66 = np.append(Array66, np.mean(Temp2STD, 1))
Array77 = np.append(Array77, np.mean(PresMean, 1))
Array88 = np.append(Array88, np.mean(PresSTD, 1))
Array99 = np.append(Array99, np.mean(PresMean2, 1))
Array100 = np.append(Array100, np.mean(PresSTD2, 1))

intersects, Means, STDS, Temps, TempSTD, Temps2, Temp2STD, PresMean, PresSTD, PresMean2, PresSTD2 = compute_avg_std(MultiplierB, Times69, df69, StarterB, 1, 0)
intersects_array.append(intersects)
Array11 = np.append(Array11, Means)
Array22 = np.append(Array22, STDS)
Array33 = np.append(Array33, np.mean(Temps, 1))
Array44 = np.append(Array44, np.mean(TempSTD, 1))
Array55 = np.append(Array55, np.mean(Temps2, 1))
Array66 = np.append(Array66, np.mean(Temp2STD, 1))
Array77 = np.append(Array77, np.mean(PresMean, 1))
Array88 = np.append(Array88, np.mean(PresSTD, 1))
Array99 = np.append(Array99, np.mean(PresMean2, 1))
Array100 = np.append(Array100, np.mean(PresSTD2, 1))

intersects, Means, STDS, Temps, TempSTD, Temps2, Temp2STD, PresMean, PresSTD, PresMean2, PresSTD2 = compute_avg_std(MultiplierB, Times610, df610, StarterB, 14, 0)
intersects_array.append(intersects)
Array11 = np.append(Array11, Means)
Array22 = np.append(Array22, STDS)
Array33 = np.append(Array33, np.mean(Temps, 1))
Array44 = np.append(Array44, np.mean(TempSTD, 1))
Array55 = np.append(Array55, np.mean(Temps2, 1))
Array66 = np.append(Array66, np.mean(Temp2STD, 1))
Array77 = np.append(Array77, np.mean(PresMean, 1))
Array88 = np.append(Array88, np.mean(PresSTD, 1))
Array99 = np.append(Array99, np.mean(PresMean2, 1))
Array100 = np.append(Array100, np.mean(PresSTD2, 1))

intersects, Means, STDS, Temps, TempSTD, Temps2, Temp2STD, PresMean, PresSTD, PresMean2, PresSTD2 = compute_avg_std(MultiplierB, Times611, df611, StarterB, 16, 0)
intersects_array.append(intersects)
Array11 = np.append(Array11, Means)
Array22 = np.append(Array22, STDS)
Array33 = np.append(Array33, np.mean(Temps, 1))
Array44 = np.append(Array44, np.mean(TempSTD, 1))
Array55 = np.append(Array55, np.mean(Temps2, 1))
Array66 = np.append(Array66, np.mean(Temp2STD, 1))
Array77 = np.append(Array77, np.mean(PresMean, 1))
Array88 = np.append(Array88, np.mean(PresSTD, 1))
Array99 = np.append(Array99, np.mean(PresMean2, 1))
Array100 = np.append(Array100, np.mean(PresSTD2, 1))

dp1 = []
dp2 = []
dp3 = []
dp4 = []
dp1u = []
dp2u = []
dp3u = []
dp4u = []

for i in range(len(Array11)):
    if(i%4==0):
        dp1.append(Array11[i])
        dp1u.append(Array22[i])
    elif(i%4==1):
        dp2.append(Array11[i])
        dp2u.append(Array22[i])
    elif(i%4==2):
        dp3.append(Array11[i])
        dp3u.append(Array22[i])
    elif(i%4==3):
        dp4.append(Array11[i])
        dp4u.append(Array22[i])
                
def plot_it(df, MultiplierA, StarterA, date, month, counter, intersects):
    for j in range(counter):
        c = j
        fig, ax = plt.subplots()
        ax.plot(df[:, MultiplierA*j], df[:, StarterA+MultiplierA*j], "o-", markersize=1)
        ax.axvspan(df[np.amin(intersects[c, 0]), MultiplierA*j], df[np.amax(intersects[c, 0]), MultiplierA*j], alpha=0.5, color='red')
        ax.axvspan(df[np.amin(intersects[c, 1]), MultiplierA*j], df[np.amax(intersects[c, 1]), MultiplierA*j], alpha=0.5, color='red')
        ax.axvspan(df[np.amin(intersects[c, 2]), MultiplierA*j], df[np.amax(intersects[c, 2]), MultiplierA*j], alpha=0.5, color='red')
        ax.axvspan(df[np.amin(intersects[c, 3]), MultiplierA*j], df[np.amax(intersects[c, 3]), MultiplierA*j], alpha=0.5, color='red')
        plt.grid()
        plt.title(r"%s.%s.2022, Measurement %s" %(date, month, c + 1))
        plt.xlabel("Time [s]")
        plt.ylabel("dp [mbar]")
        plt.savefig("./figures/DPT_2022_%s_%s_%s.png" %(month, date, j + 1), dpi=1000)
        plt.close()
        
def plot_it2(df, MultiplierA, StarterA, date, month, counter, intersects):
    for j in range(counter):
        c = j
        fig, ax = plt.subplots()
        ax.plot(df[:, MultiplierA*j], df[:, StarterA+MultiplierA*j], "o-", markersize=1)
        ax.axvspan(df[np.amin(intersects[c, 0]), MultiplierA*j], df[np.amax(intersects[c, 0]), MultiplierA*j], alpha=0.5, color='red')
        ax.axvspan(df[np.amin(intersects[c, 1]), MultiplierA*j], df[np.amax(intersects[c, 1]), MultiplierA*j], alpha=0.5, color='red')
        ax.axvspan(df[np.amin(intersects[c, 2]), MultiplierA*j], df[np.amax(intersects[c, 2]), MultiplierA*j], alpha=0.5, color='red')
        ax.axvspan(df[np.amin(intersects[c, 3]), MultiplierA*j], df[np.amax(intersects[c, 3]), MultiplierA*j], alpha=0.5, color='red')
        plt.grid()
        plt.title(r"%s.%s.2022, Measurement %s" %(date, month, c + 1))
        plt.xlabel("Time [s]")
        plt.ylabel("p [bar]")
        plt.ylim(0.9, 2.3)
        plt.savefig("./figures/PT_2022_%s_%s_%s.png" %(month, date, j + 1), dpi=1000)
        plt.close()

plot_it(df23, MultiplierB, StarterB, 23, 5, 2, intersects_array[0])
plot_it2(df23, MultiplierB, 3, 23, 5, 2, intersects_array[0])

plot_it(df24, MultiplierC, StarterB, 24, 5, 6, intersects_array[1])
plot_it2(df24, MultiplierC, 3, 24, 5, 6, intersects_array[1])

plot_it(df25, MultiplierA, StarterA, 25, 5, 5, intersects_array[2])
plot_it2(df25, MultiplierA, 3, 25, 5, 5, intersects_array[2])

plot_it(df26, MultiplierB, StarterB, 26, 5, 9, intersects_array[3])
plot_it2(df26, MultiplierB, 3, 26, 5, 9, intersects_array[3])

plot_it(df27, MultiplierB, StarterB, 27, 5, 7, intersects_array[4])
plot_it2(df27, MultiplierB, 3, 27, 5, 7, intersects_array[4])

plot_it(df30, MultiplierB, StarterB, 30, 5, 3, intersects_array[5])
plot_it2(df30, MultiplierB, 3, 30, 5, 3, intersects_array[5])

plot_it(df31, MultiplierB, StarterB, 31, 5, 6, intersects_array[6])
plot_it2(df31, MultiplierB, 3, 31, 5, 6, intersects_array[6])

plot_it(df68, MultiplierB, StarterB, 8, 6,  6, intersects_array[7])
plot_it2(df68, MultiplierB, 3, 8, 6, 6, intersects_array[7])

plot_it(df69, MultiplierB, StarterB, 9, 6,  1, intersects_array[8])
plot_it2(df69, MultiplierB, 3, 9, 6, 1, intersects_array[8])

plot_it(df610, MultiplierB, StarterB, 10, 6,  14, intersects_array[9])
plot_it2(df610, MultiplierB, 3, 10, 6, 14, intersects_array[9])

plot_it(df611, MultiplierB, StarterB, 11, 6,  16, intersects_array[10])
plot_it2(df611, MultiplierB, 3, 11, 6, 16, intersects_array[10])
intersects = np.array(intersects)

dp1 = np.array(dp1)
dp2 = np.array(dp2)
dp3 = np.array(dp3)
dp4 = np.array(dp4)
dp1u = np.array(dp1u)
dp2u = np.array(dp2u)
dp3u = np.array(dp3u)
dp4u = np.array(dp4u)
Length1 = []
Length1_Unc = []
Diff_P1 = []
Diff_P2 = []
Diff_P3 = []
Diff_P4 = []
Array771 = []
Temp1 = []
tbgpd = pd.read_excel('./TB_Exp-AllData.xlsx', sheet_name = 1)
tbg = []

def isbad(mode, max_error = 10):
    if(mode == "manual"):
        tbg = np.array(tbgpd)
        tbg = tbg[:,2:]
        return tbg
    elif(mode == "auto"):
        max_error = max_error / 100     #convert from %
        tbg = np.zeros_like(tbgpd)
        for i in range(0, len(tbg)):
            for j in range(0, len(tbg[i])):
                    if(j % 4 == 0):
                        if(dp1u[i] / dp1[i] >= max_error): tbg[i][j] = "bad"
                        else: tbg[i][j] = "good"
                    elif(j % 4 == 1):
                        if(dp2u[i] / dp2[i] >= max_error): tbg[i][j] = "bad"
                        else: tbg[i][j] = "good"
                    elif(j % 4 == 2): 
                        if(dp3u[i] / dp3[i] >= max_error): tbg[i][j] = "bad"
                        else: tbg[i][j] = "good"
                    else: 
                        if(dp4u[i] / dp4[i] >= max_error): tbg[i][j] = "bad"
                        else: tbg[i][j] = "good"
        return(tbg)

def include_bad(include):
    if(include): return 0
    else:
        for i in range(0, len(tbg)):
            for j in range(0, len(tbg[i])):
                if(tbg[i][j] == "bad"):
                    if(j % 4 == 0): dp1[i] = np.nan
                    elif(j % 4 == 1): dp2[i] = np.nan
                    elif(j % 4 == 2): dp3[i] = np.nan
                    else: dp4[i] = np.nan
        return 0

mode = "auto"
max_error = 10  #%
tbg = isbad(mode, max_error)
include = False     #set to True to include "bad" measurements
include_bad(include)

            
for i in range(len(Length)):
    if(Length[i]!=0 and Length_Unc[i]!=0):
        Diff_P1.append(dp1[i])
        Diff_P2.append(dp2[i])
        Diff_P3.append(dp3[i])
        Diff_P4.append(dp4[i])
        Array771.append(Array77[i])
        Temp1.append(Array33[i])
        Length1.append(Length[i])
        Length1_Unc.append(Length_Unc[i])
       
Diff_P_all = np.array([Diff_P1, Diff_P2, Diff_P3, Diff_P4])

nans = np.isnan(Diff_P_all)
Diff_P_all_good = Diff_P_all
Length1_good = np.array([Length1, Length1, Length1, Length1])
Array771_good = np.array([Array771, Array771, Array771, Array771])       

for i in range(0, 4):        
    plt.figure(1)
    plt.tricontourf(Length1_good[i][~nans[i]], Array771_good[i][~nans[i]], Diff_P_all_good[i][~nans[i]])
    plt.plot(Length1_good[i][~nans[i]], Array771_good[i][~nans[i]], "ko")
    plt.colorbar()
    plt.xlabel("Bubble length [mm]")
    plt.ylabel("Absolute pressure [bar]")
    plt.title(f"dp{i + 1}, \"bad\" measurements included = {include}")
    plt.savefig(f"./figures/contures/PressureDropMatrix1_DP{i + 1}_includeBad={include}.png", dpi=1000)
    plt.close()
#
#plt.figure(2)
#plt.tricontourf(Temp1, Array771, Diff_P4)
#plt.plot(Temp1[0:35], Array771[0:35], "ro")
#plt.plot(Temp1[35:], Array771[35:], "go")
#plt.colorbar()
#plt.xlabel("Temperature")
#plt.ylabel("Absolute pressure [bar]")
#plt.savefig("./figures/PressureDropMatrix2.png", dpi=1000)
#plt.close()
#
#plt.figure(3)
#plt.tricontourf(Temp1, Length1, Diff_P4)
#plt.plot(Temp1[0:35], Length1[0:35], "ro")
#plt.plot(Temp1[35:], Length1[35:], "ko")
#plt.colorbar()
#plt.ylabel("Bubble length [mm]")
#plt.xlabel("Temperature")
#plt.savefig("./figures/PressureDropMatrix3.png", dpi=1000)
#plt.close()
#
#ax = plt.axes(projection='3d')
#ax.scatter3D(Length1, Array771, Diff_P4, c=Diff_P4, cmap='Greens');
#
#ax = plt.axes(projection='3d')
#ax.plot_trisurf(Length1, Array771, Diff_P4, cmap=plt.cm.Spectral)
#ax.colorbar()

dp4_1 = []
len1 = []
len15 = []
len2 = []
abs_pres1 = []
dp4_15 = []
abs_pres15 = []
dp4_2 = []
abs_pres2 = []
Unc1 = []
Unc15 =  []
Unc2 = []
dp4uu1 = []
dp4uu2 = []
dp4uu15 = []

for i in range(len(Length1)):
    if(Array771[i]<1.2):
        dp4_1.append(Diff_P4[i])
        abs_pres1.append(Array771[i])
        len1.append(Length1[i])
        Unc1.append(Length1_Unc[i])
        dp4uu1.append(dp4u[i])
    elif(Array771[i] >1.7):
        dp4_2.append(Diff_P4[i])
        abs_pres2.append(Array771[i])
        len2.append(Length1[i])
        Unc2.append(Length1_Unc[i])
        dp4uu2.append(dp4u[i])
    else:
        dp4_15.append(Diff_P4[i])
        abs_pres15.append(Array771[i])
        len15.append(Length1[i])
        Unc15.append(Length1_Unc[i])
        dp4uu15.append(dp4u[i])

fig, ax = plt.subplots()
ax.errorbar(len1, dp4_1, xerr=Unc1, yerr=dp4uu1, fmt="ko", label = "p = 1bar")
ax.errorbar(len15, dp4_15, xerr=Unc15, yerr=dp4uu15, fmt="ro", label = "p = 1.5bar")
ax.errorbar(len2, dp4_2, xerr=Unc2, yerr=dp4uu2, fmt="go", label = "p = 2bar")
plt.grid()
plt.legend()
plt.xlabel("Bubble length [mm]")
plt.ylabel("dp4 [mbar]")
plt.savefig("./figures/1dgraphs.png", dpi=1000)
plt.close()

dp4_1 = []
len1 = []
len15 = []
len2 = []
abs_pres1 = []
dp4_15 = []
abs_pres15 = []
dp4_2 = []
abs_pres2 = []

for i in range(len(Length1)):
    if(Array771[i]<1.2 and Temp1[i]>38 and Temp1[i]<43):
        dp4_1.append(Diff_P4[i])
        abs_pres1.append(Array771[i])
        len1.append(Length1[i])
    elif(Array771[i] >1.7 and Temp1[i]>38 and Temp1[i]<43):
        dp4_2.append(Diff_P4[i])
        abs_pres2.append(Array771[i])
        len2.append(Length1[i])
    else:
        if(Temp1[i]>38 and Temp1[i]<43):
            dp4_15.append(Diff_P4[i])
            abs_pres15.append(Array771[i])
            len15.append(Length1[i])

plt.figure(5)
plt.plot(len1, dp4_1, "ko", label = "p = 1bar")
plt.plot(len15, dp4_15, "ro", label = "p = 1.5bar")
plt.plot(len2, dp4_2, "go", label = "p = 2bar")
plt.grid()
plt.legend()
plt.xlabel("Bubble length [mm]")
plt.ylabel("dp4 [mbar]")
plt.title("40 degrees")
plt.savefig("./figures/1dgraph40degrees.png", dpi=1000)
plt.close()

dp4_1 = []
len1 = []
len15 = []
len2 = []
abs_pres1 = []
dp4_15 = []
abs_pres15 = []
dp4_2 = []
abs_pres2 = []

for i in range(len(Length1)):
    if(Array771[i]<1.2 and Temp1[i]>18 and Temp1[i]<23):
        dp4_1.append(Diff_P4[i])
        abs_pres1.append(Array771[i])
        len1.append(Length1[i])
    elif(Array771[i] >1.7 and Temp1[i]>18 and Temp1[i]<23):
        dp4_2.append(Diff_P4[i])
        abs_pres2.append(Array771[i])
        len2.append(Length1[i])
    else:
        if(Temp1[i]>18 and Temp1[i]<23):
            dp4_15.append(Diff_P4[i])
            abs_pres15.append(Array771[i])
            len15.append(Length1[i])

plt.figure(2)
plt.plot(len1, dp4_1, "ko", label = "p = 1bar")
plt.plot(len15, dp4_15, "ro", label = "p = 1.5bar")
plt.plot(len2, dp4_2, "go", label = "p = 2bar")
plt.grid()
plt.legend()
plt.xlabel("Bubble length [mm]")
plt.ylabel("dp4 [mbar]")
plt.title("20 degrees")
plt.savefig("./figures/1dgraph20degrees.png", dpi=1000)
plt.close()

dp4_1 = []
len1 = []
len15 = []
len2 = []
abs_pres1 = []
dp4_15 = []
abs_pres15 = []
dp4_2 = []
abs_pres2 = []

for i in range(len(Length1)):
    if(Array771[i]<1.2 and Temp1[i]>28 and Temp1[i]<33):
        dp4_1.append(Diff_P4[i])
        abs_pres1.append(Array771[i])
        len1.append(Length1[i])
    elif(Array771[i] >1.7 and Temp1[i]>28 and Temp1[i]<33):
        dp4_2.append(Diff_P4[i])
        abs_pres2.append(Array771[i])
        len2.append(Length1[i])
    else:
        if(Temp1[i]>28 and Temp1[i]<33):
            dp4_15.append(Diff_P4[i])
            abs_pres15.append(Array771[i])
            len15.append(Length1[i])

plt.figure(3)
plt.plot(len1, dp4_1, "ko", label = "p = 1bar")
plt.plot(len15, dp4_15, "ro", label = "p = 1.5bar")
plt.plot(len2, dp4_2, "go", label = "p = 2bar")
plt.grid()
plt.legend()
plt.xlabel("Bubble length [mm]")
plt.ylabel("dp4 [mbar]")
plt.title("30 degrees")
plt.savefig("./figures/1dgraph30degrees.png", dpi=1000)
plt.close()

dp4_1 = []
len1 = []
len15 = []
len2 = []
abs_pres1 = []
dp4_15 = []
abs_pres15 = []
dp4_2 = []
abs_pres2 = []

for i in range(len(Length1)):
    if(Array771[i]<1.2 and Temp1[i]>48 and Temp1[i]<53):
        dp4_1.append(Diff_P4[i])
        abs_pres1.append(Array771[i])
        len1.append(Length1[i])
    elif(Array771[i] >1.7 and Temp1[i]>48 and Temp1[i]<53):
        dp4_2.append(Diff_P4[i])
        abs_pres2.append(Array771[i])
        len2.append(Length1[i])
    else:
        if(Temp1[i]>48 and Temp1[i]<53):
            dp4_15.append(Diff_P4[i])
            abs_pres15.append(Array771[i])
            len15.append(Length1[i])
plt.figure(4)
plt.plot(len1, dp4_1, "ko", label = "p = 1bar")
plt.plot(len15, dp4_15, "ro", label = "p = 1.5bar")
plt.plot(len2, dp4_2, "go", label = "p = 2bar")
plt.grid()
plt.legend()
plt.xlabel("Bubble length [mm]")
plt.ylabel("dp4 [mbar]")
plt.title("50 degrees")
plt.savefig("./figures/1dgraph50degrees.png", dpi=1000)
plt.close()

dp4_1 = []
dp4_2 = []
dp4_3 = []
dp4_4 = []
len1 = []
len2 = []
len3 = []
len4 = []
abs_pres1 = []
abs_pres2 = []
abs_pres3 = []
abs_pres4 = []

for i in range(len(Length1)):
    if(Array771[i]<1.2 and Temp1[i]>18 and Temp1[i]<23):
        dp4_1.append(Diff_P4[i])
        abs_pres1.append(Array771[i])
        len1.append(Length1[i])
    elif(Array771[i] <1.2 and Temp1[i]>28 and Temp1[i]<33):
        dp4_2.append(Diff_P4[i])
        abs_pres2.append(Array771[i])
        len2.append(Length1[i])
    elif(Array771[i] <1.2 and Temp1[i]>38 and Temp1[i]<43):
        dp4_3.append(Diff_P4[i])
        abs_pres3.append(Array771[i])
        len3.append(Length1[i])
    elif(Array771[i] <1.2 and Temp1[i]>48 and Temp1[i]<53):
        dp4_4.append(Diff_P4[i])
        abs_pres4.append(Array771[i])
        len4.append(Length1[i])
        
plt.figure(1)
plt.plot(len1, dp4_1, "ko", label = "T=20")
plt.plot(len2, dp4_2, "go", label = "T=30")
plt.plot(len3, dp4_3, "ro", label = "T=40")
plt.plot(len4, dp4_4, "bo", label = "T=50")
plt.grid()
plt.legend()
plt.xlabel("Bubble length [mm]")
plt.ylabel("dp4 [mbar]")
plt.title("1bar")
plt.savefig("./figures/1dgraph1bar.png", dpi=1000)
plt.close()

dp4_1 = []
dp4_2 = []
dp4_3 = []
dp4_4 = []
len1 = []
len2 = []
len3 = []
len4 = []
abs_pres1 = []
abs_pres2 = []
abs_pres3 = []
abs_pres4 = []

for i in range(len(Length1)):
    if(Array771[i]>1.7 and Temp1[i]>18 and Temp1[i]<23):
        dp4_1.append(Diff_P4[i])
        abs_pres1.append(Array771[i])
        len1.append(Length1[i])
    elif(Array771[i]>1.7 and Temp1[i]>28 and Temp1[i]<33):
        dp4_2.append(Diff_P4[i])
        abs_pres2.append(Array771[i])
        len2.append(Length1[i])
    elif(Array771[i]>1.7 and Temp1[i]>38 and Temp1[i]<43):
        dp4_3.append(Diff_P4[i])
        abs_pres3.append(Array771[i])
        len3.append(Length1[i])
    elif(Array771[i]>1.7 and Temp1[i]>48 and Temp1[i]<53):
        dp4_4.append(Diff_P4[i])
        abs_pres4.append(Array771[i])
        len4.append(Length1[i])
plt.figure(2)
plt.plot(len1, dp4_1, "ko", label = "T=20")
plt.plot(len2, dp4_2, "go", label = "T=30")
plt.plot(len3, dp4_3, "ro", label = "T=40")
plt.plot(len4, dp4_4, "bo", label = "T=50")
plt.grid()
plt.legend()
plt.xlabel("Bubble length [mm]")
plt.ylabel("dp4 [mbar]")
plt.title("2bar")
plt.savefig("./figures/1dgraph2bar.png", dpi=1000)
plt.close()

dp4_1 = []
dp4_2 = []
dp4_3 = []
dp4_4 = []
len1 = []
len2 = []
len3 = []
len4 = []
abs_pres1 = []
abs_pres2 = []
abs_pres3 = []
abs_pres4 = []

for i in range(len(Length1)):
    if(Array771[i]>1.2 and Array771[i]<1.7  and Temp1[i]>18 and Temp1[i]<23):
        dp4_1.append(Diff_P4[i])
        abs_pres1.append(Array771[i])
        len1.append(Length1[i])
    elif(Array771[i] >1.2 and Array771[i]<1.7 and Temp1[i]>27 and Temp1[i]<33):
        dp4_2.append(Diff_P4[i])
        abs_pres2.append(Array771[i])
        len2.append(Length1[i])
    elif(Array771[i] >1.2 and Array771[i]<1.7  and Temp1[i]>37 and Temp1[i]<43):
        dp4_3.append(Diff_P4[i])
        abs_pres3.append(Array771[i])
        len3.append(Length1[i])
    elif(Array771[i] >1.2 and Array771[i]<1.7 and Temp1[i]>48 and Temp1[i]<53):
        dp4_4.append(Diff_P4[i])
        abs_pres4.append(Array771[i])
        len4.append(Length1[i])
plt.figure(3)
plt.plot(len1, dp4_1, "ko", label = "T=20")
plt.plot(len2, dp4_2, "go", label = "T=30")
plt.plot(len3, dp4_3, "ro", label = "T=40")
plt.plot(len4, dp4_4, "bo", label = "T=50")
plt.grid()
plt.legend()
plt.xlabel("Bubble length [mm]")
plt.ylabel("dp4 [mbar]")
plt.title("1.5bar")
plt.savefig("./figures/1dgraph15bar.png", dpi=1000)
plt.close()


def draw1dgraph(plot_dp):
    dp20 = []
    dp30 = []
    dp40 = []
    dp50 = []
    dpu20 = []
    dpu30 = []
    dpu40 = []
    dpu50 = []
    len1 = []
    len2 = []
    len3 = []
    len4 = []
    len1u = []
    len2u = []
    len3u = []
    len4u = []
    abs_pres1 = []
    abs_pres2 = []
    abs_pres3 = []
    abs_pres4 = []
    Diff_P = []
    labels20 = []
    labels30 = []
    labels40 = []
    labels50 = []

    if(plot_dp == 1):   
        Diff_P = Diff_P1
        Diff_PU = dp1u
    elif(plot_dp == 2): 
        Diff_P = Diff_P2
        Diff_PU = dp2u
    elif(plot_dp == 3): 
        Diff_P = Diff_P3
        Diff_PU = dp3u
    else: 
        Diff_P = Diff_P4
        Diff_PU = dp4u
    
    for i in range(len(Length1)):
        if(Temp1[i]>18 and Temp1[i]<23):
            dp20.append(Diff_P[i])
            dpu20.append(Diff_PU[i])
            abs_pres1.append(Array771[i])
            len1.append(Length1[i])
            len1u.append(Length1_Unc[i])
            labels20.append(i)
        elif(Temp1[i]>27 and Temp1[i]<33):
            dp30.append(Diff_P[i])
            dpu30.append(Diff_PU[i])
            abs_pres2.append(Array771[i])
            len2.append(Length1[i])
            len2u.append(Length1_Unc[i])
            labels30.append(i)
        elif(Temp1[i]>37 and Temp1[i]<43):
            dp40.append(Diff_P[i])
            dpu40.append(Diff_PU[i])
            abs_pres3.append(Array771[i])
            len3.append(Length1[i])
            len3u.append(Length1_Unc[i])
            labels40.append(i)
        elif(Temp1[i]>48 and Temp1[i]<53):
            dp50.append(Diff_P[i])
            dpu50.append(Diff_PU[i])
            abs_pres4.append(Array771[i])
            len4.append(Length1[i])
            len4u.append(Length1_Unc[i])
            labels50.append(i)
    plt.figure(4)
    
    offset = max(Diff_P) / 30
    plt.errorbar(len1, dp20, yerr = dpu20, xerr = len1u, fmt = "ko", capsize = 5, label = "T=20")
    for i in range(len(len1)):
        plt.annotate(labels20[i], (len1[i], dp20[i]), ha = "center", xytext = (len1[i], dp20[i] + offset))
    plt.errorbar(len2, dp30, yerr = dpu30, xerr = len2u, fmt = "go", capsize = 5, label = "T=30")
    for i in range(len(len2)):
        plt.annotate(labels30[i], (len2[i], dp30[i]), ha = "center", xytext = (len2[i], dp30[i] + offset), color = "green")
    plt.errorbar(len3, dp40, yerr = dpu40, xerr = len3u, fmt = "ro", capsize = 5, label = "T=40")
    for i in range(len(len3)):
        plt.annotate(labels40[i], (len3[i], dp40[i]), ha = "center", xytext = (len3[i], dp40[i] + offset), color = "red")
    plt.errorbar(len4, dp50, yerr = dpu50, xerr = len4u, fmt = "bo", capsize = 5, label = "T=50")
    for i in range(len(len4)):
        plt.annotate(labels50[i], (len4[i], dp50[i]), ha = "center", xytext = (len4[i], dp50[i] + offset), color = "blue")
        
    plt.grid()
    plt.legend()
    plt.xlabel("Bubble length [mm]")
    plt.ylabel(f"dp{plot_dp} [mbar]")
    if(include):
        plt.title(f"includeBad={include}")
        plt.savefig(f"./figures/dp_graphs/1dgraphPres{plot_dp}_includeBad={include}.png", dpi=1000)
    else:
        plt.title(f"includeBad={include}_mode={mode}_max-error={max_error}%")
        plt.savefig(f"./figures/dp_graphs/1dgraphPres{plot_dp}_includeBad={include}_mode={mode}_maxerror={max_error}.png", dpi=1000)
    plt.close()


for i in range(1, 5):
    draw1dgraph(i)
    
