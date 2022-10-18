import numpy as np
import pandas as pd

# Reads excel file TB1_corrected.xlsx and saves every measurement as seperate .txt file

excel_file = "TB1_corrected.xlsx"
tabs = pd.ExcelFile("TB1_corrected.xlsx").sheet_names 
f = open("metadata.txt", "w")
f.write("measurement no.\tdaily measurement\tdate\tavg_T1\tavg_T2\tavg_p1\tavg_p2\n")
count = 0

for i in range(3, 15):
    df = pd.read_excel(excel_file, sheet_name = i)
    dfn = np.array(df)

    breaks = np.argwhere(dfn[0] != dfn[0])
    if(len(breaks) == 0): breaks = np.array([0])
    shift = 0

    for j in range(0, len(breaks) + 1):
        count = count + 1
        multiplier = int(breaks[0])

        t = dfn[:,multiplier * j + shift]
        nans = np.argwhere(t != t)

        t = np.delete(t, nans)
        T1 = dfn[:,multiplier * j + shift + 1]
        T1 = np.delete(T1, nans)
        T2 = dfn[:,multiplier * j + shift + 2]
        T2 = np.delete(T2, nans)
        p1 = dfn[:,multiplier * j + shift + 3]
        p1 = np.delete(p1, nans)
        p2 = dfn[:,multiplier * j + shift + 4]
        p2 = np.delete(p2, nans)
        dp = dfn[:,multiplier * j + shift + 5]
        dp = np.delete(dp, nans)

        T1_avg = np.mean(T1)
        T2_avg = np.mean(T2)
        p1_avg = np.mean(p1)
        p2_avg = np.mean(p2)

        tofile = np.array(np.transpose([t, T1, T2, p1, p2, dp]))
        np.savetxt(f"measurement{count}_{tabs[i]}.txt", tofile, fmt = "%s", delimiter = ",")

        shift = shift + 1
        f.write(f"{count}\t{j + 1}\t{tabs[i]}\t{T1_avg}\t{T2_avg}\t{p1_avg}\t{p2_avg}\n")

f.close()
