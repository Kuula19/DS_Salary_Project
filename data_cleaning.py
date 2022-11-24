# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 13:22:57 2022

@author: Utilisateur
"""

import pandas as pd

df = pd.read_csv("glassdoor_jobs.csv")

# Get rid of missing salary rows

df =df[df["Salary Estimate"] != "-1"]

# Salary parsing

df["Hourly"] = df["Salary Estimate"].apply(lambda x: 1 if "per hour" in x.lower()  else 0)
df["Employer_Provided"] = df["Salary Estimate"].apply(lambda x: 1 if "employer provided salary:" in x.lower() else 0)

salary = df["Salary Estimate"].apply(lambda x: x.split("(")[0])
minus_kd = salary.apply(lambda x: x.replace('K','').replace("$",""))

min_hr_emp = minus_kd.apply(lambda x: x.lower().replace("per hour","").replace("employer provided salary:",""))
df["Min_Salary"] = min_hr_emp.apply(lambda x: int(x.split("-")[0]))
df["Max_Salary"] = min_hr_emp.apply(lambda x: int(x.split("-")[1]))
df["Average_Salary"] = (df.Min_Salary + df.Max_Salary) / 2

# Company name only
df["Company_text"] = df.apply( lambda x: x["Company Name"] if x["Rating"]<0 else x["Company Name"][:-3], axis=1)

# State field
df["Job_State"] = df["Location"].apply(lambda x: x.split(',')[1])
df.Job_State.value_counts()

# See if the job offer is in the headquarter
df.columns
df["same_state"] = df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis=1)

# Age of company
df["Company_age"] = df.Founded.apply(lambda x: x if x<0 else 2022 - x)

# Parsing of job description
df["Job Description"][1]
# DS Tools
## Python
df["Python_yn"] = df["Job Description"].apply(lambda x: 1 if "python" in x.lower() else 0)
df.Python_yn.value_counts()
##  R Studio
df["R_Studio_yn"] = df["Job Description"].apply(lambda x: 1 if "r studio" in x.lower() or "r-studio" in x.lower() else 0)
df.R_Studio_yn.value_counts()

## Spark
df["Spark_yn"] = df["Job Description"].apply(lambda x: 1 if "spark" in x.lower() else 0)
df.Spark_yn.value_counts()

## AWS
df["AWS"] = df["Job Description"].apply(lambda x: 1 if "aws" in x.lower() else 0)
df.AWS.value_counts()

## Excel
df["Excel_yn"] = df["Job Description"].apply(lambda x: 1 if "excel" in x.lower() else 0)
df.Excel_yn.value_counts()

df = df.drop("Unnamed: 0", axis=1)

df.to_csv("Salary_data_cleaned.csv", index=False)
df_out = pd.read_csv("Salary_data_cleaned.csv")
