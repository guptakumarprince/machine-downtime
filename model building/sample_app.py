# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 17:37:01 2023

@author: Prince Kumar Gupta
"""

import streamlit as st
import pandas as pd
import numpy as np

st.title("my_sample_app")
data.info()

data1=data.isnull().sum()


df["KIT ITEM"].unique().value_counts()

df2=df['KIT ITEM'].value_counts()

df['Customer Name'].value_counts()

df1=  df['OEM'].value_counts()

unique_kit_item_count = df['KIT ITEM'].nunique()
print("Number of unique KIT ITEMs:", unique_kit_item_count)



