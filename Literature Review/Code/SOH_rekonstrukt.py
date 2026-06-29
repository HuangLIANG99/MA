#lib
from fileinput import filename
import os
import sys
import re
from pathlib import Path
from tracemalloc import start
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from FCT.battery_discharge_segments import read_battery_data

#常量构造

Datei_from = "https://iseadocker.isea.rwth-aachen.de:9000"
REQ_COLS={"time":["Zeit","Time"],"current":["Strom","Current"],"voltage":["Spannung","Voltage"],"ah":["AhAkku"],"state":["Zustand"]}
RC_COLS=["SOH_%","SOC_%","R0_ohm","R1_ohm","C1_F","TAU1_S","R2_ohm","C2_F","TAU2_S"]
RC_PARAM_COLS=[]
for c in RC_COLS:
    if c not in {"SOH_%","SOC_%"}:
        RC_PARAM_COLS.append(c)

#Datei数据库连接
def get_datei():
    key,secret=os.getenv("MINIO_ACCESS_KEY"),os.getenv("MINIO_SECRET_KEY")
    return{
        "key":key,
        "secret":secret,
        "client_kwargs":{"datei_url":Datei_from,"region_name":"us-east-1"},
        "config_kwargs":{"datei":{"addressing_style":"path"}},"signature_version":"s3v4"
    }

#读取文件
def read_battery_data(target_file:str) -> pd.pd.DataFrame:
    path=target_file if str(target_file).startswith("s3://") else f"s3//{target_file}"
    df=pd.read_parquet(path,storage_options=get_datei())
    df.attrs["source_filepath"]=target_file
    return df

#解析文件名，例如soc，dod，soh等
def extract_soc_dod_soh_from_filename(filepath:str):
    if not filepath:
        return None,None,None
    filename=Path(str(filepath)).name

    def pick(label):
        for pat in(rf"(\d+(?:[\.,]\d+)?)\s*{label}",rf"{label}\s*(\d+(\d+(?:[\.,]\d+)?)"):
            m=re.search(pat,filename,re.I)
            if m:
                return float(m.group(1).replace(",","."))
        return None
    return pick("SOC"),pick("DOD"),pick("SOH")

#放电片段提取

#放电片段展示，全部或者前？个
def select_dch_segment(total_found:int,num_segments=0):
    if num_segments in (None,0) or num_segments>=total_found:
        return np.arange(total_found),f"一共有{total_found}个放电片段"
    return np.arange(num_segments),f"读取前{num_segments}片段，一共{total_found}个片段"

#时间差计算（与第一个时间点相比）
def _time_seconds(s:pd.Series)->pd.Series:
    return (s-s.iloc[0]).dt.total_seconds()

#通过读取的文件名计算理论soc的起点和终点
def _calc_theoretical_soc(filepath):
    soc,dod=extract_soc_dod_soh_from_filename(filepath) if filepath else (None,None)
    if soc is None or dod is None:
        return soc,dod,None,None
    return soc,dod,soc+0.5*dod,soc-0.5*dod

#通过状态变化确定放电片段的时间段
def _dch_bounds(df:pd.DataFrame,state_col="Zustand"):
    state=df[state_col].astype(str).str.strip().str.upper()
    starts=state.eq("DCH")& state.shift(1,fill_value="").ne("DCH")
    ends=state.eq("DCH")&state.shift(-1,fill_value="").ne("DCH")
    return list(zip(df.index[starts],df.index[ends]))


def get_dch_point(df:pd.DataFrame,filepath=None):
    data=df.reset_index(drop=True).copy()
    data["Zustand_Clean"]=data[REQ_COLS["state"]].astype(str).str.strip().str.upper()
    bounds=_dch_bounds(data,"Zustand_Clean")
    start_points=[]
    end_points=[]
    for idx,(start_idx,end_idx) in enumerate(bounds,start=1):
        start=data.loc[start_idx]
        end=data.loc[end_idx]
        start_points.append({
            "segment_id":idx,
            "time":start[REQ_COLS["time"]],
            "voltage":start[REQ_COLS["voltage"]],
            "current":start[REQ_COLS["current"]],
            "Ah":start[REQ_COLS["ah"]],
            "state":start[REQ_COLS["state"]],
            "index":start_idx
        })
        end_points.append({
            "segment_id":idx,
            "time":end[REQ_COLS["time"]],
            "voltage":end[REQ_COLS["voltage"]],
            "current":end[REQ_COLS["current"]],
            "Ah":end[REQ_COLS["ah"]],
            "state":end[REQ_COLS["state"]],
            "index":end_idx
        })
    return pd.DataFrame(start_points),pd.DataFrame(end_points)