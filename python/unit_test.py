#!/usr/bin/env python
import pyuussmlmodels
import numpy as np
import os

def read_time_series(file_name): 
    assert os.path.exists(file_name), '{} does not exist'.format(file_name)
    infl = open(file_name, 'r')
    cdat = infl.read()
    infl.close()
    cdat = cdat.split('\n')
    n = len(cdat) - 1
    t = np.zeros(n)
    x = np.zeros(n)
    for i in range(n):
        ti, xi = cdat[i].split(',')
        t[i] = float(ti)
        x[i] = float(xi)
    return t, x

def test_p_one_component_p_detector_preprocessing():
   print("Performing P 1C UNet preprocessing test...")
   t, x = read_time_series('data/detectors/uNetOneComponentP/PB.B206.EHZ.zrunet_p.txt')
   t, y_ref = read_time_series('data/detectors/uNetOneComponentP/PB.B206.EHZ.PROC.zrunet_p.txt')
   assert len(x) == len(y_ref)
   assert len(x) == 360000
   sampling_rate = round(1./(t[1] - t[0]))
   assert sampling_rate == 100, 'sampling rate should be 100 Hz'
   preprocessor = pyuussmlmodels.Detectors.UNetOneComponentP.Preprocessing()
   assert preprocessor.target_sampling_rate == 100, 'target sampling rate should be 100 Hz'
   y = preprocessor.process(x, sampling_rate = sampling_rate)
   assert max(abs(y - y_ref)) < 1.e-1

def test_p_three_component_p_detector_preprocessing():
   print("Performing P 3C UNet preprocessing test...")
   t, vertical = read_time_series('data/detectors/uNetThreeComponentP/PB.B206.EHZ.zrunet_p.txt') 
   t, north    = read_time_series('data/detectors/uNetThreeComponentP/PB.B206.EH1.zrunet_p.txt')
   t, east     = read_time_series('data/detectors/uNetThreeComponentP/PB.B206.EH2.zrunet_p.txt')
   t, vertical_ref = read_time_series('data/detectors/uNetThreeComponentP/PB.B206.EHZ.PROC.zrunet_p.txt') 
   t, north_ref    = read_time_series('data/detectors/uNetThreeComponentP/PB.B206.EH1.PROC.zrunet_p.txt')
   t, east_ref     = read_time_series('data/detectors/uNetThreeComponentP/PB.B206.EH2.PROC.zrunet_p.txt')
   assert len(vertical) == 360000
   assert len(vertical) == len(north)
   assert len(vertical) == len(east) 
   assert len(vertical) == len(vertical_ref)
   assert len(vertical) == len(north_ref)
   assert len(vertical) == len(east_ref)
   sampling_rate = round(1./(t[1] - t[0]))
   assert sampling_rate == 100, 'sampling rate should be 100 Hz'
   preprocessor = pyuussmlmodels.Detectors.UNetThreeComponentP.Preprocessing()
   assert preprocessor.target_sampling_rate == 100, 'target sampling rate should be 100 Hz'
   vertical_proc, north_proc, east_proc \
       = preprocessor.process(vertical, north, east, sampling_rate = sampling_rate)
   assert max(abs(vertical_proc - vertical_ref)) < 1.e-1
   assert max(abs(north_proc - north_ref)) < 1.e-1
   assert max(abs(east_proc - east_ref)) < 1.e-1
   
def test_p_three_component_s_detector_preprocessing():
   print("Performing S 3C UNet preprocessing test...")
   t, vertical = read_time_series('data/detectors/uNetThreeComponentS/PB.B206.EHZ.zrunet_s.txt') 
   t, north    = read_time_series('data/detectors/uNetThreeComponentS/PB.B206.EH1.zrunet_s.txt')
   t, east     = read_time_series('data/detectors/uNetThreeComponentS/PB.B206.EH2.zrunet_s.txt')
   t, vertical_ref = read_time_series('data/detectors/uNetThreeComponentS/PB.B206.EHZ.PROC.zrunet_s.txt') 
   t, north_ref    = read_time_series('data/detectors/uNetThreeComponentS/PB.B206.EH1.PROC.zrunet_s.txt')
   t, east_ref     = read_time_series('data/detectors/uNetThreeComponentS/PB.B206.EH2.PROC.zrunet_s.txt')
   assert len(vertical) == 360000
   assert len(vertical) == len(north)
   assert len(vertical) == len(east) 
   assert len(vertical) == len(vertical_ref)
   assert len(vertical) == len(north_ref)
   assert len(vertical) == len(east_ref)
   sampling_rate = round(1./(t[1] - t[0]))
   assert sampling_rate == 100, 'sampling rate should be 100 Hz'
   preprocessor = pyuussmlmodels.Detectors.UNetThreeComponentS.Preprocessing()
   assert preprocessor.target_sampling_rate == 100, 'target sampling rate should be 100 Hz'
   vertical_proc, north_proc, east_proc \
       = preprocessor.process(vertical, north, east, sampling_rate = sampling_rate)
   assert max(abs(vertical_proc - vertical_ref)) < 1.e-1
   assert max(abs(north_proc - north_ref)) < 1.e-1
   assert max(abs(east_proc - east_ref)) < 1.e-1

def test_p_one_component_pick_regressor_preprocessing():
   #t, x = read_time_series('data/detectors/uNetOneComponentP/PB.B206.EHZ.zrunet_p.txt')
   #t, y_ref = read_time_series('data/detectors/uNetOneComponentP/PB.B206.EHZ.PROC.zrunet_p.txt')
   a = 1


def test_p_one_component_first_motion_classifier_preprocessing():
   a = 1

if __name__ == "__main__":
   test_p_one_component_p_detector_preprocessing()
   test_p_three_component_p_detector_preprocessing()
   test_p_three_component_s_detector_preprocessing() 
   #test_p_one_component_pick_regressor_preprocessing()
   #test_p_one_component_first_motion_classifier_preprocessing()
