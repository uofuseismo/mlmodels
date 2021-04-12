#!/usr/bin/env python3
import pyuussmlmodels


def test_1cpicker_zcnn():
    proc = pyuussmlmodels.OneComponentPicker.ZCNN.ProcessData()
    assert abs(proc.target_sampling_period - 1./100) < 1.e-14, 'target sampling period failed'
    


if __name__ == "__main__":
    test_1cpicker_zcnn()

    print(dir(pyuussmlmodels.OneComponentPicker.ZCNN))
