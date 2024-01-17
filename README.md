# GeomagneticForecasting-BFE
 
Repository for the paper "A framework for evaluating geomagnetic indices forecasting models"

Required packages can be found in the ``requirements.txt`` file.

The data can be downloaded using ``python download_data.py`` and will create a pickle file in the data folder ``sym_asy_indices.pkl`` for the SYM-H and ASY-H indices and another pickle ``dst_index.pkl`` for the Dst data.

The data will be downloaded from the CDAWeb, the SYM and ASY indices will be downloaded from the ``OMNI_HRO_5MIN`` dataset with a 5 minute resolution. The data will be downloaded from 1981 until the end of 2022.

The repository contains 5 notebooks

* ``0-download-and-save-data.ipynb``: Downloads the whole timeline for the SYM-H index which will be used in the `1-set-expansion.ipynb` notebook, the ACE MAG and SWEPAM datasets and the SYM-H for the selected storms that will be used in the remaining notebooks. The data will be downloaded from NASA's CDAWeb using the cdasws package.
* 
* ``1-set-expansion.ipynb``: Contains the expansion of the original Siciliano set (https://doi.org/10.1029/2020SW002589) using the storms identified in (https://doi.org/10.1007/s11069-023-06241-1). The original subsets are maintained and the new storms are properly added to the subsets. Additionally, the storms for which the test key parameters are available are used in a second test set.

* ``2-comparison-rmse-bfe.ipynb``: Contains the comparison of the RMSE and the BFE on a persistence model on all the storms and to create the comparison figures of the paper.

* ``3a-baseline-model-1h-sym.ipynb`` and ``3a-baseline-model-1h-sym.ipynb``: Training, validation and test of the baseline model described in the manuscript for the SYM-H index 1 and 2 hours ahead, along with the computation of the forecasting metrics for the test and test key storms.
