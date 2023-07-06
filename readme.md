# Swiss solar and wind electricity production model

This project aims to modeling the Swiss solar and wind electricity production using climatic data from
the Copernicus Climate Change Service information [CDS](https://cds.climate.copernicus.eu/), electricity
plants map from the Swiss Federal Office of Energy [SFOE](https://www.bfe.admin.ch/bfe/fr/home.html) and 
estimated electricity production data provided by [Pronovo](https://pronovo.ch/fr/services/rapports-et-publications/) 
and [EnergyCharts](https://www.energy-charts.info/charts/energy/chart.htm?l=fr&c=CH&chartColumnSorting=default).

It has been made to be used with [EcoDynElec](https://github.com/LESBAT-HEIG-VD/EcoDynElec), a tool allowing to compute 
the hourly carbon footprint of the Swiss electricity mix.

You can reproduce and tune the model by taking example on the example.ipynb notebook, or you can directly use the results
stored in the `ecd_enr_model/export/enr_prod_2017-2022.csv` file :
    
```python
import pandas as pd
from ecodynelec_enr_model import data_loading

enr_prod_all = pd.read_csv(f'{data_loading.root_dir}export/enr_prod_2017-2022.csv', index_col=0, parse_dates=[0])
```
