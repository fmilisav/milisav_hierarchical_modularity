import os
import numpy as np

from neuromaps.datasets import fetch_annotation
from netneurotools import datasets as nntdata
from neuromaps.parcellate import Parcellater
from neuromaps.images import dlabel_to_gifti

timescales = fetch_annotation(desc='megtimescale')
schaefer = nntdata.fetch_schaefer2018('fslr32k')['400Parcels7Networks']
schaefer = dlabel_to_gifti(schaefer)
parc = Parcellater(schaefer, 'fsLR')
parc_timescales = parc.fit_transform(timescales, 'fsLR')

np.save('data/parc_timescales.npy', parc_timescales)