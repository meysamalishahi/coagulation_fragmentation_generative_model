# CaloChallenge Dataset 1 photons

Official source: DOI `10.5281/zenodo.8099322`.

- `dataset_1_photons_1.hdf5`: official training sample, MD5
  `005d2adeda7db034b388112661265656`
- `dataset_1_photons_2.hdf5`: official evaluation sample, MD5
  `4767715ed56e99565fd9c67340661e70`

The binary HDF5 files are ignored by Git and can be restored with
`sh download_calorimeter_data.sh`. The official split is preserved: validation
events are held out from file 1 and all final test metrics use file 2 only.
