 
##### Installing and activating the environment :

`conda env create --prefix ./env --file environment.yml` 
`conda activate GFMdistill  conda env update --file environment.yaml --prune`


Instead of bash scripts, all specifications are now in configs/*.yaml, experimetnts are run like : 
`torchrun --nproc_per_node=1 run_pangaea.py +experiment=experiment1`




### 📜 Changelog

| Date       | Description                                              |
|------------|----------------------------------------------------------|
| 2025-07-10 | Initial release
| 2025-07-19 | run inference of remoteclip with FBP dataset (run_remoteclip.py)
