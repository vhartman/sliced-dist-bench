This is the repo corresponding to [the blog post on benchmarking sliced distances using python](https://vhartmann.com/fast-numpy-aggregation/).

Results can be found in `final_plots`. The code for computing the sliced distances is in `configuration.py`. 
The code for benchmarking is in `benchmark.py`.  Fittingly, the code for reproducing the plots that are in the post can be found in `plot_result.py`.

Reproducing the results requires running

```
pytest benchmark.py --benchmark-json=[path_to_folder_where_results_should_be_stored].json
python3 plot_result.py [folder_with_results] --output [folder_where_the_plots_should_go] --use_paper_style
```
