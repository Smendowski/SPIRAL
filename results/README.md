1. Raw experiment logs are in CSV files named with convention "01_NAB.csv"
2. Helios supercomputer execution logs are in the log/ directory
3. parser.py creates files with prefix "uni_mergedTable"
4. to_latex_fine_grained_table.py formats data to per-metric LaTeX tables
5. to_latex_aggregated_table.py formats data for all metrics
6. to_latex_aggregated_table_without_tl_table.py formats data without transfer learning strategy breakdown
7. plot_convergence_analysis.py produces results in convergence/ directory
8. plot_transfer_learning_analysis.py produces results in tl_analysis/ directory
