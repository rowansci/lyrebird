Benchmarking molecular conformer ensemble generation should answer 3 questions:
1. Is the model spanning conformational space?
2. Is the model producing reasonable structures?
3. Is the model producing the lowest energy structure?

Traditional CovMat scores answer the first question of distributional coverage. The energy-based benchmark seeks to answer the next two. We answer question 2 by taking the average energy difference between the optimized and non-optimized generated structures. This tells us how close to a local minimum a model generates structures from the ensemble. This is a way to define reasonableness. We answer question 3 by comparing the lowest energy generated structure after it's been optimized to the lowest energy ground truth structure. This tells us how well we find the global minimum.

This data contains the energy of all generated structures for each method before and after optimization and the energy of the ground truth structures for 20 molecular ensembles from GEOM-XL (this is very expensive so only a subset). There is frequency data for some structures for validating how well the optimization went, but we decided frequency calculations for all structures would be too slow and way too memory intensive because we're storing all intermediate structures. Some optimizations failed but very few, those are not included here. We optimized all of the ground truth structures too, but realized they were already optimized so the pre and post optimization numbers should be effectively the same.

To run this benchmark we do the following:
1. Compute the average energy difference between pre and post optimized structures for each ensemble for each generation method, and a cumulative average across all ensembles for a given method.
2. Compute the difference between the lowest energy generated structure after optimization and the lowest energy ground truth structure for each method.

The results are detailed here, with plots, comprehensive statistics and a full dump in the full_results.txt file. The idea was moving toward physically meaingful benchmarking with semiempirical methods that answers questions of physical meaningfulness to bridge the ML community with scientists who want more physically meaningful benchmarking, and we hope this represents a step toward that goal. 
