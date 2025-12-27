import ComplexityMeasures.conditional_complexity
from ComplexityMeasures.complexity import ZlibComplexity
from Core.engine import ElementaryCA
from Experiments.ConditionalComplexityExperiments.ConditionalTransitionExperiment import ConditionalTransitionExperiment
from Experiments.PCA.PCARawPhenotypes import analyze_pca
from Experiments.RegularComplexityExperiments.PopulationGrowthExperiment import PopulationGrowthExperiment
from Experiments.RegularComplexityExperiments.RobustnessExperiment import RobustnessExperiment
from strategies import *


def main():
    # 1. Setup Configuration
    # TODO with bigger L, it seems further from the limit, weird
    L = 64
    T = 64
    NUM_SEEDS = 100

    # 2. Instantiate Objects
    engine = ElementaryCA(L, T)
    metric = ZlibComplexity()  # Can easily swap this with MutualInfoComplexity()

    # # --- Experiment A: Simplicity Bias ---
    # """
    # We generate NUM_SEEDS (default 100) different seeds per the 256 different rules.
    # """
    # print("--- Starting Simplicity Bias Experiment ---")
    # # sb_exp = SimplicityBiasExperiment(engine, metric)
    # # sb_exp.run(num_seeds=NUM_SEEDS)
    # #
    # # sb_exp.run(num_seeds=NUM_SEEDS, shuffle_control=True)
    #
    # # --- Experiment B: Population Growth Simplicity Bias ---
    # print("--- Starting Population Growth Experiment ---")
    # pop_exp = PopulationGrowthExperiment(engine, metric)
    # pop_exp.run(num_seeds=NUM_SEEDS)
    #
    # # --- Experiment C: Robustness ---
    # print("--- Starting Robustness Experiments ---")
    # rob_exp = RobustnessExperiment(engine, metric)
    #
    # print("Running Rule Flip...")
    # rob_exp.run(strategy=BitFlipRuleStrategy())
    #
    # print("Running Rule Flib, Shuffled...")
    # rob_exp.run(strategy=BitFlipRuleStrategy(), shuffle_control=True)
    #
    # print("Running Seed Flip...")
    # rob_exp.run(strategy=BitFlipSeedStrategy())
    #
    # print("Running Random Seed...")
    # rob_exp.run(strategy=RandomSeedStrategy())
    #
    # print("Running Random Rule...")
    # rob_exp.run(strategy=RandomRuleStrategy())
    #
    # print("Running Random Seed and Rule...")
    # rob_exp.run(strategy=RandomSeedAndRuleStrategy())

    # --- Experiment: Conditional Complexity ---
    rle_metric = ComplexityMeasures.complexity.RLEComplexity()
    cond_trans_exp = ConditionalTransitionExperiment(engine, rle_metric)
    # TODO iterate on rules and see how conditional comp bias changes according to rule
    freq_map, complexity_map = cond_trans_exp.run(168)
    freq_map, complexity_map = cond_trans_exp.run(168, shuffle_control=True)
    cond_trans_exp.plot_results(freq_map, complexity_map)

    analyze_pca(cond_trans_exp.return_phenotype_cache(), cond_trans_exp.return_transitions())




if __name__ == "__main__":
    main()