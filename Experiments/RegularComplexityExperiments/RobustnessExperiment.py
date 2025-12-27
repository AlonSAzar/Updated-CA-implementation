from Experiments.experiments import *

class RobustnessExperiment(BaseExperiment):
    """
    Tests the relationship between Complexity and Evolutionary Robustness.
    """

    def __init__(self, engine: ElementaryCA, metric: ComplexityMetric, random_iterations: int = 20):
        super().__init__(engine, metric)
        self.random_iterations = random_iterations

    def run(self, strategy, num_seeds=20, shuffle_control = False):
        """
        Returns dictionaries {rule_id: robustness_score}, {rule_id, phenotype_complexity}.
        Robustness = Average NCC between Rule(seed) and MutantRule(seed).
        """

        print(f"--- Robustness: {strategy.name()} ---")

        # Setup dicts
        rule_robustness_scores = {}
        rule_images_dict = {}
        seeds = [self.engine.generate_seed() for _ in range(num_seeds)]

        for rule in tqdm(range(256), desc="Evolutionary Robustness"):
            rule_score = 0
            rule_images_list = []

            for seed in seeds:
                base_img = self.engine.run(rule, seed)[1:]
                rule_images_list.append(base_img)

                # Check all 8 1-bit neighbors
                mutant_scores = []

                num_vars = strategy.get_variations_count(self.engine, rule, seed, self.random_iterations)

                for i in range(num_vars):
                    m_rule, m_seed = strategy.apply(self.engine, rule, seed, i)

                    mut_img = self.engine.run(m_rule, m_seed)[1:]

                    # Optional: Shuffle image to test control
                    if shuffle_control:
                        mut_img = shuffle_image_rows(mut_img)

                    # Compute NCC (Normalized Cross Correlation)
                    ncc = compute_ncc(base_img, mut_img)
                    mutant_scores.append(ncc)
                rule_score += np.mean(mutant_scores)

            # Calculate average across seeds, per rule
            rule_robustness_scores[rule] = rule_score / len(seeds)
            rule_images_dict[rule] = rule_images_list

        rule_phenotype_complexity_scores = self.mean_phenotype_complexity(rule_images_dict)

        self.plot_results(rule_robustness_scores, rule_phenotype_complexity_scores, strategy)

        return rule_robustness_scores, rule_phenotype_complexity_scores

    # TODO change name to plot like in abstract class
    """ This function is AI generated. """
    def plot_results(self, robustness_scores: dict, phenotype_complexities: dict, mutation_type):
        """
        Plots Robustness vs Complexity and calculates correlations.
        """
        # 1. Align data (ensure we match the same rule for X and Y)
        # We sort by rule ID to ensure lists correspond index-for-index
        rules = sorted(robustness_scores.keys())

        # X = Complexity, Y = Robustness
        Xs = [phenotype_complexities[r] for r in rules]
        Ys = [robustness_scores[r] for r in rules]

        # 2. Calculate Correlations
        # TODO see if the warning is a problem
        pearson_val, _ = pearsonr(Xs, Ys)
        spearman_val, _ = spearmanr(Xs, Ys)

        # 3. Plotting
        plt.figure(figsize=(10, 7))
        plt.scatter(Xs, Ys, alpha=0.6, c='teal', edgecolors='black', linewidth=0.5)

        plt.xlabel(f"Phenotype Complexity ({self.metric.name()})")
        plt.ylabel("Robustness (Avg NCC)")

        # Pull L and T from the engine instance
        L = self.engine.L
        T = self.engine.T
        plt.title(f"1D CA Robustness VS Complexity, {mutation_type.name()}. L={L}, T={T}")

        # 4. Info Box with Stats
        stats_text = (
            f"Correlations:\n"
            f"  Spearman = {spearman_val:.3f}\n"
            f"  Pearson = {pearson_val:.3f}"
        )

        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right', bbox=props)

        plt.grid(True, alpha=0.3)
        plt.show()

    def compute_rule_complexity(self, rule: int):
        """Compute the complexity of the array represented by the rule itself."""
        binary_rule = int_to_binary_array_numpy(rule)
        return self.metric.calculate(binary_rule)

    def mean_phenotype_complexity(self, results):
        """Compute mean phenotype complexity per rule over all seeds."""
        mean_complexities = {}
        for rule, images in results.items():
            complexities = [self.metric.calculate(img) for img in images]
            mean_complexities[rule] = np.mean(complexities)
        return mean_complexities

    # TODO I can implement, but I think we're past the point of relevance
    def plot(self): pass
    def analyze(self): pass
