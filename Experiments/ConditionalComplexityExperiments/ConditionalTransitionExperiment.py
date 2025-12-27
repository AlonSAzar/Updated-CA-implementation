import strategies
from Experiments.experiments import *
from strategies import *
from scipy.stats import pearsonr, spearmanr, linregress

class ConditionalTransitionExperiment(BaseExperiment):
    def __init__(self, engine, metric):
        super().__init__(engine, metric)

    def run(self, rule_id, num_parents=5000, strategy=strategies.BitFlipSeedStrategy(), shuffle_control=False):
        """
        Maps transitions P(x -> y) vs Conditional Complexity K(y|x).
        """

        self.transitions = Counter()  # Counts of specific (hash(x), hash(y))
        self.phenotype_cache = {}  # Stores actual images by hash
        self.num_seeds_used = num_parents # Store for the plot stats

        parent_seeds = [self.engine.generate_seed() for _ in range(num_parents)]

        for p_seed in tqdm(parent_seeds, desc="Mapping Transitions"):
            # get parent phenotype
            # img_x = self.engine.run(rule_id, p_seed)
            # Changed img_x (taking only the last row):
            full_history = self.engine.run(rule_id, p_seed)
            img_x = full_history[-1:]  # Slice to keep 2D shape (1, L)
            h_x = img_x.tobytes()
            self.phenotype_cache[h_x] = img_x

            # Generate mutants (1-bit flip neighbors)
            # Assuming seed length is engine.L
            for bit in range(self.engine.L):
                # TODO also this needs refinement in accordance with the strategies,
                # such that we'll be able to use different strategies more easily
                m_rule, m_seed = strategy.apply(engine=self.engine, rule=rule_id, seed=p_seed, bit_index=bit)

                # Get Mutant Phenotype (y)
                img_y = self.engine.run(m_rule, m_seed)
                if shuffle_control:
                    img_y = shuffle_image_rows(img_y)
                img_y = img_y[-1:]
                h_y = img_y.tobytes()
                self.phenotype_cache[h_y] = img_y

                # Record Transition
                self.transitions[(h_x, h_y)] += 1

        # 3. Analyze Data points
        plot_data_k = []
        plot_data_prob = []

        total_transitions = sum(self.transitions.values())

        total_mutations_per_parent = Counter()
        for (h_x, h_y), count in self.transitions.items():
            total_mutations_per_parent[h_x] += count

        print("Calculating Conditional Complexities...")
        for (h_x, h_y), count in self.transitions.items():
            img_x = self.phenotype_cache[h_x]
            img_y = self.phenotype_cache[h_y]

            # TODO we can try to replace with RLE
            zlib_cond_complexity = ZlibConditionalComplexity()

            # Calculate K(y|x)
            k_cond = zlib_cond_complexity.calculate(img_x, img_y)

            # NORMALIZED PROBABILITY: P(x -> y)
            # This is (Total x to y transitions) / (Total transitions starting from x)
            prob = count / total_mutations_per_parent[h_x]

            plot_data_k.append(k_cond)
            plot_data_prob.append(np.log10(prob))

        return plot_data_k, plot_data_prob

    """This function specifically was written by AI, since it's a plotting function"""

    # TODO if I want I can modify such that the arguments taken will be like the other plot functions
    def plot_results(self, Ks, log_probs, title_add: str = ""):
        """
        Visualizes the Conditional Simplicity Bias with Upper Bound Fit and Stats.
        Ks: List/Array of conditional complexities K(y|x)
        log_probs: List/Array of log10(P(x -> y))
        """
        import numpy as np
        import matplotlib.pyplot as plt

        # Convert to numpy for calculations
        Ks = np.array(Ks)
        log_probs = np.array(log_probs)

        # ---------------- CORRELATIONS ----------------
        if len(Ks) > 1:
            pearson_corr, _ = pearsonr(Ks, log_probs)
            spearman_corr, _ = spearmanr(Ks, log_probs)
        else:
            pearson_corr = spearman_corr = 0

        # ---------------- UPPER BOUND FITTING ----------------
        # 1. Find max probability for each unique complexity value
        unique_ks = np.unique(Ks)
        max_log_probs = []
        for k in unique_ks:
            # Get all log_probs that have complexity == k
            max_val = np.max(log_probs[Ks == k])
            max_log_probs.append(max_val)

        unique_ks = np.array(unique_ks)
        max_log_probs = np.array(max_log_probs)

        # 2. Fit linear regression to the upper bound (y = mx + c)
        if len(unique_ks) > 1:
            slope, intercept, r_val, p_val, std_err = linregress(unique_ks, max_log_probs)
        else:
            slope, intercept = 0, 0

        # 3. Convert to form P(p) = 2^(-aK - b)
        # We fitted log10(P) = slope * K + intercept
        log2_10 = np.log2(10)
        a_param = -slope * log2_10
        b_param = -intercept * log2_10

        # ---------------- PLOTTING ----------------
        plt.figure(figsize=(10, 7))

        # Scatter Plot
        plt.scatter(Ks, log_probs, alpha=0.5, label='Transitions (x -> y)', s=20)

        # Plot Fitted Line
        if len(Ks) > 0:
            x_line = np.linspace(min(Ks), max(Ks), 100)
            y_line = slope * x_line + intercept
            plt.plot(x_line, y_line, color='red', linestyle='--', linewidth=2, label='Upper Bound Fit')

        # Labels
        plt.xlabel(f"Conditional Complexity K(y|x) ({self.metric.name()})")
        plt.ylabel("Log10 Transition Probability P(x -> y)")
        plt.title(f"Conditional Simplicity Bias (Standard CA, L={self.engine.L}, T={self.engine.T})\n" + title_add)

        # Info Box (Matching your template)
        stats_text = (
            f"Simulation Parameters:\n"
            f"  N_parent_seeds = {self.num_seeds_used}\n"
            f"  Seed Length L = {self.engine.L}\n\n"
            f"Correlations:\n"
            f"  Spearman = {spearman_corr:.3f}\n"
            f"  Pearson = {pearson_corr:.3f}\n\n"
            f"Fit P = 2^(-aK - b):\n"
            f"  a = {a_param:.3f}\n"
            f"  b = {b_param:.3f}"
        )

        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right', bbox=props)

        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.show()

    def return_phenotype_cache(self):
        return self.phenotype_cache

    def return_transitions(self):
        return self.transitions

    def plot(self): pass
    def analyze(self): pass
