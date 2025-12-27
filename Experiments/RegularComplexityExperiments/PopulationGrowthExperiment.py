from Experiments.experiments import *

"""This experiment was done close to deadline, so mostly AI-generated"""

class PopulationGrowthExperiment(BaseExperiment):
    """
    Analyzes the population dynamics (growth/shrinkage) of CA.
    Maps population change time-series to complexity.
    """

    def __init__(self, engine: ElementaryCA, metric: ComplexityMetric):
        self.engine = engine
        self.metric = metric
        self.results = {}  # rule -> list of binary sequences
        self.num_seeds_used = 0

    def run(self, num_seeds: int, rules=range(256)):
        """Generates population change sequences."""
        self.num_seeds_used = num_seeds
        seeds = [self.engine.generate_seed() for _ in range(num_seeds)]

        for rule in tqdm(rules, desc="Simulating Population Rules"):
            binary_sequences = []
            for seed in seeds:
                # Run simulation (including t=0)
                img = self.engine.run(rule, seed)

                # 1. Calculate population at each step
                # IMPORTANT: Cast to signed int to prevent underflow
                population = np.sum(img, axis=1).astype(int)

                # 2. Calculate differences between steps
                diffs = np.diff(population)

                # 3. Create binary sequence: 1 if grew/same, 0 if shrank
                binary_seq = (diffs >= 0).astype(np.uint8)

                binary_sequences.append(binary_seq)

            self.results[rule] = binary_sequences

        self.analyze_and_plot()

    def analyze_and_plot(self):
        """
        Calculates complexity, correlations, and fits the upper bound curve.
        """
        freq_map = Counter()
        complexity_map = {}

        # Flatten all sequences
        all_seqs = [seq for sublist in self.results.values() for seq in sublist]

        for seq in tqdm(all_seqs, desc="Analyzing Population Complexity"):
            # Hash the numpy array (sequence)
            h = seq.tobytes()
            freq_map[h] += 1

            if h not in complexity_map:
                # Calculate complexity of the 1D binary sequence
                complexity_map[h] = self.metric.calculate(seq)

        # ---------------- PREPARE DATA ----------------
        Ks = []
        log_probs = []
        total = sum(freq_map.values())

        for h, count in freq_map.items():
            Ks.append(complexity_map[h])
            log_probs.append(np.log10(count / total))

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

        # 2. Fit linear regression to the upper bound points (y = mx + c)
        # Using numpy polyfit or scipy linregress
        if len(unique_ks) > 1:
            slope, intercept, r_val, p_val, std_err = linregress(unique_ks, max_log_probs)
        else:
            slope, intercept = 0, 0

        # 3. Convert to form P(p) = 2^(-aK - b)
        # We fitted log10(P) = slope * K + intercept
        # P = 10^(slope*K + intercept) = 2^(log2(10) * (slope*K + intercept))
        # P = 2^( (slope * 3.32) * K + (intercept * 3.32) )
        # So: -a = slope * 3.3219  => a = -slope * 3.3219
        #     -b = intercept * 3.3219 => b = -intercept * 3.3219
        log2_10 = np.log2(10)
        a_param = -slope * log2_10
        b_param = -intercept * log2_10

        # ---------------- PLOTTING ----------------
        plt.figure(figsize=(10, 7))

        # 1. Scatter Plot
        plt.scatter(Ks, log_probs, alpha=0.4, c='purple', label='Phenotypes', edgecolors='none')

        # 2. Plot Fitted Line
        # Generate x values for the line
        x_line = np.linspace(min(Ks), max(Ks), 100)
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, color='red', linestyle='--', linewidth=2, label='Upper Bound Fit')

        # 3. Labels and Title
        plt.xlabel(f"Complexity ({self.metric.name()})")
        plt.ylabel("Log10 Probability")
        plt.title("Simplicity Bias in Population Dynamics")

        # 4. Info Box with Params and Stats
        stats_text = (
            f"Simulation Parameters:\n"
            f"  L = {self.engine.L}\n"
            f"  T = {self.engine.T}\n"
            f"  N_seeds = {self.num_seeds_used}\n\n"
            f"Correlations:\n"
            f"  Spearman = {spearman_corr:.3f}\n"
            f"  Pearson = {pearson_corr:.3f}\n\n"
            f"Fit P = 2^(-aK - b):\n"
            f"  a = {a_param:.3f}\n"
            f"  b = {b_param:.3f}"
        )

        # Place text box in upper right
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right', bbox=props)

        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower left')
        plt.show()


    def plot(self): pass
    def analyze(self): pass