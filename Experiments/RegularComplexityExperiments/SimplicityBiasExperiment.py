import utils
from Experiments.experiments import *


class SimplicityBiasExperiment(BaseExperiment):
    """
    Tests the AIT hypothesis: P(x) ~ 2^-K(x).
    Generates distribution plots.
    """

    def __init__(self, engine: ElementaryCA, metric: ComplexityMetric):
        super().__init__(engine, metric)
        self.results = {}  # rule -> list of images
        self.num_seeds_used = 0

    def run(self, num_seeds: int, rules=range(256), shuffle_control=False):
        """Generates data."""
        self.num_seeds_used = num_seeds
        seeds = [self.engine.generate_seed() for _ in range(num_seeds)]

        # tqdm displays a progress bar in the console
        for rule in tqdm(rules, desc="Simulating Rules"):
            phenotypes = []
            for seed in seeds:
                # Discard t=0 (seed) usually
                img = self.engine.run(rule, seed)[1:]
                phenotypes.append(img)
            self.results[rule] = phenotypes

        freqs, comps = self.analyze(shuffle_control)
        upr_bound_pltr = utils.UpperBoundPlotter(self.metric, self.engine, num_seeds)
        upr_bound_pltr.plot(freqs, comps, title=f"Simplicity Bias Experiment, Shuffled = {shuffle_control} ")

    def analyze(self, shuffle_control=False):
        """
        Hashes phenotypes and calculates complexity.
        returns: freq_map, complexity_map
            freq_map: Hash of image -> count
            complexity_map: Dict hash of image -> complexity score
            """
        # Dict that will store the count of each unique phenotype.
        # Hash of image -> count
        freq_map = Counter()
        # Dict hash of image -> complexity score
        complexity_map = {}

        # Flatten all results (which are a list of lists) into one big pool of phenotypes
        all_images = [img for sublist in self.results.values() for img in sublist]

        for img in tqdm(all_images, desc="Analyzing Complexity"):
            # Optional: Shuffle image to test control
            if shuffle_control:
                img = shuffle_image_rows(img)

            h = img.tobytes()  # Hash
            freq_map[h] += 1

            if h not in complexity_map:
                complexity_map[h] = self.metric.calculate(img)

        return freq_map, complexity_map

    # TODO maybe delete from the abstract class
    # In this case, the plotting is handled by utils.UpperBoundPlotter
    def plot(self): pass