import argparse
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Prediction:
    numbers: Tuple[int, int, int, int, int, int]
    confidence: float
    energy: float
    rationale: str

class PowerballPredictor:
    def __init__(self, draws: pd.DataFrame):
        self.draws = draws
        self.number_range = range(1, 70)
        self.pb_range = range(1, 27)
        self._prepare_statistics()

    @classmethod
    def from_file(cls, path: str) -> "PowerballPredictor":
        data = []
        with open(path, 'r') as f:
            for line in f:
                parts = [int(x) for x in line.strip().split()[:6]]
                if len(parts) != 6:
                    continue
                data.append(parts)
        df = pd.DataFrame(data, columns=[f"N{i}" for i in range(1,6)] + ["PB"])
        return cls(df)

    def _prepare_statistics(self):
        df = self.draws
        self.freq = {}
        for col in df.columns:
            self.freq[col] = df[col].value_counts().to_dict()
        self.position_avgs = df[[f"N{i}" for i in range(1,6)]].mean()
        self.position_std = df[[f"N{i}" for i in range(1,6)]].std()
        self.pb_mod_counts = df['PB'] % 10
        self.pb_mod_freq = self.pb_mod_counts.value_counts().to_dict()

    def _score_number(self, num: int, col: str, recent_penalty: dict) -> float:
        freq = self.freq.get(col, {}).get(num, 0) + 1
        avg = self.position_avgs[col] if col != 'PB' else np.mean(list(self.pb_range))
        std = self.position_std[col] if col != 'PB' else np.std(list(self.pb_range))
        score = freq
        score *= np.exp(-abs(num - avg)/ (std+1))
        if num in recent_penalty:
            score *= recent_penalty[num]
        if col == 'PB':
            mod = num % 10
            mod_bias = self.pb_mod_freq.get(mod, 0) + 1
            score *= mod_bias
        return score

    def _generate_candidate(self, recent_penalty: dict) -> Tuple[int, int, int, int, int, int]:
        numbers = []
        used = set()
        for i in range(1,6):
            probs = np.array([self._score_number(n, f"N{i}", recent_penalty) for n in self.number_range])
            probs[list(used)] = 0
            probs /= probs.sum()
            choice = np.random.choice(list(self.number_range), p=probs)
            numbers.append(choice)
            used.add(choice)
        pb_probs = np.array([self._score_number(n, 'PB', recent_penalty) for n in self.pb_range])
        pb_probs /= pb_probs.sum()
        pb = np.random.choice(list(self.pb_range), p=pb_probs)
        numbers.append(pb)
        return tuple(numbers)

    def simulate_draws(self, sims: int = 200) -> List[Tuple[Tuple[int, int, int, int, int, int], float]]:
        recent_nums = list(self.draws.tail(5).stack())
        recent_penalty = {num:0.5 for num in recent_nums}
        candidates = []
        for _ in range(sims):
            cand = self._generate_candidate(recent_penalty)
            energy = self._evaluate_candidate(cand)
            candidates.append((cand, energy))
        return candidates

    def _evaluate_candidate(self, cand: Tuple[int, int, int, int, int, int]) -> float:
        weights = []
        for i, n in enumerate(cand[:-1], start=1):
            freq = self.freq[f"N{i}"].get(n,0)+1
            weights.append(freq)
        pb_freq = self.freq['PB'].get(cand[-1],0)+1
        weights.append(pb_freq)
        softmax = np.exp(weights)/np.sum(np.exp(weights))
        energy = softmax.mean()
        return energy

    def predict(self) -> List[Prediction]:
        candidates = self.simulate_draws()
        energies = np.array([energy for _, energy in candidates])
        threshold = np.quantile(energies, 0.9)
        top = [(c,e) for c,e in candidates if e >= threshold]
        unique = {}
        for cand, energy in top:
            unique.setdefault(cand, []).append(energy)
        draws = []
        for cand, energies in unique.items():
            conf = np.mean(energies)
            draws.append((cand, conf))
        draws.sort(key=lambda x: x[1], reverse=True)
        selected = draws[:3]
        predictions = []
        for cand, conf in selected:
            rationale = "Frequent numbers with position and PB bias"
            predictions.append(Prediction(cand, conf, conf, rationale))
        return predictions


def main():
    parser = argparse.ArgumentParser(description="Powerball Predictor")
    parser.add_argument('file', help='Historical draw file')
    args = parser.parse_args()
    predictor = PowerballPredictor.from_file(args.file)
    preds = predictor.predict()
    for idx, p in enumerate(preds, start=1):
        nums = ' '.join(str(n) for n in p.numbers[:-1])
        print(f"Prediction {idx}: {nums} PB:{p.numbers[-1]} | Confidence:{p.confidence:.2f} | Energy:{p.energy:.2f}")
        print(f"  Rationale: {p.rationale}")

if __name__ == '__main__':
    main()
