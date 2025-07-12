# lord-ai-core

This repository provides a simple simulation-based Powerball prediction script.

## Usage

Create a text file where each line contains the five main numbers and the
Powerball number separated by spaces:

```
1 3 12 24 59 6
5 14 22 32 33 12
...
```

Run the predictor with:

```
python -m simulation.powerball_prediction path/to/data.txt
```

The script outputs three candidate predictions with confidence and energy
scores.
