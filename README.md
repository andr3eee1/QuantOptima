# QComplete 

Hey! I'm a 16-year-old high school student, and this is my independent research project on neural network quantization. 

Basically, Large Language Models (like GPT) use billions of numbers (weights). Storing them in standard 32-bit formats takes up too much space, so the industry compresses them to 8 or 4 bits. But when you only have 4 bits, you can only pick exactly 16 unique values to represent your numbers. 

**The big question I'm trying to answer is:** If we know exactly how the numbers in a neural network are distributed, what is the mathematically *perfect* set of 16 values we can pick? And how much accuracy are standard formats (like INT8, FP8, or NF4) leaving on the table compared to this theoretical optimum?

## What I'm doing
1. **Building from scratch:** Writing custom encoders/decoders for INT8, FP8 (E4M3), NF4, and Posit8 without using ready-made quantization libraries.
2. **Training & Extracting:** Training a small model (MLP/CNN) and pulling out the actual weight distributions to see what they look like.
3. **Finding the Optimum:** Using the Lloyd-Max algorithm (basically 1D k-means) to find the absolute best 16 values for each specific tensor.
4. **Benchmarking:** Comparing the accuracy of my "perfect" format against the standard ones to see if there's a big gap.

## Tech Stack
* Python
* PyTorch (for the models)
* NumPy & Matplotlib (for math and histograms)

## Repository Structure

Here is how the code is organized. (Work in progress!)

```text
├── README.md               # You are here! Project overview and (later) the final report.
├── requirements.txt        # Python dependencies
├── src/                    # The core logic
│   ├── formats/            # Hand-written encoders and decoders
│   │   ├── __init__.py
│   │   ├── int_fmt.py      # INT4 / INT8 logic
│   │   ├── fp_fmt.py       # FP4 / FP8 (E4M3) logic
│   │   ├── nf4_fmt.py      # NormalFloat4 implementation
│   │   └── posit_fmt.py    # Manual bit-level implementation for Posit
│   ├── model/              # Neural network architectures
│   │   ├── __init__.py
│   │   └── network.py      # Basic MLP/CNN setup
│   ├── quantization/       # Optimization algorithms
│   │   ├── __init__.py
│   │   ├── lloyd_max.py    # 1D k-means to find the perfect format
│   │   └── apply.py        # Applying the codebooks to the weights
│   └── utils/              # Helper functions
│       ├── metrics.py      # MSE and accuracy calculators
│       └── seed.py         # Setting fixed seeds for reproducibility
├── experiments/            # Isolated scripts for each project phase
│   ├── 00_test_random.py   # Testing formats on 1M random numbers
│   ├── 01_train_fp32.py    # Training the base model and getting histograms
│   ├── 02_benchmark.py     # Testing the standard formats
│   ├── 03_optimal_fmt.py   # Testing the Lloyd-Max optimum format
│   └── 04_kurtosis.py      # Testing synthetic distributions
├── tests/                  # Unit tests
│   └── test_posit.py       # Making sure my Posit logic actually works
└── results/                # Where the outputs go
    ├── histograms/         # Visualizing the tensor shapes
    ├── models/             # Saved base FP32 models
    └── figures/            # Final graphs comparing everything
```

## Current Status
- [ ] Phase 0: Setup and manual format implementation.
- [ ] Phase 1: Standard format tournament.
- [ ] Phase 2: Building the perfect format (Lloyd-Max).
- [ ] Phase 3: Analyzing why certain formats win (Kurtosis & Outliers).