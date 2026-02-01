# Cross-Cultural Bias in Large Language Models

Detection, measurement, and mitigation of cultural bias in LLMs using World Values Survey data.

## Research Questions

- **RQ1:** Do models exhibit bias toward culturally aligned countries?
- **RQ2:** Does LoRA fine-tuning reduce bias for worst-case personas?
- **RQ3:** Does fine-tuning introduce negative side effects?

## Data

- World Values Survey Wave 7 (2017-2022)
- Countries: China, Slovakia, USA
- Question: Importance of God (Q164, scale 1-10)
- Personas: 63 (Country × Sex × Age × Education)

## Models

- Gemma 3 12B (Google, USA)
- Bielik 11B (SpeakLeash, Poland)
- Qwen 3 4B (Alibaba, China)

## Structure
```
├── data/           # Raw and processed data
├── scripts/        # R and Python scripts
├── results/        # Experimental results
├── paper/          # LaTeX manuscript
└── docs/           # Documentation and prompts
```

## Author

Antoni Czolgowski, University of Colorado Boulder
