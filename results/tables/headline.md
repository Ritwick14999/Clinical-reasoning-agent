| Experiment | Model | Dataset | N | Answered | Accuracy (95% CI) | Hit@k | Tool P/R |
|---|---|---|---|---|---|---|---|
| final_llama31 | llama31-8k:8b | medqa | 300 | 291 | 58.1% [52.2%, 63.6%] | n/a | 95.0% / 90.5% (n=21) |
| final_llama31 | llama31-8k:8b | pubmedqa | 300 | 300 | 58.7% [53.0%, 64.0%] | 95.3% | 98.3% / 97.7% (n=300) |
| final_qwen3 | qwen3-8k:8b | medqa | 300 | 277 | 76.5% [71.5%, 81.2%] | n/a | n/a (n=21, no tools used) |
| final_qwen3 | qwen3-8k:8b | pubmedqa | 300 | 300 | 45.7% [40.3%, 51.3%] | 79.3% | 100.0% / 81.0% (n=300) |