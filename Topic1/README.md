# Topic 1

## Quantized Llama 3.2-1B MMLU Results

```json
{
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "quantization_bits": null,
  "timestamp": "20260225_030516",
  "device": "cpu",
  "duration_seconds": 211.21583,
  "overall_accuracy": 47.61904761904761,
  "total_correct": 120,
  "total_questions": 252,
  "subject_results": [
    {
      "subject": "astronomy",
      "correct": 75,
      "total": 152,
      "accuracy": 49.34210526315789
    },
    {
      "subject": "business_ethics",
      "correct": 45,
      "total": 100,
      "accuracy": 45.0
    }
  ]
}
```

## Multi-Model Benchmark Results

```json
{
  "meta-llama/Llama-3.2-3B-Instruct": {
    "astronomy": {
      "accuracy": 0.2894736842105263,
      "correct": 44,
      "total": 152,
      "time_real": 151.5119686126709
    },
    "business_ethics": {
      "accuracy": 0.45,
      "correct": 45,
      "total": 100,
      "time_real": 120.01293158531189
    }
  },
  "Qwen/Qwen2.5-1.5B-Instruct": {
    "astronomy": {
      "accuracy": 0.28289473684210525,
      "correct": 43,
      "total": 152,
      "time_real": 74.28629636764526
    },
    "business_ethics": {
      "accuracy": 0.44,
      "correct": 44,
      "total": 100,
      "time_real": 56.54457473754883
    }
  }
}
```
