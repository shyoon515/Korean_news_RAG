# vLLM Serve Commands

아래는 vLLM으로 세 가지 모델을 서빙하는 예시입니다.

## LG AI (EXAONE 1.2B)
```bash
vllm serve LGAI-EXAONE/EXAONE-4.0-1.2B --port 8000 --gpu-memory-utilization 0.8
```

## KT (Midm 2.0 Mini Instruct)
```bash
vllm serve K-intelligence/Midm-2.0-Mini-Instruct --port 8000 --gpu-memory-utilization 0.8
```

## NAVER (HyperCLOVAX SEED Text Instruct 1.5B)
```bash
vllm serve naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B --port 8000 --gpu-memory-utilization 0.8
```


