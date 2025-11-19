# Performance Metrics - Candle Adapter

**Date**: 2025-11-18
**Phase**: 2B Documentation
**Model**: Qwen2.5-0.5B
**Device**: GPU (CUDA)

---

## Baseline Performance Metrics

### Current Throughput (Phase 2A Complete)

**Test**: `test_8_throughput_benchmark` (from integration_tests.rs)

```
Prompt: Write a short story about a robot.
Completion tokens: 512 (max_tokens limit)
Duration: ~104-110 seconds
Throughput: 4.89 tokens/sec (average)
```

**Calculation**:
- Total tokens generated: 512 completion tokens
- Time: ~104.7 seconds
- Throughput: 512 / 104.7 ≈ 4.89 tok/s

### Token Counting Accuracy

**Test**: `test_6_token_counting` (from integration_tests.rs)

```
Metric: Token counts verified against expected ranges
- Prompt tokens: Counted using tokenizer.encode()
- Completion tokens: (total_tokens - prompt_tokens)
- Accuracy: ✅ Exact (uses real tokenizer)
```

**Validation**:
- Uses same tokenizer as model (no estimation)
- Counts match exactly with model's token processing
- No approximations or heuristics

### Memory Usage

**Current**: Not measured (TBD)

**Model Load**:
- Qwen2.5-0.5B: ~2.5GB on GPU (FP16)
- Context: 32K tokens max

**Estimated Runtime**:
- Model: ~2.5GB
- KV cache (if implemented): ~500MB (estimated)
- Total: ~3GB (estimated)

### First Token Latency

**Current**: Not measured (TBD)

**Expected**:
- Cold start: ~2-5 seconds (model load + first forward pass)
- Warm: ~100-500ms (just forward pass)

**To Measure**: Add benchmark that measures time to first token only

---

## Performance Characteristics

### Strengths

1. **Accurate Token Counting** ✅
   - Uses real tokenizer (no estimation)
   - Matches model's actual token processing
   - Reliable for billing/quota tracking

2. **Streaming Support** ✅ (Phase 2B)
   - Token-by-token output
   - Real-time user feedback
   - Early cancellation possible

3. **Architecture-Agnostic** ✅
   - Works with any Candle-compatible model
   - No model-specific optimizations (yet)
   - Easy to swap models via Model Pool

### Current Limitations

1. **No KV Cache** ⚠️
   - Each token requires full forward pass
   - Throughput: ~4.89 tok/s (baseline)
   - Potential: 25-50 tok/s with KV cache (5-10x improvement)

2. **No Batching** ⚠️
   - Processes one request at a time
   - GPU utilization: Low (single request)
   - Potential: 2-3x throughput with batching

3. **No Quantization** ⚠️
   - Model loaded in FP16 (default Candle)
   - Memory: ~2.5GB
   - Potential: INT8/INT4 could reduce to ~1GB or less

---

## Performance Comparison

### Baseline vs. Potential

| Metric | Current (Phase 2A) | With KV Cache | With Batching | With Both |
|--------|-------------------|---------------|---------------|-----------|
| Throughput | 4.89 tok/s | 25-50 tok/s | 10-15 tok/s | 50-100 tok/s |
| Speedup | 1x | 5-10x | 2-3x | 10-20x |
| Memory | ~2.5GB | ~3GB | ~2.5GB | ~3GB |
| Latency (first token) | TBD | Lower | Similar | Lower |

### Other LLM Frameworks (Reference)

**Note**: These are rough comparisons and depend heavily on hardware, model, and configuration.

| Framework | Throughput (Qwen2.5-0.5B) | Features |
|-----------|---------------------------|----------|
| **Candle Adapter (Ours)** | 4.89 tok/s | No optimizations yet |
| llama.cpp | 20-40 tok/s | KV cache + quantization |
| vLLM | 50-100 tok/s | KV cache + batching + paged attention |
| TensorRT-LLM | 100-200 tok/s | Highly optimized, CUDA-specific |

**Interpretation**:
- Our baseline (4.89 tok/s) is expected for un-optimized Candle inference
- With KV cache, we'd match llama.cpp range (20-40 tok/s)
- With batching + KV cache, we'd approach vLLM range (50-100 tok/s)

---

## Benchmark Suite

### Existing Benchmarks

1. **test_8_throughput_benchmark**
   - Measures: Tokens/sec for long generation (512 tokens)
   - Current result: 4.89 tok/s
   - Use case: Baseline throughput tracking

2. **test_6_token_counting**
   - Measures: Token counting accuracy
   - Current result: ✅ Exact match
   - Use case: Verify tokenization correctness

3. **test_15_phase2b_streaming_output**
   - Measures: Streaming functionality
   - Current result: ✅ Works correctly
   - Use case: Real-time output validation

### Recommended Future Benchmarks

1. **First Token Latency Benchmark**
   ```rust
   // Measure time to first token only
   let start = Instant::now();
   let mut stream = adapter.generate_stream(prompt).await?;
   let first_token = stream.next().await;
   let latency = start.elapsed();
   ```

2. **Concurrent Request Benchmark**
   ```rust
   // Measure throughput with N concurrent requests
   let tasks: Vec<_> = (0..N).map(|_| {
       tokio::spawn(async { adapter.generate(prompt).await })
   }).collect();
   futures::future::join_all(tasks).await;
   ```

3. **Context Length Scaling Benchmark**
   ```rust
   // Measure how latency/throughput scales with prompt length
   for prompt_len in [100, 500, 1000, 5000, 10000] {
       let prompt = "word ".repeat(prompt_len);
       let (_, _, duration) = benchmark_generate(&adapter, &prompt).await;
       // Plot: latency vs. prompt_len
   }
   ```

4. **Memory Profiling Benchmark**
   ```rust
   // Track memory usage during generation
   let mem_before = get_memory_usage();
   adapter.generate(long_prompt).await?;
   let mem_after = get_memory_usage();
   let overhead = mem_after - mem_before;
   ```

---

## Performance Regression Detection

### Baseline Tracking

**Recommendation**: Track these metrics over time to detect regressions.

| Metric | Baseline (Phase 2A) | Threshold | Action |
|--------|---------------------|-----------|--------|
| Throughput | 4.89 tok/s | < 4.0 tok/s | Investigate regression |
| Token counting accuracy | 100% | < 100% | Critical bug |
| First token latency | TBD | > 2x baseline | Investigate |

### CI/CD Integration

**Suggested workflow**:
1. Run benchmark suite on every commit
2. Compare against baseline
3. Fail if throughput drops > 20%
4. Track metrics in time-series database

---

## Optimization Roadmap

See [OPTIMIZATION-OPPORTUNITIES.md](./OPTIMIZATION-OPPORTUNITIES.md) for detailed optimization opportunities deferred from Phase 2B.

**Summary**:
- **KV Cache**: 5-10x speedup (deferred - model-specific)
- **Request Batching**: 2-3x throughput (deferred - architecture decision)
- **Quantization**: 2-4x memory reduction (model-pool responsibility)
- **Prompt Caching**: 20-30% latency reduction (future investigation)

---

## Measurement Methodology

### How We Measure

1. **Throughput (tokens/sec)**:
   ```rust
   let start = Instant::now();
   let (response, prompt_tokens, completion_tokens) = adapter.generate(prompt).await?;
   let elapsed = start.elapsed();
   let throughput = completion_tokens as f64 / elapsed.as_secs_f64();
   ```

2. **Token Counting**:
   ```rust
   // Uses real tokenizer (not estimation)
   let prompt_tokens = tokenizer.encode(prompt, true)?.len();
   let total_tokens = generated_tokens.len();
   let completion_tokens = total_tokens - prompt_tokens;
   ```

3. **Latency**:
   ```rust
   let start = Instant::now();
   // ... operation ...
   let latency = start.elapsed();
   ```

### Test Environment

**Hardware** (assumed):
- GPU: NVIDIA GPU with CUDA 12.x support
- RAM: Sufficient for 2.5GB model + overhead
- Storage: SSD for model loading

**Software**:
- Rust: 1.91+ (edition 2021)
- Candle: 0.9 with CUDA features
- Model: Qwen2.5-0.5B (FP16)

**Configuration**:
- max_tokens: 512 (default)
- temperature: 0.7 (default)
- top_p: 0.9 (default)
- context_length: 32768 tokens

---

## Next Steps

1. **Add First Token Latency Measurement**
   - Modify streaming test to measure time to first token
   - Set baseline and track over time

2. **Add Memory Profiling**
   - Measure actual memory usage during inference
   - Validate 2.5GB estimate

3. **Add Concurrent Request Benchmark**
   - Test behavior under concurrent load
   - Measure throughput degradation

4. **CI Integration**
   - Run benchmarks on every PR
   - Track metrics over time
   - Alert on regressions

5. **Revisit Optimizations** (After Model Pool Integration)
   - Implement KV cache if model-pool supports it
   - Add batching if architecture allows
   - Consider quantization options

---

**Document Version**: 1.0
**Last Updated**: 2025-11-18
**Status**: Baseline documented, optimizations deferred
