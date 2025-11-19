# Optimization Opportunities - Candle Adapter

**Date**: 2025-11-18
**Phase**: 2B Documentation
**Status**: Deferred for Future Implementation

---

## Overview

This document catalogs performance optimization opportunities that were identified during Phase 2B but deferred due to architectural constraints. These optimizations are **model-specific** or require **architecture decisions** that should be made after the model-pool integration is complete.

**Why Deferred?**
- The adapter will be integrated with a model-pool that handles multiple different model architectures
- Model-specific optimizations (like KV cache) require tight coupling to specific model forward() implementations
- Batching strategies depend on how the model-pool manages concurrent requests
- Implementing these now would create technical debt when transitioning to the multi-model architecture

---

## 1. KV Cache (Key-Value Cache)

### Description

Cache the key and value tensors from attention layers to avoid recomputing them for each token generation step.

### Current Behavior

```rust
// Without KV cache:
for each new token:
    forward pass over ALL previous tokens + new token
    compute attention for entire sequence
    sample next token
```

**Problem**: Each token requires a full forward pass over the entire sequence, which is O(n²) complexity.

### With KV Cache

```rust
// With KV cache:
first token:
    forward pass over input tokens
    cache key/value tensors for each attention layer

subsequent tokens:
    forward pass over ONLY the new token
    reuse cached key/value tensors
    append new key/value to cache
```

**Benefit**: O(n) complexity instead of O(n²)

### Implementation Details

**Qwen2 Model Structure**:
```rust
pub struct Cache {
    key_cache: Vec<Tensor>,    // One per layer
    value_cache: Vec<Tensor>,  // One per layer
}

// In model.forward():
fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
    // Pass cache to each attention layer
    for layer in &mut self.layers {
        x = layer.forward(x, seqlen_offset, &mut self.cache)?;
    }
}
```

**Adapter Changes**:
```rust
pub struct CandleAdapter {
    model: Arc<LoadedModel>,
    config: CandleConfig,
    kv_cache: Option<Cache>,  // NEW: Store KV cache between generations
}

impl CandleAdapter {
    pub async fn generate(&self, prompt: &str) -> Result<(String, u64, u64)> {
        // Clear cache on new prompt
        self.kv_cache = Some(Cache::new(num_layers));

        // Pass cache to generate_tokens
        let tokens = generate_tokens(..., &mut self.kv_cache)?;

        // Cache is now populated and can be reused
    }
}
```

### Estimated Impact

| Metric | Without KV Cache | With KV Cache | Improvement |
|--------|------------------|---------------|-------------|
| Throughput | 4.89 tok/s | 25-50 tok/s | **5-10x** |
| Memory | ~2.5GB | ~3GB | +500MB |
| First token latency | Same | Lower (cached) | 20-30% faster |

### Why Deferred

**Reason**: Model-specific optimization

- **Tight coupling**: Requires modifying model.forward() signature to accept cache
- **Architecture-specific**: Different models (Qwen2, Llama, GPT) have different cache structures
- **Model pool conflict**: If model-pool manages multiple model types, each needs different cache implementation

**Trigger for Implementation**:
- Model-pool architecture supports model-specific optimizations
- Single model type is confirmed (or abstraction layer is designed)
- Model-pool can manage cache lifecycle

### References

- PHASE-2-CHECKLIST.md: Tasks 12-14
- Candle KV cache example: https://github.com/huggingface/candle/blob/main/candle-examples/examples/llama/main.rs

---

## 2. Request Batching

### Description

Process multiple inference requests in a single batch to improve GPU utilization and throughput.

### Current Behavior

```rust
// Sequential processing:
for request in requests:
    response = adapter.generate(request.prompt).await
    // GPU is idle between requests
```

**Problem**: GPU utilization is low when processing one request at a time.

### With Batching

```rust
// Batch processing:
let batch = collect_requests(max_batch_size);
let batch_inputs = pad_to_same_length(batch);  // Pad sequences

// Single forward pass for all requests
let batch_outputs = model.forward(&batch_inputs);

// Unbatch results
for (request, output) in zip(batch, batch_outputs):
    send_response(request.id, output)
```

**Benefit**: Higher GPU utilization, better throughput for concurrent requests

### Implementation Strategies

**Option 1: Static Batching**
```rust
// Collect requests until batch is full or timeout
let batch = wait_for_batch(max_size = 8, timeout = 50ms);
process_batch(batch).await;
```

**Option 2: Dynamic Batching**
```rust
// Continuously add requests to current batch
while let Some(request) = requests.recv().await {
    batch.push(request);
    if batch.len() >= max_size || time_since_last_batch > timeout {
        process_batch(batch).await;
        batch.clear();
    }
}
```

**Option 3: Continuous Batching** (like vLLM)
```rust
// Requests can finish at different times (different sequence lengths)
// More complex but highest efficiency
```

### Implementation Details

**Sequence Padding**:
```rust
fn pad_sequences(sequences: Vec<Vec<u32>>) -> Tensor {
    let max_len = sequences.iter().map(|s| s.len()).max().unwrap();

    // Pad all sequences to max_len
    for seq in &mut sequences {
        while seq.len() < max_len {
            seq.push(PAD_TOKEN_ID);
        }
    }

    // Convert to batch tensor: [batch_size, max_len]
    Tensor::new(&sequences.concat(), device)?.reshape((sequences.len(), max_len))
}
```

**Attention Masking**:
```rust
// Need to mask padding tokens in attention
fn create_attention_mask(sequences: &[Vec<u32>]) -> Tensor {
    // 1 for real tokens, 0 for padding
    // Shape: [batch_size, max_len]
}
```

### Estimated Impact

| Metric | Without Batching | With Batching (8x) | Improvement |
|--------|------------------|---------------------|-------------|
| Throughput (single req) | 4.89 tok/s | 4.89 tok/s | 1x |
| Throughput (8 concurrent) | 4.89 tok/s | 10-15 tok/s | **2-3x** |
| Latency (single req) | X ms | X ms | Same |
| Latency (concurrent) | X ms | X + batch_delay | Slightly higher |
| GPU Utilization | ~10-20% | ~60-80% | **Much better** |

### Why Deferred

**Reason**: Architecture decision needed

- **Model-pool design**: Batching strategy depends on how model-pool manages concurrent requests
- **Request routing**: Need to decide where batching logic lives (adapter vs. model-pool)
- **Complexity**: Requires careful coordination of request lifecycle, padding, masking

**Trigger for Implementation**:
- Model-pool architecture defines request management strategy
- Decision on where batching logic should live
- Need to support high concurrent load (>5-10 requests/sec)

### References

- PHASE-2-CHECKLIST.md: Task 16
- vLLM batching: https://vllm.readthedocs.io/en/latest/
- TensorRT-LLM batching: https://nvidia.github.io/TensorRT-LLM/

---

## 3. Model Quantization

### Description

Reduce model precision from FP16 to INT8 or INT4 to reduce memory usage and improve inference speed.

### Current Behavior

```rust
// Model loaded in FP16 (Candle default)
// Size: ~2.5GB for Qwen2.5-0.5B
```

### With Quantization

**INT8 Quantization**:
- Size: ~1.25GB (2x reduction)
- Accuracy: Minimal degradation (< 1%)
- Speed: 1.5-2x faster on some hardware

**INT4 Quantization**:
- Size: ~625MB (4x reduction)
- Accuracy: Some degradation (2-5%)
- Speed: 2-3x faster on some hardware

### Implementation Details

**Option 1: Candle Built-in Quantization**
```rust
// Load quantized model
let model = Qwen2Model::load_quantized(
    model_path,
    QuantizationType::Q8_0,  // or Q4_0, Q4_1, etc.
)?;
```

**Option 2: GGUF Format** (used by llama.cpp)
```rust
// Use pre-quantized GGUF models
// Requires separate model conversion step
```

**Option 3: Dynamic Quantization**
```rust
// Quantize on-the-fly during loading
// More flexible but slower load time
```

### Estimated Impact

| Metric | FP16 | INT8 | INT4 |
|--------|------|------|------|
| Model Size | 2.5GB | 1.25GB | 625MB |
| Memory Reduction | 1x | **2x** | **4x** |
| Inference Speed | 1x | 1.5-2x | 2-3x |
| Accuracy | Baseline | ~99% | ~95-98% |

### Why Deferred

**Reason**: Model-pool responsibility

- **Loading phase**: Quantization happens during model loading, not inference
- **Model management**: Model-pool is responsible for loading models
- **Format compatibility**: Need to decide on model format (safetensors, GGUF, etc.)

**Trigger for Implementation**:
- Model-pool supports quantized model loading
- Memory constraints require smaller models
- Decision on acceptable accuracy vs. size tradeoff

### References

- Candle quantization: https://github.com/huggingface/candle/tree/main/candle-transformers
- GGUF format: https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/README.md

---

## 4. Prompt Caching

### Description

Cache common prompt prefixes (like system messages, demonstrations) to avoid re-tokenizing and re-processing them.

### Current Behavior

```rust
// Every request:
tokenize(system_message + demonstrations + user_prompt)
forward_pass(all_tokens)
```

**Problem**: System messages and demonstrations are the same across requests but reprocessed every time.

### With Prompt Caching

```rust
// First request:
let prefix_tokens = tokenize(system_message + demonstrations);
let prefix_cache = forward_pass(prefix_tokens);  // Cache this

// Subsequent requests:
let user_tokens = tokenize(user_prompt);
let output = forward_pass_with_prefix(prefix_cache, user_tokens);  // Reuse prefix
```

**Benefit**: Skip tokenization and forward pass for common prefixes

### Implementation Details

**Cache Structure**:
```rust
struct PromptCache {
    prefix_text: String,           // The cached prefix
    prefix_tokens: Vec<u32>,       // Tokenized prefix
    prefix_kv_cache: Cache,        // KV cache for prefix
    last_used: Instant,            // For LRU eviction
}

struct PromptCacheManager {
    caches: HashMap<String, PromptCache>,
    max_size: usize,
}
```

**Usage**:
```rust
// Check cache
let prefix = signature.instruction() + format_demos(signature.demos());
if let Some(cached) = cache_manager.get(&prefix) {
    // Reuse cached prefix
    return generate_with_prefix(cached, user_input);
}

// Cache miss: compute and cache
let prefix_cache = compute_prefix(prefix);
cache_manager.insert(prefix, prefix_cache);
```

### Estimated Impact

| Metric | Without Caching | With Caching | Improvement |
|--------|-----------------|--------------|-------------|
| Latency (cached prefix) | X ms | 0.7X ms | **20-30% faster** |
| Latency (cache miss) | X ms | X ms | Same |
| Token processing | All tokens | Only new tokens | Depends on prefix size |

**Example Scenario**:
- System message: 50 tokens
- Demonstrations: 200 tokens (5 examples)
- User prompt: 20 tokens
- **Without caching**: Process 270 tokens
- **With caching**: Process 20 tokens (13.5x fewer for prefix!)

### Why Deferred

**Reason**: Future investigation

- **Requires KV cache**: Prompt caching builds on top of KV cache
- **Cache management**: Need LRU eviction, TTL, memory limits
- **Correctness**: Must ensure cache invalidation works correctly

**Trigger for Implementation**:
- KV cache is implemented
- Workload analysis shows repeated prefixes (>50% requests)
- Latency optimization is high priority

### References

- Anthropic prompt caching: https://docs.anthropic.com/claude/docs/prompt-caching
- vLLM prefix caching: https://docs.vllm.ai/en/latest/features/prefix_caching.html

---

## 5. Flash Attention

### Description

Use optimized attention kernels (Flash Attention) to speed up attention computation.

### Current Behavior

```rust
// Standard attention:
Q * K^T = scores     // O(n²) memory
softmax(scores)
scores * V = output
```

**Problem**: Attention is memory-bandwidth bound, especially for long contexts.

### With Flash Attention

```rust
// Flash Attention:
// Tiled computation that reduces memory I/O
// Same mathematical result, much faster
```

**Benefit**: 2-4x faster attention, especially for long contexts (>2K tokens)

### Estimated Impact

| Context Length | Standard | Flash Attention | Speedup |
|----------------|----------|-----------------|---------|
| 512 tokens | 1x | 1.5x | 1.5x |
| 2048 tokens | 1x | 2.5x | **2.5x** |
| 8192 tokens | 1x | 4x | **4x** |

### Why Deferred

**Reason**: Candle/model support

- **Library support**: Requires Flash Attention implementation in Candle
- **Hardware requirement**: Works best on newer NVIDIA GPUs (A100, H100)
- **Model modification**: Requires changing attention layer implementation

**Trigger for Implementation**:
- Candle adds Flash Attention support
- Long context use cases (>2K tokens) are common
- Hardware supports Flash Attention 2

### References

- Flash Attention paper: https://arxiv.org/abs/2205.14135
- Flash Attention 2: https://arxiv.org/abs/2307.08691

---

## 6. Speculative Decoding

### Description

Use a small "draft" model to propose multiple tokens, then verify with the large model in parallel.

### How It Works

```rust
// Standard:
for i in 0..max_tokens:
    next_token = large_model.forward(context)  // Slow

// Speculative:
draft_tokens = small_model.generate(context, k=5)  // Fast, k tokens
verified = large_model.verify(context, draft_tokens)  // Parallel verification
accepted = verified[:first_rejection]
```

**Benefit**: If draft model is good, can generate multiple tokens per large model call

### Estimated Impact

| Draft Model Quality | Speedup |
|---------------------|---------|
| Poor (50% acceptance) | 1.5x |
| Good (70% acceptance) | 2x |
| Excellent (90% acceptance) | **3-4x** |

### Why Deferred

**Reason**: Complex implementation

- **Requires second model**: Need to load and manage a smaller draft model
- **Model compatibility**: Draft and target models must have compatible vocabularies
- **Complexity**: Verification logic is non-trivial

**Trigger for Implementation**:
- Other optimizations (KV cache, batching) are exhausted
- Need further speedup without hardware upgrade
- Good draft model is available for target model

### References

- Speculative Decoding paper: https://arxiv.org/abs/2211.17192
- Medusa: Multi-head speculative decoding: https://arxiv.org/abs/2401.10774

---

## Priority Ranking

Based on **impact vs. complexity** and **architectural readiness**:

1. **KV Cache** (Priority: HIGH)
   - Impact: 5-10x speedup
   - Complexity: Medium
   - Blocker: Model-pool architecture
   - **Recommendation**: Implement first after model-pool integration

2. **Request Batching** (Priority: MEDIUM-HIGH)
   - Impact: 2-3x throughput for concurrent requests
   - Complexity: Medium-High
   - Blocker: Model-pool request management design
   - **Recommendation**: Implement after KV cache if concurrent load is high

3. **Model Quantization** (Priority: MEDIUM)
   - Impact: 2-4x memory reduction, 1.5-2x speedup
   - Complexity: Low (if Candle supports it)
   - Blocker: Model-pool loading mechanism
   - **Recommendation**: Implement if memory is constrained

4. **Prompt Caching** (Priority: LOW-MEDIUM)
   - Impact: 20-30% latency reduction (for cached prefixes)
   - Complexity: Medium
   - Blocker: Requires KV cache first
   - **Recommendation**: Implement after KV cache if workload has repeated prefixes

5. **Flash Attention** (Priority: LOW)
   - Impact: 2-4x for long contexts
   - Complexity: High (requires library support)
   - Blocker: Candle support, hardware
   - **Recommendation**: Revisit if Candle adds support and long contexts are common

6. **Speculative Decoding** (Priority: VERY LOW)
   - Impact: 2-4x (if good draft model)
   - Complexity: Very High
   - Blocker: Requires second model, complex verification
   - **Recommendation**: Only if other optimizations are exhausted

---

## Action Items

**Phase 2B** (Current):
- ✅ Document baseline performance
- ✅ Document optimization opportunities
- ✅ Create this document

**Phase 3** (After Model-Pool Integration):
1. **Revisit KV Cache**
   - Assess model-pool architecture
   - Design cache lifecycle management
   - Implement if feasible

2. **Assess Batching Need**
   - Measure concurrent request patterns
   - Design batching interface
   - Implement if load justifies it

3. **Evaluate Quantization**
   - Test quantized models for accuracy
   - Measure memory/speed tradeoff
   - Implement if model-pool supports it

**Phase 4** (Future Optimizations):
- Prompt caching (after KV cache)
- Flash Attention (if Candle supports it)
- Speculative decoding (if needed)

---

**Document Version**: 1.0
**Last Updated**: 2025-11-18
**Status**: Cataloged for future implementation
