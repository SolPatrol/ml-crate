# dspy-rs Architecture Summary

## Overview

dspy-rs is a complete agent system that handles agent selection, context formatting, and LLM generation with tool calling.

---

## Single Entry Point

```
dspy_rs.execute(agent_type, game_state, request) → AgentOutput { text, tool_calls }
```

---

## Internal Flow (3 Steps)

```
┌──────────────────────────────────────────────┐
│              dspy-rs                         │
│                                              │
│  Step 1: Agent Registry                     │
│  ┌────────────────────────────────────────┐ │
│  │ Input:  agent_type ("StoryTeller")     │ │
│  │ Does:   Select DSPy module             │ │
│  │ Output: ReAct<Agent> module            │ │
│  └────────────────────────────────────────┘ │
│              ↓                               │
│  Step 2: Context Builder                    │
│  ┌────────────────────────────────────────┐ │
│  │ Input:  agent_type + game_state        │ │
│  │ Does:   Format relevant game state     │ │
│  │ Output: "Narrative: ..., Health: ..."  │ │
│  └────────────────────────────────────────┘ │
│              ↓                               │
│  Step 3: Generate & Tool Calling            │
│  ┌────────────────────────────────────────┐ │
│  │ Input:  module + context + request     │ │
│  │ Does:   ReAct/ChainOfThought execution │ │
│  │ Output: response + tool_calls          │ │
│  └────────────────────────────────────────┘ │
│                                              │
└──────────────────────────────────────────────┘
```

---

## What Each Step Does

### Step 1: Agent Registry
**Purpose:** Choose which agent to use

**Process:**
- Input: agent_type ("StoryTeller", "MapBuilder", "RuleEnforcer", "NPC")
- Does: Selects the appropriate DSPy module from storage
- Output: The selected ReAct<Agent> or ChainOfThought<Agent> module

**Example:**
- Request asks for "StoryTeller"
- Registry returns the StoryTeller DSPy module

---

### Step 2: Context Builder
**Purpose:** Format game state data that's relevant to the selected agent's role

**Process:**
- Input: agent_type + current game_state
- Does: Formats only the game state data relevant to this agent type
- Output: Context string with relevant information

**Example:**
- StoryTeller gets: "Narrative: ancient ruins, Player health: 80, Active quests: Find artifact"
- MapBuilder gets: "Biome: forest, Difficulty: hard, Nearby areas: cave, village"
- RuleEnforcer gets: "Rules: no magic in town, Player state: casting spell"

**Key Point:** Different agents get different context based on their role

---

### Step 3: Generate & Tool Calling
**Purpose:** Run the DSPy predictor to generate response and identify tool calls

**Process:**
- Input: Selected agent module + formatted context + player request
- Does: Executes ReAct or ChainOfThought predictor
- Output: response text + tool_calls (if any)

**Predictor Options:**
- **ReAct**: For agents that need tools (think → act → observe → repeat)
- **ChainOfThought**: For reasoning without tools (think → answer)
- **Predict**: Simple direct responses

**Example Output:**
- text: "You hear rustling in the bushes. A goblin emerges!"
- tool_calls: [{ name: "spawn_enemy", params: { type: "goblin", location: "bushes" } }]

---

## Complete Game Flow

```
┌──────────────────────────────────┐
│      Game Request                │
│  + Current Game State            │
└──────────────────────────────────┘
              ↓
┌──────────────────────────────────┐
│         dspy-rs                  │
│  (Registry → Context → Generate) │
└──────────────────────────────────┘
              ↓
┌──────────────────────────────────┐
│      Tool Registry               │
│  (Execute tools in game)         │
└──────────────────────────────────┘
              ↓
┌──────────────────────────────────┐
│      Game Response               │
└──────────────────────────────────┘
```

---

## Key Points

### What dspy-rs IS:
- ✅ Complete agent system (all 3 steps)
- ✅ Stores all agent programs
- ✅ Formats context dynamically based on agent type
- ✅ Handles generation + tool identification
- ✅ Single `.execute()` method

### What dspy-rs is NOT:
- ❌ Not just generation (it does selection + context too)
- ❌ Not just registry (it does all 3 steps)
- ❌ Not tool execution (that's your game's Tool Registry)

### Dynamic vs Static:
- **Static:** Agent programs (loaded once)
- **Dynamic:** Game context (injected fresh every request)

### Core Principle:
- One model, multiple programs
- Programs = different optimized prompts per agent type
- Dynamic context injection per request

---

## Agent Types

Each agent type gets different context and may use different tools:

- **StoryTeller**: Narrative context → spawn enemies, trigger events
- **MapBuilder**: Map/biome context → generate areas, place objects
- **RuleEnforcer**: Rules context → validate actions, apply penalties
- **NPC (various)**: NPC-specific context → dialogue, trade, quest management

---

## Optimization (Optional)

dspy-rs programs can be optimized offline:

1. Collect training examples for each agent type
2. Run DSPy optimizer (BootstrapFewShot, MIPRO, etc.)
3. Save optimized programs
4. Load at runtime

**Effect:** Better quality responses, same speed

**When to optimize:**
- You have 50+ training examples
- Quality/immersion matters
- You want data-driven improvements

**When to skip:**
- Prototyping
- No training data
- Simple responses work fine

---

## Summary Table

| Step | Input | Output | Purpose |
|------|-------|--------|---------|
| **Agent Registry** | agent_type | DSPy module | Select which agent |
| **Context Builder** | agent_type + game_state | context string | Format relevant data |
| **Generate & Tool Calling** | module + context + request | text + tool_calls | Run ReAct/ChainOfThought |
