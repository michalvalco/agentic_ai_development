# ReAct Pattern: Reasoning + Acting

**Source:** Yao et al. (2022), "ReAct: Synergizing Reasoning and Acting in Language Models" (arXiv:2210.03629)  
**Date Accessed:** 2025-11-06  
**Relevance:** The ReAct pattern is THE theoretical foundation for all modern agentic AI systems. It's the pattern underlying Anthropic's tool use, LangChain's agents, and ultimately all five of our capabilities. Understanding ReAct is understanding why agents work at all.

---

## The Core Insight

**The Problem ReAct Solves:**

Before ReAct, LLM capabilities were split:
- **Reasoning (Chain-of-Thought):** LLMs could think through problems step-by-step, but were limited to internal knowledge → led to hallucination and error propagation
- **Acting (Action Generation):** LLMs could plan actions in environments, but lacked abstract reasoning → couldn't maintain high-level goals or adapt plans

**ReAct's Breakthrough:**

Combine reasoning and acting in an **interleaved** manner:
- **Reasoning traces** help the model induce, track, and update action plans and handle exceptions
- **Actions** allow the model to interface with external sources (knowledge bases, environments) to gather additional information

**The Synergy:**
- **Reason to Act:** Dynamic reasoning creates, maintains, and adjusts plans for acting
- **Act to Reason:** Interaction with external environments incorporates additional information into reasoning

---

## The ReAct Loop

The fundamental pattern:

```
1. THOUGHT: Reason about what to do next
2. ACTION: Take a specific action (call tool, search, etc.)
3. OBSERVATION: Receive feedback from the environment
4. REPEAT: Back to step 1 until task is solved
```

This loop continues until the model has sufficient information to solve the user's task.

### Concrete Example from Paper (Question Answering)

**Question:** "Aside from the Apple Remote, what other device can control the program Apple Remote was originally designed to interact with?"

**ReAct Trajectory:**

```
Thought 1: I need to search Apple Remote and find the program it was originally designed to interact with.
Action 1: Search[Apple Remote]
Observation 1: The Apple Remote is a remote control introduced by Apple Inc. for use with Apple TV...

Thought 2: Apple Remote was originally designed to interact with Apple TV. I need to search Apple TV and find what other device can control it.
Action 2: Search[Apple TV]
Observation 2: Apple TV is a digital media player... It can be controlled by Apple Remote, iPhone, iPad, or iPod Touch...

Thought 3: iPhone, iPad, or iPod Touch can control Apple TV. I need to find which one is not Apple Remote.
Action 3: Finish[iPhone]
```

**Key Observations:**
- Each thought builds on previous observations
- Actions are grounded in what was just learned
- The model adapts its plan based on what it discovers

---

## Comparison to Alternatives

### Chain-of-Thought (CoT) Only

**Approach:** Pure internal reasoning, no external actions

**Problems:**
- Limited to model's internal knowledge (training data cutoff)
- Can hallucinate facts
- Errors propagate through reasoning chain
- No way to verify or update knowledge

**Example Failure:**
```
Question: What year was the programmer born who created PyTorch?
CoT: PyTorch was created by Facebook AI Research. The lead developer was likely Soumith Chintala. 
He was probably born around 1985... [continues with hallucinated reasoning]
```

### Act-Only

**Approach:** Generate actions without explicit reasoning

**Problems:**
- Can't decompose complex goals into subgoals
- No ability to track progress or handle exceptions
- Fails to synthesize information from multiple observations
- Lacks interpretability—hard to understand *why* actions were chosen

**Example Failure:**
Even with correct actions and observations, fails to synthesize the final answer because it lacks reasoning capability.

### ReAct (Reasoning + Acting)

**Advantages:**
- ✅ Grounds reasoning in factual observations
- ✅ Adapts plans based on new information
- ✅ Handles exceptions and reformulates strategies
- ✅ Human-interpretable trajectories
- ✅ Combines internal knowledge (CoT) with external information (actions)

**Best Practice from Paper:**
Combine ReAct + CoT for optimal results—use internal knowledge when confident, external tools when uncertain.

---

## Types of Reasoning Traces

The paper identifies several useful reasoning patterns:

### 1. Task Decomposition

"I need to search X and find Y, then search Y and find Z"

### 2. Injecting Commonsense Knowledge
"X is not Y, so Z must instead be..."

### 3. Extracting Important Information
"From the observation, the key fact is..."

### 4. Tracking Progress
"So far I've found X, now I need Y"

### 5. Handling Exceptions
"That search failed. Maybe I can search/look up X instead"

### 6. Search Reformulation
"The result was not useful. Let me try a different query"

### 7. Arithmetic Reasoning
"1844 < 1989, so..."

### 8. Synthesizing Final Answer
"Based on observations, the answer is X"

---

## Implementation Strategies

### Task-Dependent Prompting

**For Reasoning-Heavy Tasks (e.g., Q&A):**
Alternate Thought-Action-Observation in every step:
```
Thought → Action → Observation → Thought → Action → Observation → ...
```

**For Decision-Making Tasks (e.g., navigation, games):**
Use **sparse reasoning**—thoughts only at critical junctures:
```
Action → Action → Thought → Action → Action → Action → Thought → ...
```

The LLM decides asynchronously when to think vs. act.

### Few-Shot Prompting

ReAct uses in-context learning with 1-6 examples:
- Human-written reasoning traces
- Domain-specific actions
- Environment observations

**Key Finding:** More examples don't always improve performance. 1-2 well-designed examples often suffice.


### Combining ReAct + CoT

The paper's best results came from hybrid approaches:

**Strategy A: ReAct → CoT fallback**
- Try ReAct first
- If fails to return answer within N steps, fall back to CoT
- Use when external info might help, but internal knowledge could suffice

**Strategy B: CoT → ReAct fallback**
- Try CoT first
- If majority vote confidence is low (< 50%), fall back to ReAct
- Use when internal knowledge might not be sufficient

---

## Empirical Results

### Question Answering (HotPotQA) & Fact Verification (Fever)

- ReAct outperformed Act-only baselines
- ReAct competitive with CoT, but reduces hallucination
- **Best:** ReAct + CoT combination

**Key Insight:** ReAct overcame hallucination by grounding in Wikipedia API interactions

### Decision Making (ALFWorld, WebShop)

- ReAct outperformed imitation learning by 34% (ALFWorld)
- ReAct outperformed reinforcement learning by 10% (WebShop)
- Used only 1-2 in-context examples (vs. expensive policy learning)

---

## Limitations & Failure Modes

### 1. Dependence on Tool Quality

**Problem:** ReAct depends heavily on the information it retrieves

**Example Failure:** Non-informative search results derail reasoning, making it difficult to recover

**Mitigation:** Design tools that return rich, relevant results with clear error messages


### 2. Structural Constraint Reduces Flexibility

**Problem:** Interleaving reasoning-action-observation reduces flexibility in formulating reasoning steps

**Result:** Higher reasoning error rate on some tasks vs. pure CoT

**Trade-off:** More grounded and trustworthy, but less flexible

### 3. Step Limits

**Problem:** Complex tasks may require more steps than allocated

**Finding from Paper:** 
- HotPotQA: 7 steps optimal (more didn't help)
- Fever: 5 steps optimal

**Implication:** Need to design tasks and tools that can solve problems within reasonable step limits

### 4. Tool Failure Cascades

**Problem:** If an action fails, the model might not recover gracefully

**Solution:** Build error handling into both tools (return informative errors) and prompts (include exception-handling examples)

---

## Connection to Our Five Capabilities

### 1. Prompt Routing

**ReAct Application:**
- **Thought:** "What kind of query is this? Internal data or external search?"
- **Action:** Route to appropriate destination
- **Observation:** Confirm routing succeeded
- **Note:** For simple routing, one Thought-Action cycle suffices (this is the "Router" architecture)

**Pattern:**
```
Thought: User is asking about company policy → route internally
Action: route_internal()
Observation: Routing confirmed
```

### 2. Query Writing

**ReAct Application:**
- **Thought:** "What data do I need? What filters? What order?"
- **Action:** Construct and execute query
- **Observation:** Analyze query results
- **Thought:** "Do I need to refine the query?"
- **Action (if needed):** Reformulate query with adjusted parameters

**Pattern:**
```
Thought: User wants sales data for Q3, filtered by region
Action: write_query(table="sales", filters={"quarter": "Q3", "region": "West"}, order_by="date")
Observation: Query returned 47 results
Thought: Results look good, proceeding to summarize
```

### 3. Data Processing

**ReAct Application:**
- **Thought:** "What transformations are needed?"
- **Action:** Apply transformation (extract, clean, enrich)
- **Observation:** Check transformation result
- **Thought:** "Does data quality meet requirements? What's next?"
- **Action (sequential):** Apply next transformation

**Pattern (Multi-step):**
```
Thought: Raw data needs cleaning, then enrichment
Action: clean_data(remove_nulls=True, standardize=True)
Observation: Cleaned 1000 records, 50 nulls removed
Thought: Now ready for enrichment with external API
Action: enrich_data(api="geocoding")
Observation: Enriched 950 records successfully
Thought: Data ready for analysis
```

### 4. Tool Orchestration

**ReAct Application:**
- **THIS IS THE PATTERN:** Tool orchestration IS the ReAct loop
- **Thought:** "Which tool should I use? In what order?"
- **Action:** Call tool(s)
- **Observation:** Process tool results
- **Thought:** "What tool is needed next? Or am I done?"

**Pattern (Complex orchestration):**
```
Thought: Need to fetch user data, then check permissions, then update record
Action: fetch_user(user_id=123)
Observation: User exists, has "editor" role
Thought: User has sufficient permissions, proceeding
Action: update_record(record_id=456, data={...})
Observation: Record updated successfully
Thought: Task complete
```

### 5. Decision Support & Planning

**ReAct Application:**
- **Thought:** "What are the options? What criteria matter?"
- **Action:** Gather information about each option
- **Observation:** Analyze gathered data
- **Thought:** "Based on observations, which option is best?"
- **Action:** Recommend decision with reasoning

**Pattern (Multi-step analysis):**
```
Thought: User needs to choose between three vendors. I should compare pricing, features, and reviews.
Action: get_vendor_info(vendor="A")
Observation: Vendor A: $100/month, 50 features, 4.5 stars
Thought: Good baseline, checking vendor B
Action: get_vendor_info(vendor="B")
Observation: Vendor B: $80/month, 45 features, 4.8 stars
Thought: Better value, checking final option
Action: get_vendor_info(vendor="C")
Observation: Vendor C: $120/month, 60 features, 4.2 stars
Thought: Based on observations, Vendor B offers best value/quality ratio
Action: recommend(vendor="B", reasoning="Best balance of cost, features, and satisfaction")
```

---

## Our Takeaways

### For Agentic_AI_Development

**1. ReAct Is the Universal Pattern**

Everything we're building follows this loop:
- **Routing:** Thought about intent → Action to route
- **Query Writing:** Thought about data needs → Action to construct query
- **Data Processing:** Thought about transformations → Action to transform
- **Tool Orchestration:** Thought about tool sequence → Action to call tools
- **Decision Support:** Thought about options → Action to recommend

There are no five separate patterns. There is ONE pattern applied five different ways.

**2. Observations Must Be Rich and Informative**

The quality of the loop depends on observation quality. Our tools must return:
- ✅ Clear, structured results
- ✅ Informative error messages (not just "failed")
- ✅ Enough context for next reasoning step
- ✅ Indication of confidence/completeness

**3. Prompt Design Determines Success**

From the paper's findings:
- Include 1-2 high-quality examples (more doesn't always help)
- Show diverse reasoning types (decomposition, reformulation, exception handling)
- Use domain-specific actions appropriate to the task
- For reasoning-heavy tasks: alternate thought-action-observation strictly
- For action-heavy tasks: use sparse reasoning at key decision points

**4. Step Limits Are Real**

Don't design systems that require infinite iteration:
- Set maximum steps (7-10 is typical)
- Design tools that provide sufficient information per call
- Enable efficient information gathering
- Build in fallback strategies when step limit reached

**5. Error Recovery Must Be Explicit**

The loop breaks when observations are unhelpful. Design for recovery:
- Tools return detailed error messages
- Prompts include reformulation examples
- Agent can recognize dead ends and pivot
- Alternative strategies are available

**6. Combine Internal and External Knowledge**

The paper's best results: ReAct + CoT
- Use internal reasoning when model is confident
- Use external tools when uncertainty is high
- Design systems that switch seamlessly between modes

**Our Implementation:** For each capability, decide upfront when to rely on model knowledge vs. when to call tools.

**7. Interpretability Is a Feature, Not a Bug**

ReAct's explicit reasoning traces make debugging possible:
- See where reasoning went wrong
- Identify bad tool calls
- Spot hallucinations
- Understand decision process

**Design principle:** Log all thoughts, actions, and observations. This isn't overhead—it's essential for production systems.

**8. Grounding Solves Hallucination**

Pure reasoning (CoT) hallucinates. ReAct grounds reasoning in facts:
- Every claim can be traced to an observation
- Observations come from external, verifiable sources
- The model can't "make up" data—it must retrieve it

**Implication:** Our tools are the grounding mechanism. Their reliability determines overall system reliability.

---

## Implementation Checklist

Based on ReAct principles:

### Tool Design
- [ ] Tools return structured, informative results
- [ ] Error messages are detailed and actionable
- [ ] Results include enough context for next step
- [ ] Tools are designed to provide value in 1-2 calls (not 10+)

### Prompt Design
- [ ] 1-2 high-quality examples included
- [ ] Examples show diverse reasoning patterns
- [ ] Domain-specific actions clearly defined
- [ ] Exception handling demonstrated

### System Architecture
- [ ] Maximum step limit defined (typically 7-10)
- [ ] Fallback strategies for step limit
- [ ] Logging of all thoughts/actions/observations
- [ ] Combine internal knowledge (CoT) with external tools (ReAct)

### Testing
- [ ] Verify reasoning traces are interpretable
- [ ] Check recovery from tool failures
- [ ] Validate that observations lead to correct next actions
- [ ] Ensure step limits are sufficient but not excessive

---

## The Fundamental Truth

**ReAct revealed:** Reasoning alone isn't enough. Acting alone isn't enough. The synergy between the two—the back-and-forth dance of thinking and doing, with observations grounding each step—this is what makes agents work.

Every modern agentic system, whether it explicitly mentions ReAct or not, implements this pattern. Anthropic's tool use is ReAct. LangChain's agents are ReAct. Our five capabilities are variations on ReAct.

**Master this pattern, and you master agentic AI.**

---

## References & Further Reading

**Original Paper:**
- Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. ICLR 2023. arXiv:2210.03629

**Related Work:**
- Chain-of-Thought Prompting (Wei et al., 2022) - The reasoning foundation
- LangChain ReAct Agent - Practical implementation
- Anthropic Tool Use - Industrial application of ReAct principles

---

**Summary:** ReAct combines reasoning and acting in an interleaved loop: Thought → Action → Observation → Repeat. This pattern grounds reasoning in factual observations from external sources, solving hallucination problems in pure reasoning while adding strategic thinking to pure action. It's the theoretical foundation for all modern agents. All five of our capabilities are ReAct applications.
