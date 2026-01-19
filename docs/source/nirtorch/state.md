(nirtorch_state)=
# State management in NIRTorch

Managing state is crucial when dealing with (real) neurons because we have to maintain a state, such as a membrane potential, leak, or otherwise.
There are two main ways of doing that: implicitly and explicitly.
NIRTorch uses **explicit** state handling.
Below, we briefly explain the difference and show how the state handling works in practice when executing NIRTorch graphs.

## Implicit vs Explicit State: A High-Level Comparison

This high-level comparison shows the fundamental differences between implicit and explicit state handling approaches. Let's break down each aspect:

1. **Control of State**: 
   - Implicit: The framework manages state changes
   - Explicit: The developer directly controls state transitions

2. **Visibility**:
   - Implicit: State changes can happen automatically
   - Explicit: State changes must be explicitly coded

3. **Traceability**:
   - Implicit: State transitions may be hidden
   - Explicit: Clear state transition flow

### Dataflow Patterns

To understand how these patterns work in practice, let's examine their dataflow characteristics:

```{mermaid}
flowchart LR
    subgraph Implicit["Module with Internal State"]
        direction TB
        I_Input[/"Input Data"/]
        I_State[("Internal State")]
        I_Process["Process"]
        I_Output[/"Output"/]
        
        I_Input --> I_Process
        I_Process <--> I_State
        I_Process --> I_Output
        
        style I_State fill:#ff9999
    end
    
    subgraph Explicit["Module with External State"]
        direction TB
        E_Input[/"Input Data"/]
        E_State[("Current State")]
        E_Process["Process"]
        E_Output[/"Output + New State"/]
        
        E_Input --> E_Process
        E_State --> E_Process
        E_Process --> E_Output
        
        style E_State fill:#99ff99
    end
    
    ImplicitNote["State lives inside
    module and is mutated
    during processing.
    The user never sees
    the state, but may have
    to reset it"]
    
    ExplicitNote["State flows through
    module as input/output,
    never mutated internally"]
    
    Implicit -.-> ImplicitNote
    Explicit -.-> ExplicitNote
    
    style ImplicitNote fill:#fff,stroke:#999
    style ExplicitNote fill:#fff,stroke:#999
```

## Advantages and Trade-offs

### Implicit State

Advantages:
- Less boilerplate code
- Can feel more intuitive for simple applications
- Automatic state synchronization

Disadvantages:
- Harder to test
- State changes can be difficult to track
- Can lead to unexpected side effects

### Explicit State

Advantages:
- Predictable data flow
- Easier to test
- Clear state transitions
- Better debugging experience

Disadvantages:
- More verbose
- Can feel overengineered for simple cases

## State handling in Python and NIRTorch

NIRTorch uses **explicit** state management, which may be more cumbersome to write but makes data flow more visible:

```python
class MyState:
    voltage: float

def stateful_function(data, state):
    # 1. Calculate a new voltage
    new_voltage = ... 
    # 2. Calculate the function output
    output = ... 
    # 3. Define a new state
    new_state = MyState(voltage=new_voltage)
    # 4. A tuple of (data, state) is returned
    #    Note that the new state returned and the original remains unchanged
    return output, new_state 
```

Once NIRTorch has parsed a NIR module into Torch modules (read more about that in the page about [To PyTorch: Interpreting NIR](#nirtorch_interpreting)),
the resulting module expects a second `state` parameter, like the function above.
Similarly, it will return a tuple of `(data, state)`.
Here is a full example where we first initialize a Torch module from NIRTorch, and then applies it several times with the correct state

```python
import nir
import nirtorch
import numpy as np

nir_weight = np.ones((2, 2))
nir_graph = nir.NIRGraph.from_list(nir.Linear(weight=nir_weight))

torch_module = nirtorch.nir_to_torch(
    nir_graph=nir_graph, 
    node_map={} # We can leave this empty since we only 
                # use a linear layer which has a default mapping
)

##
## This is where the state handling happens
##
# Assume some time-series data with 100 entries each with two datapoints
time_data = torch.random(100, 2)
state = None
results = []
# Loop through the time-series data, one entry at the time
for single_data in time_data:
    if state is None:
        # If state is None, we leave the state blank
        output, state = torch_module(single_data)
    else:
        # If state is not None, we need to feed it back in
        output, state = torch_module(single_data, state)
    results.append(output)

# results is now a tensor of shape (100, 2)
results = torch.stack(results)
```