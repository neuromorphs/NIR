# Supported Primitives in NIR
This document lists which primitives are supported by the software frameworks for conversion to and from NIR:
- `→`: Supported for conversion from NIR
- `←`: Supported for conversion to NIR
- `⟷`: Supported for both conversion directions (to and from NIR)
Please note that this list is generated automatically and may not be entirely accurate.
<br />


| Primitive | hxtorch | jaxsnn | Lava | Nengo | Norse | rockpool | sinabs | snntorch | SpiNNaker2 | Spyx |
|-----------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Conv1d |  |  |  |  |  |  | ⟷ |  | → | → |
| Conv2d |  |  | → |  | ⟷ |  | ⟷ | ⟷ | → | ⟷ |
| Delay |  |  |  |  |  |  |  |  |  | → |
| Flatten |  |  | → |  | → |  | ⟷ | ⟷ | → | → |
| Affine |  | → | → | ⟷ | ⟷ | ⟷ | ⟷ | ⟷ | → | ⟷ |
| Linear | ⟷ | → | → |  | ← | ⟷ |  | ⟷ | → | ⟷ |
| Scale |  |  |  |  |  |  |  |  |  | → |
| CubaLI | ⟷ |  |  |  |  |  |  |  |  |  |
| CubaLIF | ⟷ | → | → |  | ⟷ | ⟷ |  | ⟷ | → | ⟷ |
| I |  |  |  |  |  |  |  |  |  | → |
| IF |  |  | → |  | ⟷ |  | ⟷ | → | → | ⟷ |
| LI |  |  |  | ⟷ | ⟷ | ⟷ | ⟷ |  |  | ← |
| LIF |  |  | → | ⟷ | ⟷ | ⟷ | ⟷ | ⟷ | → | ⟷ |
| AvgPool2d |  |  |  |  |  |  |  | ⟷ |  |  |
| SumPool2d |  |  | → |  | → |  | ⟷ |  | → | → |
| Threshold |  |  |  |  |  |  |  |  |  | → |