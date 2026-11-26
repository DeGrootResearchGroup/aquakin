# Models

A **model** is the set of species, parameters and reactions that `aquakin`
integrates. Every reactor and plant runs one, so choose or write the model
first.

| If you want to… | Read |
|---|---|
| use a published model as it is: ozonation, UV/H₂O₂, ASM1–ASM3, ADM1, WATS, precipitation | [Model catalogue](model_catalog.md) |
| write a model of your own, step by step | [Your first model](adding_models.md) |
| look up a key, a function or a rule of the model file | [YAML model format](model_format.md) |

A shipped model can also be the starting point for your own: a model file that
`extends:` one changes only what differs
([Inheriting a model](model_format.md#inheriting-a-model-extends)).

```{toctree}
:maxdepth: 2

model_catalog
adding_models
model_format
```
