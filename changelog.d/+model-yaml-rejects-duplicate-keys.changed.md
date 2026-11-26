A model YAML file that repeats a key in a mapping, such as a species listed
twice in one reaction's stoichiometry, is now refused at load with an error
naming the key and both lines. YAML loaders otherwise keep the last value
without a word, so a mistyped or pasted entry went unnoticed. The shipped
`asm2d` listed `XMeOH` and `XMeP` twice (with equal values, so its results do
not change).
