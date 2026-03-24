# Recipe for Adding a New Cosmetic Feature

## Steps

1. **Implement the feature**
   - Add to the appropriate module (shapes in render/borders/, edges in render/edges/, etc.)
   - Wire into the main render pipeline
   - Add any new style fields to NodeStyle/EdgeStyle/ClusterStyle

2. **Add to feature tuning gallery**
   - In the gallery generation script, add feature cards:
     a. Default appearance (the feature with no other options)
     b. Parameter variations (every knob the feature exposes)
     c. Combinations with other features (borders, fills, shadows, labels)
     d. Adversarial cases (long text, tiny nodes, extreme values)

3. **Run critic evaluation**
   - Generate gallery images at multiple sizes (4in, 8in, 16in)
   - Score each card: 1-10 on correctness, aesthetics, edge cases
   - Iterate on rendering until minimum score >= 8, mean >= 9

4. **Add to theme system**
   - If the feature is a shape: add to any relevant themes
   - If the feature is a style field: set sensible defaults

5. **Add tests**
   - Unit test: feature produces valid output
   - Regression test: known input -> expected output
   - Combination test: feature + other features don't conflict
