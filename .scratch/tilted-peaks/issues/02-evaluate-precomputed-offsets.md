# Evaluate profiles on precomputed offsets

Type: task
Status: ready-for-agent
Blocked by: 01

## Objective

Give the lineshape module a small interface that accepts Hz offsets. Keep
coordinate geometry out of the six profile families.

## Work

- Add offset-based evaluation to the `Shape` interface and the four shared
  singlet and doublet templates.
- Return the offset derivative needed for the multidimensional chain rule.
- Keep the current axis-aligned evaluation as a compatibility path during the
  refactor, then remove any duplicate evaluation logic.
- Run equivalence and derivative tests for all twelve registered lineshapes.

## Done when

Every existing profile can evaluate the same point grid from precomputed Hz
offsets, and zero-tilt values match the current implementation.

## Comments
