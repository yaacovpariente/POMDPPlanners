---
name: 🧩 Type Hint Issue
about: Report missing, incorrect, or unclear type hints
title: "[TYPE] <Short description of the type hint issue>"
labels: type-hint
assignees: ''

---

## 📝 Description

Describe the problem with the type hint(s). Is it missing, incorrect, too loose, or too strict?  
Include the relevant function/class/module name(s).

## 🧩 Location

- Module / file:
- Function or class (if applicable):

## 💡 Suggested Fix (Optional)

If you have a suggestion for how the type hint should look, provide it here.

```python
# Example
def example_function(x: int) -> str:  # Current
# Suggested
def example_function(x: int | None) -> str:
