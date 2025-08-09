# Test Documentation Summary: test_tree.py

**Original File:** `/home/kobi/Documents/github/POMDPPlanners/POMDPPlanners/tests/test_core/test_tree.py`  
**Total Tests:** 14  
**Documented Tests:** 13  
**Documentation Coverage:** 92.9%

## Test Distribution by Type

- **Undocumented:** 1 tests
- **Unit:** 13 tests

---

## Test Documentation Details

### 1. `test_belief`

**Line:** 24

**Description:** Create weighted particle belief for tree node testing.

---

### 2. `test_env`

**Line:** 33

**Purpose:** Validates env

**Given:** Test setup conditions

**When:** Test operation is performed

**Then:** Expected behavior is verified

**Test Type:** unit

---

### 3. `test_action_node_initialization_creates_mcts_tree_node`

**Line:** 47

**Purpose:** Validates ActionNode initializes correctly for MCTS tree construction

**Given:** Action string "move_forward" and optional parent node with test data

**When:** ActionNode instances are created with basic and parent-child configurations

**Then:** Nodes are initialized with correct action, default values, and proper tree relationships

**Test Type:** unit

---

### 4. `test_belief_node_initialization`

**Line:** 86

**Purpose:** Validates proper initialization of belief node

**Given:** Constructor parameters and initial conditions

**When:** Object is initialized

**Then:** Object is properly constructed with expected attributes

**Test Type:** unit

---

### 5. `test_tree_structure`

**Line:** 115

**Purpose:** Validates tree structure

**Given:** Test setup conditions

**When:** Test operation is performed

**Then:** Expected behavior is verified

**Test Type:** unit

---

### 6. `test_get_optimal_action`

**Line:** 141

**Purpose:** Validates get optimal action

**Given:** Test setup conditions

**When:** Test operation is performed

**Then:** Expected behavior is verified

**Test Type:** unit

---

### 7. `test_node_properties`

**Line:** 169

**Purpose:** Validates node properties

**Given:** Test setup conditions

**When:** Test operation is performed

**Then:** Expected behavior is verified

**Test Type:** unit

---

### 8. `test_sample_child_node`

**Line:** 208

**Purpose:** Validates sampling behavior for  child node

**Given:** Configured object with sampling capabilities

**When:** Sample method is called

**Then:** Valid samples are returned according to distribution

**Test Type:** unit

---

### 9. `test_sample_child_node_single_child`

**Line:** 265

**Purpose:** Validates sampling behavior for  child node single child

**Given:** Configured object with sampling capabilities

**When:** Sample method is called

**Then:** Valid samples are returned according to distribution

**Test Type:** unit

---

### 10. `test_sample_child_node_no_children`

**Line:** 285

**Purpose:** Validates sampling behavior for  child node no children

**Given:** Configured object with sampling capabilities

**When:** Sample method is called

**Then:** Valid samples are returned according to distribution

**Test Type:** unit

---

### 11. `test_get_belief_node_child`

**Line:** 303

**Purpose:** Validates get belief node child

**Given:** Test setup conditions

**When:** Test operation is performed

**Then:** Expected behavior is verified

**Test Type:** unit

---

### 12. `test_get_belief_node_child_no_children`

**Line:** 337

**Purpose:** Validates get belief node child no children

**Given:** Test setup conditions

**When:** Test operation is performed

**Then:** Expected behavior is verified

**Test Type:** unit

---

### 13. `test_get_belief_node_child_duplicate_observations`

**Line:** 355

**Purpose:** Validates get belief node child duplicate observations

**Given:** Test setup conditions

**When:** Test operation is performed

**Then:** Expected behavior is verified

**Test Type:** unit

---

### 14. `test_get_belief_node_child_none_observation`

**Line:** 377

**Purpose:** Validates get belief node child none observation

**Given:** Test setup conditions

**When:** Test operation is performed

**Then:** Expected behavior is verified

**Test Type:** unit

---

