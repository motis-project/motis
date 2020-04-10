# Description

This schedule contains
- rule services that have no matching counterpart ("standalone")
- rule services that are operating without their counterpart

1 | 011 (bits 1: DE) | 5 -> 2
2 | 100 (bits 2: E6) |      6 -> 3 -> 4 -> 7
3 | 111 (bits 3: FE) | 1 -> 2 -> 3 -> 4

# Rules

THROUGH | (1, 3) | 2   | 010 (bits 4: D6) -> 010
MSS     | (1, 2) | 3-4 | 011 (bits 1: DE) -> 000

# Rest

1 | 001
2 | 100 -> standalone!
3 | 101