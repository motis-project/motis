sketch:

    1 --walk-- 2 --walk-- 3 --S6-- 4 --S6-- 5 --walk-- 6 --walk-- 8
                                            |           |         |
                                           S9          S7        S8
                                            |           |         |
                                           12          11         9

services:
  - S6: 3,4,5   [18:10,19:00,19:10,20:00]
  - S7: 6,11    [20:00,21:00]
  - S8: 8,9     [21:10,22:00]
  - S9: 5,12    [20:10,21:00]

foot edges:
  1->2 [10]
  2->3 [10]
  5->6 [10]
  6->8 [10]

test cases:
  A: cancel first enter
  B: cancel last leave
  C: leave+enter = interchange
     - walk(s) / same station
     - delay / cancel

A:     3,4,5     | cancel 3
   1,2,3,4,5     | cancel 3
B:     3,4,5     | cancel 5
       3,4,5,6,8 | cancel 5
C: 2,3,4,5,11    | cancel 5 (arrival S6)
   2,3,4,5,11    | cancel 5 (departure S7)
   2,3,4,5,6,8,9 | cancel 5 (arrival S6)
   2,3,4,5,6,8,9 | cancel 8 (departure S8)