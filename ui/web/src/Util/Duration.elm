module Util.Duration exposing (Duration, diff)

import Time exposing (Posix, posixToMillis)

-- Milliseconds
type alias Duration = Int

diff : Posix -> Posix -> Duration
diff a b:
   (posixToMillis b) - (posixToMillis a)
