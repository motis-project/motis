module Helpers
    exposing
        ( deferredCmd
        )

import Time exposing (Posix)
import Task exposing (Task)
import Process


deferredCmd : Float -> a -> Cmd a
deferredCmd delay a =
    Task.perform (always a) (Process.sleep delay)


