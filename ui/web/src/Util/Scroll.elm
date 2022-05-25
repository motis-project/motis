module Util.Scroll exposing (toTop)

import Browser exposing (Dom)

-- ID of an DOM element
toTop: String -> Task Error
toTop id =
    Dom.setViewportOf id 0 0
