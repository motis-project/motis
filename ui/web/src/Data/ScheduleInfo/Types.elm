module Data.ScheduleInfo.Types exposing (ScheduleInfo)

import Date exposing (Date)


type alias ScheduleInfo =
    { name : String
    , begin : Date
    , end : Date
    }
