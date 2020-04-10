module Util.Date exposing (atNoon, combineDateTime, isSameDay, noon, toDate, unixTime)

import Date exposing (Date)
import Date.Extra.Core exposing (intToMonth)
import Date.Extra.Create exposing (dateFromFields, timeFromFields)


unixTime : Date -> Int
unixTime d =
    floor (Date.toTime d) // 1000


combineDateTime : Date.Date -> Date.Date -> Date.Date
combineDateTime date time =
    dateFromFields (Date.year date)
        (Date.month date)
        (Date.day date)
        (Date.hour time)
        (Date.minute time)
        (Date.second time)
        (Date.millisecond time)


noon : Date.Date
noon =
    timeFromFields 12 0 0 0


atNoon : Date.Date -> Date.Date
atNoon date =
    combineDateTime date noon


toDate : Int -> Int -> Int -> Date
toDate year month day =
    dateFromFields year (intToMonth month) day 0 0 0 0


isSameDay : Date -> Date -> Bool
isSameDay a b =
    let
        firstDay =
            ( Date.day a, Date.month a, Date.year a )

        secondDay =
            ( Date.day b, Date.month b, Date.year b )
    in
    firstDay == secondDay
