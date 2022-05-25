module Util.DateUtil exposing (atNoon, combineDateTime, isSameDay, noon, toDate, unixTime)

import Date exposing (Date)

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


atNoon : Posix.Time -> Posix.Time
atNoon t =
    let
        millis = Time.posixToMillis t
        millisPerDay = 24 * 60 * 60 * 1000
        millisAfterMidnight = modBy millisPerDay millis
        halfDay = millisPerDay / 2
    in
        Time.millisToPosix (millis - millisAfterMidnight) + halfDay


toDate : Int -> Int -> Int -> Date
toDate year month day =
    Date.fromCalendarDate year (Date.numberToMonth month) day


isSameDay : Date -> Date -> Bool
isSameDay a b =
    let
        firstDay =
            ( Date.day a, Date.month a, Date.year a )

        secondDay =
            ( Date.day b, Date.month b, Date.year b )
    in
    firstDay == secondDay
