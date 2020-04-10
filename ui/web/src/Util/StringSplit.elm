module Util.StringSplit exposing (intNthToken, nthToken)

import String


nthToken : Int -> String -> String -> Maybe String
nthToken pos splitToken str =
    String.split splitToken str
        |> List.drop pos
        |> List.head


intNthToken : Int -> String -> String -> Maybe Int
intNthToken pos splitToken str =
    case nthToken pos splitToken str of
        Just str ->
            String.toInt str |> Result.toMaybe

        Nothing ->
            Nothing
