module Data.ScheduleInfo.Request exposing (request)

import Json.Encode as Encode


request : Encode.Value
request =
    Encode.object
        [ ("destination"
            , Encode.object
                [ ("type" , Encode.string "Module")
                , ("target" , Encode.string "/lookup/schedule_info")
                ])
        , ("content_type" , Encode.string "MotisNoMessage")
        , ("content"
            , Encode.object
                [])
        ]
