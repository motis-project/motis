module Widgets.Helpers.ApiErrorUtil exposing (errorText, motisErrorMsg)

import Localization.Base exposing (..)
import Util.Api as Api exposing (..)


errorText : Localization -> ApiError -> String
errorText locale err =
    case err of
        MotisError err_ ->
            motisErrorMsg locale err_

        TimeoutError ->
            locale.t.errors.timeout

        NetworkError ->
            locale.t.errors.network

        HttpError status ->
            locale.t.errors.http status

        DecodeError msg ->
            locale.t.errors.decode msg


motisErrorMsg : Localization -> MotisErrorInfo -> String
motisErrorMsg { t } err =
    case err of
        RoutingError JourneyDateNotInSchedule ->
            t.errors.journeyDateNotInSchedule

        AccessError AccessTimestampNotInSchedule ->
            t.errors.journeyDateNotInSchedule

        ModuleError TargetNotFound ->
            t.errors.moduleNotFound

        OsrmError OsrmProfileNotAvailable ->
            t.errors.osrmProfileNotAvailable

        OsrmError OsrmNoRoutingResponse ->
            t.errors.osrmNoRoutingResponse

        TripBasedError TripBasedJourneyDateNotInSchedule ->
            t.errors.journeyDateNotInSchedule

        PprError PprProfileNotAvailable ->
            t.errors.pprProfileNotAvailable

        _ ->
            t.errors.internalError (toString err)
