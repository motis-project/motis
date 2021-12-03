import React from 'react';
import Maybe, { nothing } from 'true-myth/maybe';

import Localization from '../../src/Localization/Base';
import MyDate from '../../src/Util/Date';
import ConnectionDetails from '../../src/Widgets/ConnectionDetails';
import Routing from '../../src/Widgets/Routing';
import TripSearch from '../../src/Widgets/TripSearch';


interface Model {
    routing : Routing.Model,
    //railViz : RailViz.Model,
    connectionDetails : Maybe<ConnectionDetails.State>,
    tripDetails : Maybe<ConnectionDetails.State>,
    //stationEvents : Maybe StationEvents.Model,
    tripSearch : TripSearch.Model,
    subView : Maybe<SubView>,
    selectedConnectionIdx : Maybe<number>,
    //scheduleInfo : Maybe ScheduleInfo,
    locale : Localization,
    apiEndpoint : String,
    currentTime : Date,
    timeOffset : number,
    overlayVisible : Boolean,
    //stationSearch : Typeahead.Model,
    //programFlags : ProgramFlags,
    //simTimePicker : SimTimePicker.Model,
    updateSearchTime : Boolean
}

interface SubView{
    //TODO: Das muss ein maybe mit TripDetailsView, StationEventsView und TripSearchView sein
    TripSearchView : any
}

const connectionDetailsView = (locale : Localization, currentTime : Date, state : ConnectionDetails) : any/*TODO: List(Html Msg)*/ => {
    return true; //TODO
}

const getCurrentTime = (model : Model) : Date => {
    return new Date()//TODO: MyDate.toTime(model.currentTime + model.timeOffset)
}

const getCurrentDate = (model : Model) : MyDate => {
    return MyDate.fromTime(getCurrentTime(model))
}

export const OverlayView: React.FC<Model> = (props) => {
    let mainOverlayContent = props.connectionDetails.match({
        Just: ({ c }) => connectionDetailsView(props.locale, getCurrentDate(props), c),
        Nothing: () => [Routing.view, props.locale, props.routing].map(() => true/*TODO: Html.map RoutingUpdate*/)
    });
    let subOverlayContent = props.subView.match({
        //TODO: 2 Just hinzufügen
        Just: ({ TripSearchView }) => TripSearchView(props.locale, props.tripSearch),
        Nothing: () => nothing
    });
    //let subOverlay = ...

    return(
        <div className='overlay-container'>{/*hidden>*/}
            <div className='overlay'>
                <div id='overlay-content'>
                    {/*mainOverlayContent, subOverlay*/}
                </div>
            </div>
            <div className='overlay-tabs'>
                <div className='overlay-toggle'>{/**onClick */ }
                    <i className='icon'>arrow_drop_down</i>
                </div>
                <div className='trip-search-toggle'>{/**enabled, onClick*/}
                    <i className='icon'>train</i>
                </div>
            </div>
        </div>
    );
};