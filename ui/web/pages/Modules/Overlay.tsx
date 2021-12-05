import React from 'react';

import Maybe, { nothing } from 'true-myth/maybe';

import { Search } from './Search';

//interface Model {
    //routing : Routing.Model,
    //railViz : RailViz.Model,
    //connectionDetails : Maybe<ConnectionDetails.State>,
    //tripDetails : Maybe<ConnectionDetails.State>,
    //stationEvents : Maybe StationEvents.Model,
    //tripSearch : TripSearch.Model,
    //subView : Maybe<SubView>,
    //selectedConnectionIdx : Maybe<number>,
    //scheduleInfo : Maybe ScheduleInfo,
    //locale : Localization,
    //apiEndpoint : String,
    //currentTime : Date,
    //timeOffset : number,
    //overlayVisible : Boolean,
    //stationSearch : Typeahead.Model,
    //programFlags : ProgramFlags,
    //simTimePicker : SimTimePicker.Model,
    //updateSearchTime : Boolean
//}

interface SubView{
    //TODO: Das muss ein maybe mit TripDetailsView, StationEventsView und TripSearchView sein
    TripSearchView : any
}

export const Overlay: React.FC = (props) => {

    return(
        <div className='overlay-container'>{/*hidden>*/}
            <div className='overlay'>
                <div id='overlay-content'>
                    <Search />
                    {/*<Connections />*/}
                </div>
                <div className="sub-overlay">
                    {/*<SubOverlay />*/}
                </div>
            </div>
            <div className='overlay-tabs'>
                <div className='overlay-toggle'>{/**onClick */ }
                    {/*<img className='icon'>arrow_drop_down</img>*/}
                </div>
                <div className='trip-search-toggle'>{/**enabled, onClick*/}
                    {/*<img className='icon'>train</img>*/}
                </div>
            </div>
        </div>
    );
};