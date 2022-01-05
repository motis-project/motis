import React, { useState } from 'react';

import Maybe, { nothing } from 'true-myth/maybe';

import { Search } from './Search';
import { SubOverlay } from './SubOverlay';
import { Connection } from './ConnectionTypes';

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

    // Boolean used to decide if the Overlay is being displayed
    const [overlayHidden, setOverlayHidden] = useState<Boolean>(false);

    // Boolean used to decide if the SubOverlay is being displayed
    const [subOverlayHidden, setSubOverlayHidden] = useState<Boolean>(true);

    // Connections
    const [connections, setConnections] = useState<Connection[]>([]);

    return(
        <div className={overlayHidden ? 'overlay-container' : 'overlay-container hidden' }>
            <div className='overlay'>
                <div id='overlay-content'>
                    <Search setConnections={setConnections}/>
                    <div id='connections'>
                        <div className='no-results'>
                            <div className='schedule-range'>Auskunft von 19.10.2020 bis 21.10.2020 möglich</div>
                        </div>
                    </div>
                </div>
                <SubOverlay subOverlayHidden={subOverlayHidden} setSubOverlayHidden={setSubOverlayHidden}/>
            </div>
            <div className='overlay-tabs'>
                <div className='overlay-toggle' onClick={() => setOverlayHidden(!overlayHidden)}>
                    <i className='icon'>arrow_drop_down</i>
                </div>
                <div className={subOverlayHidden ? 'trip-search-toggle' : 'trip-search-toggle enabled'} onClick={() => setSubOverlayHidden(!subOverlayHidden)}>
                    <i className='icon'>train</i>
                </div>
            </div>
        </div>
    );
};