import React, { useState } from 'react';

import Maybe, { nothing } from 'true-myth/maybe';

import { Search } from './Search';
import { SubOverlay } from './SubOverlay';
import { Connection, TransportInfo, WalkInfo } from './ConnectionTypes';
import { Translations } from './Localization';
import { ConnectionRender } from './ConnectionRender';

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

interface SubView {
    //TODO: Das muss ein maybe mit TripDetailsView, StationEventsView und TripSearchView sein
    TripSearchView: any
}

const displayTime = (posixTime) => {
    let today = new Date(posixTime * 1000);
    let h = String(today.getHours());
    let m = String(today.getMinutes()).padStart(2, '0');
    return h + ':' + m;
}

const displayDuration = (posixTime) => {
    let dur = String(posixTime + ' min');
    return dur;
}

export const Overlay: React.FC<{'translation': Translations}> = (props) => {

    // Boolean used to decide if the Overlay is being displayed
    const [overlayHidden, setOverlayHidden] = useState<Boolean>(true);

    // Boolean used to decide if the SubOverlay is being displayed
    const [subOverlayHidden, setSubOverlayHidden] = useState<Boolean>(true);

    // Connections
    const [connections, setConnections] = useState<Connection[]>(null);

    return (
        <div className={overlayHidden ? 'overlay-container' : 'overlay-container hidden'}>
            <div className='overlay'>
                <div id='overlay-content'>
                    <Search setConnections={setConnections} translation={props.translation}/>
                    {!connections ? 
                    <div className='spinner'>
                        <div className='bounce1'></div>
                        <div className='bounce2'></div>
                        <div className='bounce3'></div>
                    </div> : 
                    <div id='connections'>
                        {connections.map((connectionElem: Connection, index) => (
                            <div className='connection' key={index}>
                                <div className='pure-g'>
                                    <div className='pure-u-4-24 connection-times'>
                                        <div className='connection-departure'>
                                            {displayTime(connectionElem.trips[0].id.time)}
                                        </div>
                                        <div className='connection-arrivial'>
                                            {displayTime(connectionElem.trips[0].id.target_time)}
                                        </div>
                                    </div>
                                    <div className='pure-u-4-24 connection-duration'>
                                        {displayDuration((connectionElem.trips[0].id.target_time - connectionElem.trips[0].id.time) / 60)}
                                    </div>
                                    <div className='pure-u-16-24 connection-trains'>
                                        <div className='transport-graph'>
                                            <ConnectionRender transports={connectionElem.transports}/>
                                            <div className='tooltip' style={{ position: 'absolute', left: '0px', top: '23px' }}>
                                                <div className='stations'>
                                                    <div className='departure'><span className='station'>Frankfurt (Main) Hauptbahnhof</span><span
                                                        className='time'>14:20</span></div>
                                                    <div className='arrival'><span className='station'>Darmstadt Hauptbahnhof</span><span
                                                        className='time'>14:35</span></div>
                                                </div>
                                                <div className='transport-name'><span>IC 117</span></div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>}
                </div>
                <SubOverlay subOverlayHidden={subOverlayHidden} setSubOverlayHidden={setSubOverlayHidden} translation={props.translation}/>
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