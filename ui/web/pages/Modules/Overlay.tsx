import React, { useState } from 'react';

import Maybe, { nothing } from 'true-myth/maybe';

import { Search } from './Search';
import { SubOverlay } from './SubOverlay';
import { Connection } from './ConnectionTypes';
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

export const Overlay: React.FC = (props) => {

    // Boolean used to decide if the Overlay is being displayed
    const [overlayHidden, setOverlayHidden] = useState<Boolean>(false);

    // Boolean used to decide if the SubOverlay is being displayed
    const [subOverlayHidden, setSubOverlayHidden] = useState<Boolean>(true);

    // Connections
    const [connections, setConnections] = useState<Connection[]>([]);

    return (
        <div className={overlayHidden ? 'overlay-container' : 'overlay-container hidden'}>
            <div className='overlay'>
                <div id='overlay-content'>
                    <Search setConnections={setConnections} />
                    <div id='connections'>
                        {connections.map((connectionElem: Connection) => (
                            <div className='connection'>
                                <div className='pure-g'>
                                    <div className='pure-u-4-24 connection-times'>
                                        <div className='connection-departure'>
                                            {connectionElem.trips[0].id.time}
                                        </div>
                                        <div className='connection-arrivial'>
                                            {connectionElem.trips[0].id.target_time}
                                        </div>
                                    </div>
                                    <div className='pure-u-4-24 connection-duration'>
                                        {(connectionElem.trips[0].id.target_time - connectionElem.trips[0].id.time) / 60}
                                    </div>
                                    <div className='pure-u-16-24 connection-trains'>
                                        <div className='transport-graph'>
                                            <svg width='335' height='40' viewBox='0 0 335 40'>
                                                <g>
                                                    <g className='part train-className-2 acc-0'>
                                                        <line x1='0' y1='12' x2='326' y2='12' className='train-line'></line>
                                                        <circle cx='12' cy='12' r='12' className='train-circle'></circle>
                                                        <use xlinkHref='#train' className='train-icon' x='4' y='4' width='16' height='16'></use><text
                                                            x='0' y='40' textAnchor='start' className='train-name'>IC 117</text>
                                                        <rect x='0' y='0' width='323' height='24' className='tooltipTrigger'></rect>
                                                    </g>
                                                </g>
                                                <g className='destination'>
                                                    <circle cx='329' cy='12' r='6'></circle>
                                                </g>
                                            </svg>
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
                    </div>
                </div>
                <SubOverlay subOverlayHidden={subOverlayHidden} setSubOverlayHidden={setSubOverlayHidden} />
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