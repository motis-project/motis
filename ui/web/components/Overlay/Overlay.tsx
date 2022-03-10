import React, { useState } from 'react';

import moment from 'moment';

import { Search } from './Search';
import { SubOverlay } from './SubOverlay';
import { Connection, Station, Transport, TripId } from '../Types/ConnectionTypes';
import { Translations } from '../App/Localization';
import { ConnectionRender, JourneyRender } from './ConnectionRender';
import { getFromLocalStorage } from '../App/LocalStorage';
import { Address } from '../Types/SuggestionTypes';
import { Interval } from '../Types/RoutingTypes';
import { elmAPIResponse } from '../Types/IntermodalRoutingTypes';
import { ScheduleInfoResponse } from '../Types/ScheduleInfo';


const displayTime = (posixTime) => {
    let today = new Date(posixTime * 1000);
    let h = String(today.getHours());
    let m = String(today.getMinutes()).padStart(2, '0');
    return h + ':' + m;
}

const displayDuration = (posixTime) => {
    let today = new Date(posixTime * 1000);
    let h = String(today.getUTCHours());
    let m = String(today.getUTCMinutes()).padStart(2, '0');
    if (h === '0') {
        return m + 'min';
    } else {
        return h + 'h ' + m + 'min';
    }
}


export const Overlay: React.FC<{ 'translation': Translations}> = (props) => {

    // Hold the available Interval for Scheduling Information
    const [scheduleInfo, setScheduleInfo] = useState<Interval>(null);

    // Hold the currently displayed Date
    const [displayDate, setDisplayDate] = useState<moment.Moment>(null);
    
    // Boolean used to decide if the Overlay is being displayed
    const [overlayHidden, setOverlayHidden] = useState<Boolean>(true);

    // Boolean used to decide if the SubOverlay is being displayed
    const [subOverlayHidden, setSubOverlayHidden] = useState<Boolean>(true);

    // Connections
    const [connections, setConnections] = useState<Connection[]>(null);

    // Boolean used to signal <Search> that extendForward was clicked
    const [extendForwardFlag, setExtendForwardFlag] = useState<boolean>(true);

    // Boolean used to signal <Search> that extendBackward was clicked
    const [extendBackwardFlag, setExtendBackwardFlag] = useState<boolean>(true);
    
    const [detailViewHidden, setDetailViewHidden] = useState<Boolean>(true);

    const [indexOfConnection, setIndexOfConnection] = useState<number>(0);

    const [trainSelected, setTrainSelected] = useState<TripId>(undefined);
    
    const [start, setStart] = useState<Station | Address>(getFromLocalStorage("motis.routing.from_location"));

    const [destination, setDestination] = useState<Station | Address>(getFromLocalStorage("motis.routing.to_location"));

    React.useEffect(() => {
        let requestURL = 'https://europe.motis-project.de/?elm=requestScheduleInfo';

        fetch(requestURL, { method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({content: {}, content_type: 'MotisNoMessage', destination: { target: '/lookup/schedule_info', type: 'Module' }})})
        .then(res => res.json())
        .then((res: elmAPIResponse) => {
            console.log("Response came in");
            console.log(res);
            let intv = {begin: (res.content as ScheduleInfoResponse).begin, end: (res.content as ScheduleInfoResponse).end}
            let intvBegin = moment.unix(intv.begin);
            intvBegin.hour(moment().hour())
            intvBegin.minute(moment().minute())
            setDisplayDate(intvBegin);
            setScheduleInfo(intv);
        })
    }, []);

    return (
        <div className={overlayHidden ? 'overlay-container' : 'overlay-container hidden'}>
            <div className='overlay'>
                <div id='overlay-content'>
                    {detailViewHidden ?
                        <>
                            <Search setConnections={setConnections} 
                                    translation={props.translation} 
                                    extendForwardFlag={extendForwardFlag}
                                    extendBackwardFlag={extendBackwardFlag}
                                    displayDate={displayDate}
                                    setDisplayDate={setDisplayDate}/>
                            {!connections ?
                                scheduleInfo && (displayDate.unix() < scheduleInfo.begin || displayDate.unix() > scheduleInfo.end) ?
                                    <div id='connections'>
                                        <div className="main-error">
                                            <div className="">{props.translation.errors.journeyDateNotInSchedule}</div>
                                            <div className="schedule-range">{props.translation.connections.scheduleRange(scheduleInfo.begin, scheduleInfo.end - 3600 * 24)}</div>
                                        </div>
                                    </div>
                                    :
                                    <div className='spinner'>
                                        <div className='bounce1'></div>
                                        <div className='bounce2'></div>
                                        <div className='bounce3'></div>
                                    </div> 
                                : 
                                <div id='connections'>
                                <div className='connections'>
                                    <div className='extend-search-interval search-before' onClick={() => setExtendBackwardFlag(!extendBackwardFlag)}><a>{props.translation.connections.extendBefore}</a></div>
                                    <div className='connection-list'>
                                        {connections.map((connectionElem: Connection, index) => (
                                            connectionElem.dummyDay ?
                                            <div className='date-header divider' key={index}><span>{connectionElem.dummyDay}</span></div>
                                            :
                                            <div className='connection' key={index} onClick={() => { setDetailViewHidden(false); setIndexOfConnection(index) }}>
                                                <div className='pure-g'>
                                                    <div className='pure-u-4-24 connection-times'>
                                                        <div className='connection-departure'>
                                                            {displayTime(connectionElem.stops[0].departure.time)}
                                                        </div>
                                                        <div className='connection-arrival'>
                                                            {displayTime(connectionElem.stops[connectionElem.stops.length - 1].arrival.time)}
                                                        </div>
                                                    </div>
                                                    <div className='pure-u-4-24 connection-duration'>
                                                        {displayDuration(new Date(connectionElem.stops[connectionElem.stops.length - 1].arrival.time).getTime() - new Date(connectionElem.stops[0].departure.time).getTime())}
                                                    </div>
                                                    <div className='pure-u-16-24 connection-trains'>
                                                        <div className='transport-graph'>
                                                            <ConnectionRender connection={connectionElem} setDetailViewHidden={setDetailViewHidden}/>
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
                                        <div className='divider footer'></div>
                                        <div className='extend-search-interval search-after' onClick={() => setExtendForwardFlag(!extendForwardFlag)}><a>{props.translation.connections.extendAfter}</a></div>
                                    </div>
                                </div>
                            </div>
                            }
                        </> :
                        <div className="connection-details">
                            <div className="connection-info">
                                <div className="header">
                                    <div className="back"><i className="icon" onClick={() => setDetailViewHidden(true)}>arrow_back</i></div>
                                    <div className="details">
                                        <div className="date">24.1.2022</div>
                                        <div className="connection-times">
                                            <div className="times">
                                                <div className="connection-departure">{displayTime(connections[indexOfConnection].stops[0].departure.time)}</div>
                                                <div className="connection-arrival">{displayTime(connections[indexOfConnection].stops[connections[indexOfConnection].stops.length - 1].arrival.time)}</div>
                                            </div>
                                            <div className="locations">
                                                <div>{start.name}</div>
                                                <div>{destination.name}</div>
                                            </div>
                                        </div>
                                        <div className="summary">
                                            <span className="duration">
                                                <i className="icon">schedule</i>
                                                {displayDuration(new Date(connections[indexOfConnection].stops[connections[indexOfConnection].stops.length - 1].arrival.time).getTime() - new Date(connections[indexOfConnection].stops[0].departure.time).getTime())}
                                            </span>
                                            <span className="interchanges">
                                                <i className="icon">transfer_within_a_station</i>
                                                {connections[indexOfConnection].trips.length - 1 + ' Umstiege'}
                                            </span>
                                        </div>
                                    </div>
                                    <div className="actions"><i className="icon">save</i><i className="icon">share</i></div>
                                </div>
                            </div>
                            <div className="connection-journey" id="connection-journey">
                                <JourneyRender connection={connections[indexOfConnection]} setSubOverlayHidden={setSubOverlayHidden} setTrainSelected={setTrainSelected} detailViewHidden={detailViewHidden}/>
                            </div>
                        </div>
                    }
                </div>
                <SubOverlay subOverlayHidden={subOverlayHidden} setSubOverlayHidden={setSubOverlayHidden} trainSelected={trainSelected} setTrainSelected={setTrainSelected} translation={props.translation} detailViewHidden={detailViewHidden}/>
            </div>
            <div className='overlay-tabs'>
                <div className='overlay-toggle' onClick={() => setOverlayHidden(!overlayHidden)}>
                    <i className='icon'>arrow_drop_down</i>
                </div>
                <div className={subOverlayHidden ? 'trip-search-toggle' : 'trip-search-toggle enabled'} onClick={() => {setSubOverlayHidden(!subOverlayHidden), setTrainSelected(undefined)}}>
                    <i className='icon'>train</i>
                </div>
            </div>
        </div>
    );
};