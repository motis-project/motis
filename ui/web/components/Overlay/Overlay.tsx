import React, { useEffect, useState } from 'react';

import moment from 'moment';

import { Search } from './Search';
import { SubOverlay } from './SubOverlay';
import { Spinner } from './LoadingSpinner';
import { ConnectionRender } from './ConnectionRender';
import { JourneyRender, duration } from './Journey';
import { Translations } from '../App/Localization';
import { getFromLocalStorage } from '../App/LocalStorage';
import { Connection, Station, Transport, TransportInfo, TripId } from '../Types/Connection';
import { Address } from '../Types/SuggestionTypes';
import { Interval } from '../Types/RoutingTypes';


const getTransportCountString = (transports: Transport[], translation: Translations) => {
    let count = 0;
    for (let index = 0; index < transports.length; index++) {
        if (transports[index].move_type === 'Transport' && index > 0) {
            count++
        }
    }
    return translation.connections.interchanges(count);
}

export const Overlay: React.FC<{ 'translation': Translations, 'scheduleInfo': Interval, 'subOverlayHidden': boolean, 'setSubOverlayHidden': React.Dispatch<React.SetStateAction<boolean>>, 'stationEventTrigger': boolean, 'setStationEventTrigger': React.Dispatch<React.SetStateAction<boolean>>, 'station': (Station | Address), 'searchDate': moment.Moment}> = (props) => {

    // Boolean used to decide if the Overlay is being displayed
    const [overlayHidden, setOverlayHidden] = useState<Boolean>(true);

    // searchDate manages the currently used Time for IntermodalRoutingRequests
    const [searchDate, setSearchDate] = useState<moment.Moment>(null);

    // Connections
    const [connections, setConnections] = useState<Connection[]>(null);

    // Boolean used to signal <Search> that extendForward was clicked
    const [extendForwardFlag, setExtendForwardFlag] = useState<boolean>(false);

    // Boolean used to signal <Search> that extendBackward was clicked
    const [extendBackwardFlag, setExtendBackwardFlag] = useState<boolean>(false);

    const [detailViewHidden, setDetailViewHidden] = useState<Boolean>(true);

    const [indexOfConnection, setIndexOfConnection] = useState<number>(0);

    const [trainSelected, setTrainSelected] = useState<TripId>(undefined);

    const [start, setStart] = useState<Station | Address>(getFromLocalStorage("motis.routing.from_location"));

    const [destination, setDestination] = useState<Station | Address>(getFromLocalStorage("motis.routing.to_location"));

    // If true, renders the Loading animation for the connectionList
    const [loading, setLoading] = useState<boolean>(false);

    const [connectionHighlighted, setConnectionHighlighted] = useState<boolean>(false);

    const [connectionDoNothing, setConnectionDoNothing] = useState<boolean>(true);

    
    // On initial render searchDate will be null, waiting for the ScheduleInfoResponse. This useEffect should fire only once.
    useEffect(() => {
        setSearchDate(props.searchDate);
    }, [props.searchDate]);

    return (
        <div className={overlayHidden ? 'overlay-container' : 'overlay-container hidden'}>
            <div className='overlay'>
                <div id='overlay-content'>
                    {detailViewHidden ?
                        <>
                            <Search translation={props.translation} 
                                    scheduleInfo={props.scheduleInfo}
                                    start={start}
                                    destination={destination}
                                    extendForwardFlag={extendForwardFlag}
                                    extendBackwardFlag={extendBackwardFlag}
                                    searchDate={searchDate}
                                    setStart={setStart}
                                    setDestination={setDestination}
                                    setConnections={setConnections} 
                                    setExtendForwardFlag={setExtendForwardFlag}
                                    setExtendBackwardFlag={setExtendBackwardFlag}
                                    setSearchDate={setSearchDate}
                                    setLoading={setLoading}/>
                            {props.scheduleInfo ?
                                loading ?
                                    <Spinner />
                                    :
                                    connections ?
                                        connections.length !== 0 ? 
                                            <div id='connections'>
                                                <div className='connections'>
                                                <div className='extend-search-interval search-before' onClick={() => setExtendBackwardFlag(true)}>
                                                    {extendBackwardFlag ?
                                                        <Spinner />
                                                        :
                                                        <a>{props.translation.connections.extendBefore}</a>
                                                    }
                                                </div>
                                                <div className='connection-list'>
                                                    {connections.map((connectionElem: Connection, index) => (
                                                        connectionElem.dummyDay ?
                                                        <div className='date-header divider' key={index}><span>{connectionElem.dummyDay}</span></div>
                                                        :
                                                        <div  className={(connectionDoNothing) ? `connection ${connectionElem.new}` : `connection ${connectionElem.new} ${(connectionHighlighted) ? 'highlighted' : 'faded'}`}
                                                            key={index}
                                                            onClick={() => { setDetailViewHidden(false); setIndexOfConnection(index) }}
                                                            onMouseEnter={() => { let ids = []; ids.push(connectionElem.id); window.portEvents.pub('mapHighlightConnections', ids); setConnectionHighlighted(true)}}
                                                            onMouseLeave={() => { window.portEvents.pub('mapHighlightConnections', []); setConnectionHighlighted(false)}}>
                                                            <div className='pure-g'>
                                                                <div className='pure-u-4-24 connection-times'>
                                                                    <div className='connection-departure'>
                                                                        {moment.unix(connectionElem.stops[0].departure.time).format('HH:mm')}
                                                                    </div>
                                                                    <div className='connection-arrival'>
                                                                        {moment.unix(connectionElem.stops[connectionElem.stops.length - 1].arrival.time).format('HH:mm')}
                                                                    </div>
                                                                </div>
                                                                <div className='pure-u-4-24 connection-duration'>
                                                                    {duration(connectionElem.stops[0].departure.time, connectionElem.stops[connectionElem.stops.length - 1].arrival.time)}
                                                                </div>
                                                                <div className='pure-u-16-24 connection-trains'>
                                                                    <div className={(connectionHighlighted ? 'transport-graph highlighting' : 'transport-graph')}>
                                                                        <ConnectionRender   connection={connectionElem}
                                                                                            setDetailViewHidden={setDetailViewHidden}
                                                                                            setConnectionHighlighted={setConnectionHighlighted}
                                                                                            connectionDoNothing={connectionDoNothing}
                                                                                            connectionHighlighted={connectionHighlighted}/>
                                                                    </div>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    ))}
                                                    <div className='divider footer'></div>
                                                    <div className='extend-search-interval search-after' onClick={() => setExtendForwardFlag(true)}>
                                                        {extendForwardFlag ?
                                                            <Spinner />
                                                            :
                                                            <a>{props.translation.connections.extendAfter}</a>
                                                        }
                                                    </div>
                                                </div>
                                                </div>
                                            </div>
                                            :
                                            <div id='connections'>
                                                {props.searchDate && (props.searchDate.unix() < props.scheduleInfo.begin || props.searchDate.unix() > props.scheduleInfo.end) ?
                                                    <div className="main-error">
                                                        <div className="">{props.translation.errors.journeyDateNotInSchedule}</div>
                                                        <div className="schedule-range">{props.translation.connections.scheduleRange(props.scheduleInfo.begin, props.scheduleInfo.end - 3600 * 24)}</div>
                                                    </div>
                                                    :
                                                    <div className='no-results'>
                                                        <div>{props.translation.connections.noResults}</div>
                                                        <div className="schedule-range">{props.translation.connections.scheduleRange(props.scheduleInfo.begin, props.scheduleInfo.end - 3600 * 24)}</div>
                                                    </div>
                                                }
                                            </div>
                                        :
                                        <div id='connections'>
                                            <div className='no-results'>
                                                <div className="schedule-range">{props.translation.connections.scheduleRange(props.scheduleInfo.begin, props.scheduleInfo.end - 3600 * 24)}</div>
                                            </div>
                                        </div>
                                :
                                <div className='no-results'>
                                    {''}
                                </div>
                            }
                        </> 
                        :
                        <div className="connection-details">
                            <div className="connection-info">
                                <div className="header">
                                    <div className="back" onClick={() => setDetailViewHidden(true)}><i className="icon">arrow_back</i></div>
                                    <div className="details">
                                        <div className="date">{props.searchDate.format(props.translation.dateFormat)}</div>
                                        <div className="connection-times">
                                            <div className="times">
                                                <div className="connection-departure">{moment.unix(connections[indexOfConnection].stops[0].departure.time).format('HH:mm')}</div>
                                                <div className="connection-arrival">{moment.unix(connections[indexOfConnection].stops[connections[indexOfConnection].stops.length - 1].arrival.time).format('HH:mm')}</div>
                                            </div>
                                            <div className="locations">
                                                <div>{start.name}</div>
                                                <div>{destination.name}</div>
                                            </div>
                                        </div>
                                        <div className="summary">
                                            <span className="duration">
                                                <i className="icon">schedule</i>
                                                {duration(connections[indexOfConnection].stops[0].departure.time, connections[indexOfConnection].stops[connections[indexOfConnection].stops.length - 1].arrival.time)}
                                            </span>
                                            <span className="interchanges">
                                                <i className="icon">transfer_within_a_station</i>
                                                {getTransportCountString(connections[indexOfConnection].transports, props.translation)}
                                            </span>
                                        </div>
                                    </div>
                                    <div className="actions"><i className="icon">save</i><i className="icon">share</i></div>
                                </div>
                            </div>
                            <div className="connection-journey" id="connection-journey">
                                <JourneyRender connection={connections[indexOfConnection]} setSubOverlayHidden={props.setSubOverlayHidden} setTrainSelected={setTrainSelected} detailViewHidden={detailViewHidden} translation={props.translation} />
                            </div>
                        </div>
                    }
                </div>
                <SubOverlay translation={props.translation} 
                            scheduleInfo={props.scheduleInfo}
                            searchDate={props.searchDate}
                            station={props.station}
                            stationEventTrigger={props.stationEventTrigger}
                            subOverlayHidden={props.subOverlayHidden} 
                            trainSelected={trainSelected} 
                            detailViewHidden={detailViewHidden}
                            setTrainSelected={setTrainSelected} 
                            setStationEventTrigger={props.setStationEventTrigger}
                            setSubOverlayHidden={props.setSubOverlayHidden} 
                            />
            </div>
            <div className='overlay-tabs'>
                <div className='overlay-toggle' onClick={() => setOverlayHidden(!overlayHidden)}>
                    <i className='icon'>arrow_drop_down</i>
                </div>
                <div className={props.subOverlayHidden ? 'trip-search-toggle' : 'trip-search-toggle enabled'} onClick={() => { props.setSubOverlayHidden(!props.subOverlayHidden), setTrainSelected(undefined) }}>
                    <i className='icon'>train</i>
                </div>
            </div>
        </div>
    );
};
