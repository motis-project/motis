import React, { useEffect, useState } from 'react';

import moment from 'moment';

import { Search } from './Search';
import { SubOverlay } from './SubOverlay';
import { Spinner } from './LoadingSpinner';
import { ConnectionRender } from './ConnectionRender';
import { JourneyRender, duration } from './Journey';
import { Translations } from '../App/Localization';
import { getFromLocalStorage } from '../App/LocalStorage';
import { Connection, Station, Transport, TransportInfo, TripId, WalkInfo } from '../Types/Connection';
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

export const getMapFilter = (connection: Connection) => {
    let filter;
    let trains = [];
    for(let k = 0; k < connection.trips.length; k++){
        let trip = connection.trips[k].id;
        let sections = [];
        for(let l = connection.trips[k].range.from; l < connection.trips[k].range.to; l++){
            sections.push({ 'arrivalStation': connection.stops[l+1].station,
                            'departureStation': connection.stops[l].station,
                            'scheduledArrivalTime': connection.stops[l+1].arrival.schedule_time,
                            'scheduledDepartureTime': connection.stops[l].departure.schedule_time});
        }
        trains.push({'sections': sections, 'trip': trip});
    }
    let walks = [];
    for(let k = 0; k < connection.transports.length; k++){
        if(connection.transports[k].move_type === 'Walk'){
            let walk = connection.transports[k].move as WalkInfo;
            walks.push({'arrivalStation': connection.stops[walk.range.to].station,
                        'departureStation': connection.stops[walk.range.from].station,
                        'accessibility': walk.accessibility,
                        'mumoType': walk.mumo_type})
        }
    }
    let interchanges = [];
    for(let i = 0; i < trains.length; i++){
        interchanges.push(trains[i].sections[0].departureStation);
        interchanges.push(trains[i].sections[trains[i].sections.length-1].arrivalStation);
    }
    filter = {'interchangeStations': interchanges, 'trains': trains, 'walks': walks};
    return filter
}

export const Overlay: React.FC<{ 'translation': Translations, 'scheduleInfo': Interval, 'subOverlayHidden': boolean, 'setSubOverlayHidden': React.Dispatch<React.SetStateAction<boolean>>, 'stationEventTrigger': boolean, 'setStationEventTrigger': React.Dispatch<React.SetStateAction<boolean>>, 'station': (Station | Address), 'setStation': React.Dispatch<React.SetStateAction<(Station | Address)>>, 'searchDate': moment.Moment, 'setSearchDate': React.Dispatch<React.SetStateAction<moment.Moment>>}> = (props) => {

    // Hold the currently displayed Date
    const [displayDate, setDisplayDate] = useState<moment.Moment>(null);
    
    // Boolean used to decide if the Overlay is being displayed
    const [overlayHidden, setOverlayHidden] = useState<Boolean>(true);

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

    const [mapFilter, setMapFilter] = useState<any>(null);

    //when clicking on train in the map
    React.useEffect(() => {
        window.portEvents.sub('showTripDetails', function(data){
            setTrainSelected(data);
            props.setSubOverlayHidden(false);
        });
    });

    React.useEffect(() =>{
        window.portEvents.sub('showStationDetails', function(data){
            setMapFilter(null); 
            window.portEvents.pub('mapSetDetailFilter', null);
            props.setStation({id: data, name: ''});
        })
    })    

    React.useEffect(() => {
        if(detailViewHidden){
            setMapFilter(null);
            window.portEvents.pub('mapSetDetailFilter', null);
        }
    }, [detailViewHidden]);

    React.useEffect(() => {
        if (props.scheduleInfo !== null) {
            let currentTime = moment();
            let adjustedDisplayDate = moment.unix(props.scheduleInfo.begin);
            adjustedDisplayDate.hour(currentTime.hour());
            adjustedDisplayDate.minute(currentTime.minute());
            setDisplayDate(adjustedDisplayDate);
        }
    }, [props.scheduleInfo]);

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
                                    displayDate={displayDate}
                                    extendForwardFlag={extendForwardFlag}
                                    extendBackwardFlag={extendBackwardFlag}
                                    setStart={setStart}
                                    setDestination={setDestination}
                                    setConnections={setConnections} 
                                    setDisplayDate={setDisplayDate}
                                    setExtendForwardFlag={setExtendForwardFlag}
                                    setExtendBackwardFlag={setExtendBackwardFlag}
                                    searchDate={props.searchDate}
                                    setSearchDate={props.setSearchDate}/>
                            {!connections ?
                                props.scheduleInfo && displayDate && (displayDate.unix() < props.scheduleInfo.begin || displayDate.unix() > props.scheduleInfo.end) ?
                                    <div id='connections'>
                                        <div className="main-error">
                                            <div className="">{props.translation.errors.journeyDateNotInSchedule}</div>
                                            <div className="schedule-range">{props.translation.connections.scheduleRange(props.scheduleInfo.begin, props.scheduleInfo.end - 3600 * 24)}</div>
                                        </div>
                                    </div>
                                    :
                                    <Spinner />
                                : 
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
                                                <div className='connection' key={index} onClick={() => { setDetailViewHidden(false);
                                                                                                         setIndexOfConnection(index);
                                                                                                         setMapFilter(getMapFilter(connectionElem));
                                                                                                         window.portEvents.pub('mapSetDetailFilter', getMapFilter(connectionElem)); 
                                                                                                        }}
                                                                                        onMouseEnter={() => {   let ids = [];
                                                                                                                ids.push(index-1);
                                                                                                                window.portEvents.pub('mapHighlightConnections', ids);
                                                                                                            }}
                                                                                        onMouseLeave={() => { window.portEvents.pub('mapHighlightConnections', []); }}>
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
                                                            <div className='transport-graph'>
                                                                <ConnectionRender connection={connectionElem} setDetailViewHidden={setDetailViewHidden} />
                                                                <div className='tooltip' style={{ position: 'absolute', left: '0px', top: '23px' }}>
                                                                    <div className='stations'>
                                                                        <div className='departure'><span className='station'>{connectionElem.stops[(connectionElem.transports[0].move as TransportInfo).range.from].station.name}</span><span
                                                                            className='time'>{moment.unix(connectionElem.stops[(connectionElem.transports[0].move as TransportInfo).range.from].departure.time).format('HH:mm')}</span></div>
                                                                        <div className='arrival'><span className='station'>{connectionElem.stops[(connectionElem.transports[0].move as TransportInfo).range.to].station.name}</span><span
                                                                            className='time'>{moment.unix(connectionElem.stops[(connectionElem.transports[0].move as TransportInfo).range.to].arrival.time).format('HH:mm')}</span></div>
                                                                    </div>
                                                                    <div className='transport-name'><span>{(connectionElem.transports[0].move as TransportInfo).name}</span></div>
                                                            </div>
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
                            }
                        </> :
                        <div className="connection-details">
                            <div className="connection-info">
                                <div className="header">
                                    <div className="back"><i className="icon" onClick={() => setDetailViewHidden(true)}>arrow_back</i></div>
                                    <div className="details">
                                        <div className="date">{displayDate.format(props.translation.dateFormat)}</div>
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
                                <JourneyRender connection={connections[indexOfConnection]} setSubOverlayHidden={props.setSubOverlayHidden} setTrainSelected={setTrainSelected} detailViewHidden={detailViewHidden} translation={props.translation}/>
                            </div>
                        </div>
                    }
                </div>
                <SubOverlay subOverlayHidden={props.subOverlayHidden} 
                            setSubOverlayHidden={props.setSubOverlayHidden} 
                            trainSelected={trainSelected} 
                            setTrainSelected={setTrainSelected} 
                            translation={props.translation} 
                            detailViewHidden={detailViewHidden} 
                            scheduleInfo={props.scheduleInfo}
                            displayDate={displayDate}
                            stationEventTrigger={props.stationEventTrigger}
                            setStationEventTrigger={props.setStationEventTrigger}
                            station={props.station}
                            searchDate={props.searchDate}
                            mapFilter={mapFilter}/>
            </div>
            <div className='overlay-tabs'>
                <div className='overlay-toggle' onClick={() => setOverlayHidden(!overlayHidden)}>
                    <i className='icon'>arrow_drop_down</i>
                </div>
                <div className={props.subOverlayHidden ? 'trip-search-toggle' : 'trip-search-toggle enabled'} onClick={() => {props.setSubOverlayHidden(!props.subOverlayHidden), setTrainSelected(undefined)}}>
                    <i className='icon'>train</i>
                </div>
            </div>
        </div>
    );
};