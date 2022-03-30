import React, { useState } from 'react';

import moment from 'moment';

import { Search } from './Search';
import { SubOverlay } from './SubOverlay';
import { getStationCoords } from './TripView';
import { Spinner } from './LoadingSpinner';
import { ConnectionRender } from './ConnectionRender';
import { duration } from './Journey';
import { Translations } from '../App/Localization';
import { getFromLocalStorage } from '../App/LocalStorage';
import { Connection, Station, Transport, TripId, WalkInfo } from '../Types/Connection';
import { Address } from '../Types/SuggestionTypes';
import { Interval } from '../Types/RoutingTypes';
import { TripView } from './TripView';
import { SubOverlayEvent } from '../Types/EventHistory';


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

export const Overlay: React.FC<{ 'translation': Translations, 'scheduleInfo': Interval, 'searchDate': moment.Moment, 'mapData': any, 'subOverlayContent': SubOverlayEvent[], 'setSubOverlayContent': React.Dispatch<React.SetStateAction<SubOverlayEvent[]>>}> = (props) => {

    // Boolean used to decide if the Overlay is being displayed
    const [overlayHidden, setOverlayHidden] = useState<boolean>(true);

    // searchDate manages the currently used Time for IntermodalRoutingRequests
    const [searchDate, setSearchDate] = useState<moment.Moment>(null);

    // Connections
    const [connections, setConnections] = useState<Connection[]>(null);

    // Boolean used to signal <Search> that extendForward was clicked
    const [extendForwardFlag, setExtendForwardFlag] = useState<boolean>(false);

    // Boolean used to signal <Search> that extendBackward was clicked
    const [extendBackwardFlag, setExtendBackwardFlag] = useState<boolean>(false);

    // True: Display connections as List. False: Show detailed Information for one Connection
    const [tripViewHidden, setTripViewHidden] = useState<boolean>(true);

    // stores the index of selected connection to be rendered in TripView
    const [indexOfConnection, setIndexOfConnection] = useState<number>(0);
    // stores the tripID of selected Train via trainbox
    const [trainSelected, setTrainSelected] = useState<TripId>(undefined);
    // is true if connection line in map is being hovered
    const [connectionHighlighted, setConnectionHighlighted] = useState<boolean>(false);

    const [start, setStart] = useState<Station | Address>(getFromLocalStorage("motis.routing.from_location"));

    const [destination, setDestination] = useState<Station | Address>(getFromLocalStorage("motis.routing.to_location"));

    const [mapFilter, setMapFilter] = useState<any>(null);
    // stores all connection Ids being highlighted by the segtion hovered in map
    const [selectedConnectionIds, setSelectedConnectionIds] = useState<number[]>([]);
  
    // If true, renders the Loading animation for the connectionList
    const [loading, setLoading] = useState<boolean>(false);

    const [subOverlayToggle, setSubOverlayToggle] = useState<boolean>(false);

    //when clicking on train in the map
    React.useEffect(() => {
        window.portEvents.sub('showTripDetails', function(data){
            props.setSubOverlayContent([...props.subOverlayContent, {id: 'tripView', train: data}])
            setTrainSelected(data);
        });
    });

    React.useEffect(() =>{
        window.portEvents.sub('showStationDetails', function(data){
            setMapFilter(null);
            window.portEvents.pub('mapSetDetailFilter', null);
        });
    });    

    React.useEffect(() => {
        if(tripViewHidden){
            setMapFilter(null);
            window.portEvents.pub('mapSetDetailFilter', null);
        }
    }, [tripViewHidden])

    // On initial render searchDate will be null, waiting for the ScheduleInfoResponse. This useEffect should fire only once.
    React.useEffect(() => {
        setSearchDate(props.searchDate);
    }, [props.searchDate]);

    // if connection line on map is being hovered, collect all ids of connections to be highlighted
    React.useEffect(() => {
        let connectionIds = [];
        if(props.mapData !== undefined && props.mapData.hoveredTripSegments !== null){
            props.mapData.hoveredTripSegments.map((elem: any) => {
                connectionIds.push(elem.connectionIds[0] + 1);
            });
            setSelectedConnectionIds(connectionIds);
            setConnectionHighlighted(true);
        }else if(props.mapData !== undefined && props.mapData.hoveredWalkSegment !== null){
            connectionIds.push(props.mapData.hoveredWalkSegment.connectionIds + 1);
            setSelectedConnectionIds(connectionIds);
            setConnectionHighlighted(true);
        }else{
            setConnectionHighlighted(false);
        }
    }, [props.mapData]);

    return (
        <div className={overlayHidden ? 'overlay-container' : 'overlay-container hidden'}>
            <div className='overlay'>
                <div id='overlay-content'>
                    {tripViewHidden ?
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
                            {props.scheduleInfo ? // As long as the scheduleInfo Fetch hasnt returned a schedule Info, we display nothing
                                loading ? // If any action needs a loading animation, display Spinner
                                    <Spinner />
                                    :
                                    connections ? // If connections is not null anymore, display connections
                                        connections.length !== 0 ?  //Only display connections if any are presesnt
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
                                                        <div className={ `connection ${connectionElem.new} ${(connectionHighlighted) ? `${(selectedConnectionIds.includes(index)) ? 'highlighted' : 'faded'}` : ''}`}
                                                            key={index}
                                                            onClick={() => { setTripViewHidden(false);
                                                                             setIndexOfConnection(index);
                                                                             setMapFilter(getMapFilter(connectionElem));
                                                                             window.portEvents.pub('mapSetDetailFilter', getMapFilter(connectionElem));
                                                                             window.portEvents.pub('mapFitBounds', getStationCoords(connectionElem));}}
                                                            onMouseEnter={() => { let ids = []; ids.push(connectionElem.id); window.portEvents.pub('mapHighlightConnections', ids)}}
                                                            onMouseLeave={() => { window.portEvents.pub('mapHighlightConnections', [])}}>
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
                                                                        <ConnectionRender   translation={props.translation}
                                                                                            connection={connectionElem}
                                                                                            connectionHighlighted={connectionHighlighted}
                                                                                            mapData={props.mapData}
                                                                                            parentIndex={index}/>
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
                        <TripView   translation={props.translation}
                                    trainSelected={connections[indexOfConnection]}
                                    setTrainSelected={setTrainSelected}
                                    setTripViewHidden={setTripViewHidden}
                                    mapFilter={props.mapData}
                                    subOverlayContent={props.subOverlayContent} 
                                    setSubOverlayContent={props.setSubOverlayContent}/>
                    }
                </div>
                <SubOverlay translation={props.translation}
                            scheduleInfo={props.scheduleInfo}
                            searchDate={props.searchDate}
                            trainSelected={trainSelected}
                            mapFilter={mapFilter}
                            setTrainSelected={setTrainSelected}
                            subOverlayContent={props.subOverlayContent}
                            setSubOverlayContent={props.setSubOverlayContent}
                            setSubOverlayToggle={setSubOverlayToggle}
                            />
            </div>
            <div className='overlay-tabs'>
                <div className='overlay-toggle' onClick={() => setOverlayHidden(!overlayHidden)}>
                    <i className='icon'>arrow_drop_down</i>
                </div>
                <div    className={ `trip-search-toggle ${subOverlayToggle ? 'enabled': ''}`} 
                        onClick={() => {
                            if (subOverlayToggle) {
                                props.setSubOverlayContent([]);
                                setSubOverlayToggle(false);
                            }else {
                                props.setSubOverlayContent([{id: 'tripSearch'}]);
                                setSubOverlayToggle(true);
                            }}}>
                    <i className='icon'>train</i>
                </div>
            </div>
        </div>
    );
};
