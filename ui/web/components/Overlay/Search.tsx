import React, { useEffect, useState } from 'react';

import moment from 'moment';

import { DatePicker } from './DatePicker';
import { Mode, IntermodalRoutingResponse } from '../Types/IntermodalRoutingTypes';
import { Connection, Position, Station, WalkInfo } from '../Types/ConnectionTypes';
import { Translations } from '../App/Localization';
import { Address } from '../Types/SuggestionTypes';
import { SearchInputField } from './SearchInputField';
import { Modepicker } from './ModePicker';
import { getFromLocalStorage, setLocalStorage } from '../App/LocalStorage';
import { Interval } from '../Types/RoutingTypes';
import { markerSearch } from '../Map/RailvizContextMenu';


const getRoutingOptions = (startType: string, startModes: Mode[], start: Station | Address, searchType: string, searchDirection: string, destinationType: string, destinationModes: Mode[], destination: Station | Address, interval: Interval ) => {
    return {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({  destination: {type: "Module", target: "/intermodal"}, 
                                content_type: 'IntermodalRoutingRequest',
                                content: {start_type: startType, start_modes: startModes, start: { station: start, min_connection_count: 5, interval: interval, extend_interval_later: true, extend_interval_earlier: true }, search_type: searchType, search_dir: searchDirection, destination_type: destinationType, destination_modes: destinationModes, destination: destination } })
    };
};

const mapConnections = (connections: Connection[]) => {
    let cons = [];
    if(connections){
        for(let i = 0; i < connections.length; i++){
            let stations = [];
            for(let k = 0; k < connections[i].stops.length; k++){
                stations.push(connections[i].stops[k].station);
            }
            let trains = [];
            for(let k = 0; k < connections[i].trips.length; k++){
                let trip = connections[i].trips[k].id;
                let sections = [];
                for(let l = connections[i].trips[k].range.from; l < connections[i].trips[k].range.to; l++){
                    sections.push({ 'arrivalStation': connections[i].stops[l+1].station,
                                    'departureStation': connections[i].stops[l].station,
                                    'scheduledArrivalTime': connections[i].stops[l+1].arrival.schedule_time,
                                    'scheduledDepartureTime': connections[i].stops[l].departure.schedule_time});
                }
                trains.push({'sections': sections, 'trip': trip});
            }
            let walks = [];
            for(let k = 0; k < connections[i].transports.length; k++){
                if(connections[i].transports[k].move_type === 'Walk'){
                    let walk = connections[i].transports[k].move as WalkInfo;
                    walks.push({'arrivalStation': connections[i].stops[walk.range.to].station,
                                'departureStation': connections[i].stops[walk.range.from].station,
                                'accessibility': walk.accessibility,
                                'mumoType': walk.mumo_type})
                }
            }
            cons.push({'id': i, 'stations': stations, 'trains': trains, 'walks': walks});
        }
    }
    return cons;
};

// This Dummy Object will be used to Identify Day Changes in the Connections List which will be swaped against Dividers
const dummyConnection = (dummyDate: string) => {
    return {
        stops: [],
        transports: [],
        trips: [],
        problems: [],
        dummyDay: dummyDate
    };
}


const sendConnectionsToOverlay = (setConnections: React.Dispatch<React.SetStateAction<Connection[]>>, connections: Connection[], setAllConnectionsWithoutDummies: React.Dispatch<React.SetStateAction<Connection[]>>) => {
    let dummyIndexes = [0];
    let connectionsWithDummies = [...connections];
    let previousConnectionDay = moment.unix(connections[0].stops[0].departure.schedule_time);
    let dummyDays = [previousConnectionDay.format('D.M.YYYY')];
    setAllConnectionsWithoutDummies(connections);

    for (let i = 1; i < connections.length; i++){
        if (moment.unix(connections[i].stops[0].departure.schedule_time).day() != previousConnectionDay.day()){
            dummyIndexes.push(i);
            dummyDays.push(moment.unix(connections[i].stops[0].departure.schedule_time).format('D.M.YYYY'));
            previousConnectionDay.add(1, 'day');
        }
    };
    dummyIndexes.map((val, idx) => {
        connectionsWithDummies.splice(val + idx, 0, dummyConnection(dummyDays[idx]));
    })
    setConnections(connectionsWithDummies);
};


const handleErrors = (response) => {
    if (!response.ok) {
        throw Error(response.statusText);
    }
    return response;
}


export const Search: React.FC<{'setConnections': React.Dispatch<React.SetStateAction<Connection[]>>, 'translation': Translations, 'extendForwardFlag': boolean, 'extendBackwardFlag': boolean, 'displayDate': moment.Moment, 'setDisplayDate': React.Dispatch<React.SetStateAction<moment.Moment>>}> = (props) => {
 
    // Start
    // StartType
    const [startType, setStartType] = useState<string>('PretripStart');
    
    // StartModes
    const [startModes, setStartModes] = useState<Mode[]>([]);

    // Start Station or Position
    const [start, setStart] = useState<Station | Address>(getFromLocalStorage("motis.routing.from_location"));
    
    
    // Destination
    // DestinationType
    const [destinationType, setDestinationType] = useState<string>('InputStation');
    
    // Destination_modes tracks the ModePicker for the Destination
    const [destinationModes, setDestinationModes] = useState<Mode[]>([]);
    
    // Destination holds the Value of 'to location' input field
    const [destination, setDestination] = useState<Station | Address>(getFromLocalStorage("motis.routing.to_location"));
    

    // Current Date
    const [searchDate, setSearchDate] = useState<moment.Moment>(props.displayDate);
    
    // SearchTime
    const [searchTime, setSearchTime] = useState<string>(props.displayDate.format('HH:mm'));
    
    // SearchType
    const [searchType, setSearchType] = useState<string>('Accessibility');
    
    // SearchDirection
    const [searchDirection, setSearchDirection] = useState<string>('Forward');

    // SearchInterval
    const [searchInterval, setSearchInterval] = useState<Interval>({begin: searchDate.unix(), end: searchDate.unix() + 7200});

    // Save currently displayed List of Connections. Will be extended with every fetch.
    const [allConnectionsWithoutDummies, setAllConnectionsWithoutDummies] = useState<Connection[]>([]);

    // Boolean used to determine if allConnectionsWithoutDummies needs to be expanded at the head or at the tail
    const [extendBackward, setExtendBackward] = useState<boolean>(true);


    // This Effect is one of 2 IntermodalConnectionRequest API Calls.
    // If this one is triggered, then we want to discard the currently shown Connections and load a new list
    useEffect(() => {
        if (start !== null && destination !== null) {
            props.setConnections(null); // Only when connections=null will the Loading animation be shown
            let requestURL = 'https://europe.motis-project.de/?elm=IntermodalConnectionRequest';
            //console.log('Fire searchQuery')

            fetch(requestURL, getRoutingOptions(startType, startModes, start, searchType, searchDirection, destinationType, destinationModes, destination, searchInterval))
                .then(handleErrors)
                .then(res => res.json())
                .then((res: IntermodalRoutingResponse) => {
                    console.log("Response came in");
                    console.log(res);
                    //props.setConnections(res.content.connections);
                    sendConnectionsToOverlay(props.setConnections, res.content.connections, setAllConnectionsWithoutDummies);
                    window.portEvents.pub('mapSetMarkers', {'startPosition': getFromLocalStorage("motis.routing.from_location").pos,
                                                            'startName': getFromLocalStorage("motis.routing.from_location").name,
                                                            'destinationPosition': getFromLocalStorage("motis.routing.to_location").pos,
                                                            'destinationName': getFromLocalStorage("motis.routing.to_location").name});
                    
                    window.portEvents.pub('mapSetConnections', {'mapId': 'map', 'connections': mapConnections(res.content.connections), 'lowestId': 0});
                })
                .catch(error => {});
        }
    }, [start, startModes, destination, destinationModes, searchDirection]);


    // This Effect is one of 2 IntermodalConnectionRequest API Calls.
    // If this one is triggered, then we want to keep the currently shown Connections and add the newly fetched ones to this list
    useEffect(() => {
        if (start !== null && destination !== null) {
            props.setConnections(null); // Only when connections=null will the Loading animation be shown
            let requestURL = 'https://europe.motis-project.de/?elm=IntermodalConnectionRequest';
            //console.log('Fire searchQuery')

            fetch(requestURL, getRoutingOptions(startType, startModes, start, searchType, searchDirection, destinationType, destinationModes, destination, searchInterval))
                .then(handleErrors)
                .then(res => res.json())
                .then((res: IntermodalRoutingResponse) => {
                    console.log("Response came in");
                    console.log(res);
                    //props.setConnections(res.content.connections);
                    if(extendBackward){
                        sendConnectionsToOverlay(props.setConnections, [...res.content.connections, ...allConnectionsWithoutDummies], setAllConnectionsWithoutDummies);
                    } else {
                        sendConnectionsToOverlay(props.setConnections, [...allConnectionsWithoutDummies, ...res.content.connections], setAllConnectionsWithoutDummies);
                    }
                    window.portEvents.pub('mapSetMarkers', {'startPosition': getFromLocalStorage("motis.routing.from_location").pos,
                                                            'startName': getFromLocalStorage("motis.routing.from_location").name,
                                                            'destinationPosition': getFromLocalStorage("motis.routing.to_location").pos,
                                                            'destinationName': getFromLocalStorage("motis.routing.to_location").name});
                    window.portEvents.pub('mapSetConnections', {'mapId': 'map', 'connections': mapConnections(res.content.connections), 'lowestId': 0});
                })
                .catch(error => {});
        }
    }, [searchInterval]);


    // On searchDate change, discard currently displayed Connections and compute new Interval for the IntermodalConnectionRequest
    useEffect(() => {
        setAllConnectionsWithoutDummies([]);
        setSearchInterval({begin: searchDate.unix(), end: searchDate.unix() + 3600 * 2});
        props.setDisplayDate(searchDate);
    }, [searchDate]);


    // Handle Interval change after extend-search-interval search-backward Button in Overlay was clicked
    useEffect(() => {
        setSearchInterval({begin: searchInterval.begin - 3600 * 4, end: searchInterval.end - 3600 * 4});
        setExtendBackward(true);
    }, [props.extendBackwardFlag])


    // Handle Interval change after extend-search-interval search-forwad Button in Overlay was clicked
    useEffect(() => {
        setSearchInterval({begin: searchInterval.begin + 3600 * 4, end: searchInterval.end + 3600 * 4});
        setExtendBackward(false);
    }, [props.extendForwardFlag])


    useEffect(() => {
        window.portEvents.sub('mapSetMarkers', function(){
            if(markerSearch){
                if(markerSearch[0]){
                    setStart(markerSearch[1]);
                }else{
                    setDestination(markerSearch[1]);
                }
            }
        });
    });


    return (
        <div id='search'>
            <div className='pure-g gutters'>
                <div className='pure-u-1 pure-u-sm-12-24 from-location'>
                    <div>
                        <SearchInputField   translation={props.translation}
                                        label={props.translation.search.start}
                                        station={start}
                                        setSearchDisplay={setStart}
                                        localStorageStation='motis.routing.from_location'/>
                        <Modepicker translation={props.translation} 
                                    title={props.translation.search.startTransports} 
                                    setModes={setStartModes}
                                    localStorageModes='motis.routing.from_modes'/>
                    </div>
                    <div className='swap-locations-btn'>
                        <label className='gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select'>
                            <input  type='checkbox' 
                                    onClick={() => {
                                        let swapStation = destination;
                                        setDestination(start);
                                        setStart(swapStation);
                            }}/>
                            <i className='icon'>swap_vert</i>
                        </label>
                    </div>
                </div>
                <div className='pure-u-1 pure-u-sm-12-24'>
                    <DatePicker translation={props.translation}
                                currentDate={searchDate}
                                setCurrentDate={setSearchDate}/>
                </div>
            </div>
            <div className='pure-g gutters'>
                <div className='pure-u-1 pure-u-sm-12-24 to-location'>
                    <div>
                        <SearchInputField   translation={props.translation}
                                            label={props.translation.search.destination}
                                            station={destination}
                                            setSearchDisplay={setDestination}
                                            localStorageStation='motis.routing.to_location'/>
                        <Modepicker translation={props.translation} 
                                    title={props.translation.search.destinationTransports} 
                                    setModes={setDestinationModes}
                                    localStorageModes='motis.routing.to_modes'/>
                    </div>
                </div>
                <div className='pure-u-1 pure-u-sm-9-24'>
                    <div>
                        <div className='label'>{props.translation.search.time}</div>
                        <div className='gb-input-group'>
                            <div className='gb-input-icon'><i className='icon'>schedule</i></div>
                            <input
                                className='gb-input' 
                                tabIndex={4} 
                                value={searchTime}
                                onChange={(e) => {
                                    setSearchTime(e.currentTarget.value);
                                    /* Wie sollen wir mit fehlerhfatem Input umgehen?
                                    if (e.currentTarget.value.split(':').length == 2) {
                                        let [hour, minute] = e.currentTarget.value.split(':');
                                        if (!isNaN(+hour) && !isNaN(+minute)){
                                            setSearchHours(moment(searchHours.hour(hour as unknown as number > 23 ? 23 : hour as unknown as number)));
                                            setSearchHours(moment(searchHours.minute(minute as unknown as number > 59 ? 59 : hour as unknown as number)));
                                }}*/}}/>
                            <div className='gb-input-widget'>
                                <div className='hour-buttons'>
                                    <div><a
                                            className='gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select' 
                                            onClick={() => {
                                                let newSearchDate = searchDate.clone().subtract(1, 'h')
                                                setSearchDate(newSearchDate); 
                                                setSearchTime(newSearchDate.format('HH:mm'))}}>
                                            <i className='icon'>chevron_left</i></a></div>
                                    <div><a
                                            className='gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select' 
                                            onClick={() => {
                                                let newSearchDate = searchDate.clone().add(1, 'h')
                                                setSearchDate(newSearchDate);
                                                setSearchTime(newSearchDate.format('HH:mm'))}}>
                                            <i className='icon'>chevron_right</i></a></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div className='pure-u-1 pure-u-sm-3-24 time-option'>
                    <form>
                        <input  type='radio' 
                                id='search-forward' 
                                name='time-option' 
                                value='Forward'
                                checked={searchDirection === 'Forward'} 
                                onChange={() => setSearchDirection('Forward')}/>
                        <label htmlFor='search-forward'>{props.translation.search.departure}</label>
                    </form>
                    <form>
                        <input  type='radio' 
                                id='search-backward' 
                                name='time-option' 
                                value='Backward' 
                                checked={searchDirection === 'Backward'} 
                                onChange={() => setSearchDirection('Backward')}/>
                        <label htmlFor='search-backward'>{props.translation.search.arrival}</label>
                    </form>
                </div>
            </div>
        </div>
    )
};