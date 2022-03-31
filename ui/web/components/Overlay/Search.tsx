import React, { useEffect, useState } from 'react';

import moment from 'moment';
import equal from 'deep-equal';

import { DatePicker } from './DatePicker';
import { Modepicker } from './ModePicker';
import { SearchInputField } from './SearchInputField';
import { Translations } from '../App/Localization';
import { getFromLocalStorage, ModeLocalStorage } from '../App/LocalStorage';
import useFetchPreventionOnDoubleClick from '../App/CancelablePromises';
import { useOutsideAlerter } from '../App/OutsideAlerter';
import { Interval } from '../Types/RoutingTypes';
import { Address } from '../Types/SuggestionTypes';
import { Connection, Station, WalkInfo } from '../Types/Connection';
import { Mode, IntermodalRoutingResponse } from '../Types/IntermodalRoutingTypes';
import { markerSearch } from '../Map/RailvizContextMenu';


interface SearchTypes {
    'translation': Translations, 
    'scheduleInfo': Interval,
    'start': Station | Address,
    'destination': Station | Address, 
    'connections': Connection[],
    'extendForwardFlag': boolean, 
    'extendBackwardFlag': boolean,
    'tripViewHidden': boolean,
    'searchDate': moment.Moment, 
    'setStart': React.Dispatch<React.SetStateAction<Station | Address>>, 
    'setDestination': React.Dispatch<React.SetStateAction<Station | Address>>, 
    'setConnections': React.Dispatch<React.SetStateAction<Connection[]>>, 
    'setExtendForwardFlag' : React.Dispatch<React.SetStateAction<boolean>>, 
    'setExtendBackwardFlag': React.Dispatch<React.SetStateAction<boolean>>,
    'setSearchDate': React.Dispatch<React.SetStateAction<moment.Moment>>,
    'setLoading': React.Dispatch<React.SetStateAction<boolean>>
}


// Helperfunction to differentiate objects that can be either a Station or an Address
const isStation = (f: Station | Address): f is Station => {
    return (f as Station).id !== undefined
}


// Get StartType for IntermodalRoutingRequests depending on start being a Station or an Address
const getStartType = (start: Station | Address) => {
    if (isStation(start)) {
        return 'PretripStart';
    }else {
        return 'IntermodalPretripStart';
    }
}

// Get DestinationType for IntermodalRoutingRequests depending on destination being a Station or an Address
const getDestinationType = (destination: Station | Address) => {
    if (isStation(destination)) {
        return 'InputStation';
    }else {
        return 'InputPosition';
    }
}


// return name and id for a Station and Position for an Address
const parseStationOrAddress = (s: Station | Address) => {
    if (isStation(s)) {
        return { name: s.name, id: s.id, };
    }else {
        return { position: s.pos };
    }
}


// Returns a properly formatted start Object for IntermodalRoutingRequests
const getStart = (start: Station | Address, min_connection_count: number, interval: Interval, extend_interval_earlier: boolean, extend_interval_later: boolean) => {
    if (isStation(start)) {
        return { station: parseStationOrAddress(start), min_connection_count: min_connection_count, interval: interval, extend_interval_later: extend_interval_later, extend_interval_earlier: extend_interval_earlier}
    } else {
        return { position: start.pos, min_connection_count: min_connection_count, interval: interval, extend_interval_later: extend_interval_later, extend_interval_earlier: extend_interval_earlier}
    }
}


// Return payload for IntermodalRoutingRequests
const getRoutingOptions = (startModes: Mode[], start: Station | Address, searchType: string, searchDirection: string, destinationModes: Mode[], destination: Station | Address, interval: Interval , min_connection_count: number, extend_interval_later: boolean, extend_interval_earlier: boolean) => {
    return {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({  destination: {type: 'Module', target: '/intermodal'}, 
                                content_type: 'IntermodalRoutingRequest',
                                content: {start_type: getStartType(start), start_modes: startModes, start: getStart(start, min_connection_count, interval, extend_interval_later, extend_interval_earlier) , search_type: searchType, search_dir: searchDirection, destination_type: getDestinationType(destination), destination_modes: destinationModes, destination: parseStationOrAddress(destination) } })
    };
};


// Helperfunction for Initialization of startModes and destinationModes State
const getModes = (key: string) => {
    let modes: ModeLocalStorage = getFromLocalStorage(key);
    let res = [];

    if (modes) {
        if (modes.walk.enabled) {
            res.push({ mode_type: 'FootPPR', mode: { search_options: { profile: modes.walk.search_profile.profile, duration_limit: modes.walk.search_profile.max_duration * 60 } }})
        }
        if (modes.bike.enabled) {
            res.push({ mode_type: 'Bike', mode: { max_duration: modes.bike.max_duration * 60 } })
        }
        if (modes.car.enabled) {
            if (modes.car.use_parking) {
                res.push({ mode_type: 'CarParking', mode: { max_car_duration: modes.car.max_duration * 60, ppr_search_options: { profile: 'default', duration_limit: 300 } } })
            }else {
                res.push({ mode_type: 'Car', mode: { max_duration: modes.car.max_duration * 60 } })
            }
        }
    }

    return res;
}


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
        dummyDay: dummyDate,
        new: ''
    };
}


// Helperfunction to manage IntermodalRoutingResponse when extend-earlier Button was clicked. Appends dummyEntries which will be used during ConnectionList rendering to determine daychanges.
const appendConnectionsAtHead = (setConnections: React.Dispatch<React.SetStateAction<Connection[]>>, allConnectionsWithoutDummies: Connection[], newConnections: Connection[], oldConnections: Connection[], setAllConnectionsWithoutDummies: React.Dispatch<React.SetStateAction<Connection[]>>, dateFormat: string, setLoading: React.Dispatch<React.SetStateAction<boolean>>) => {
    let dummyIndexes = [0];
    let newConnectionsWithDummies = [...newConnections];
    let followingConnectionDay = moment.unix(allConnectionsWithoutDummies[0].stops[0].departure.schedule_time);
    let dummyDays = [moment.unix(newConnections[0].stops[0].departure.schedule_time).format(dateFormat)];
    let connectionID = allConnectionsWithoutDummies.at(0).id - 1;
    setAllConnectionsWithoutDummies([...newConnections, ...allConnectionsWithoutDummies]);

    // Find all pairs of entries that start on different days.
    for (let i = 0; i < newConnections.length; i++){
        newConnectionsWithDummies[newConnections.length - 1 - i].id = connectionID - i;
        if (moment.unix(newConnections[newConnections.length - 1 - i].stops[0].departure.schedule_time).day() != followingConnectionDay.day()){
            dummyIndexes.push(newConnections.length - 1 - i);
            dummyDays.push(followingConnectionDay.format(dateFormat));
            followingConnectionDay.subtract(1, 'day');
        }
    };
    dummyIndexes.map((val, idx) => {
        newConnectionsWithDummies.splice(val + idx, 0, dummyConnection(dummyDays[idx]));
    });

    let tmp = oldConnections;
    tmp.splice(0, 1);

    console.log([...newConnectionsWithDummies, ...tmp]);
    setConnections([...newConnectionsWithDummies, ...tmp]);
    setLoading(false);
}


// Helperfunction to manage IntermodalRoutingResponse when extend-later Button was clicked. Appends dummyEntries which will be used during ConnectionList rendering to determine daychanges.
const appendConnectionsAtTail = (setConnections: React.Dispatch<React.SetStateAction<Connection[]>>, allConnectionsWithoutDummies: Connection[], newConnections: Connection[], oldConnections: Connection[], setAllConnectionsWithoutDummies: React.Dispatch<React.SetStateAction<Connection[]>>, dateFormat: string, setLoading: React.Dispatch<React.SetStateAction<boolean>>) => {
    let dummyIndexes = [];
    let newConnectionsWithDummies = [...newConnections];
    let previousConnectionDay = moment.unix(allConnectionsWithoutDummies.at(-1).stops[0].departure.schedule_time);
    let dummyDays = [];
    let connectionID = allConnectionsWithoutDummies.at(-1).id + 1;
    setAllConnectionsWithoutDummies([...allConnectionsWithoutDummies, ...newConnections]);

    // Find all pairs of entries that start on different days.
    for (let i = 0; i < newConnections.length; i++){
        newConnectionsWithDummies[i].id = connectionID + i;
        if (moment.unix(newConnections[i].stops[0].departure.schedule_time).day() != previousConnectionDay.day()){
            dummyIndexes.push(i);
            dummyDays.push(moment.unix(newConnections[i].stops[0].departure.schedule_time).format(dateFormat));
            previousConnectionDay.add(1, 'day');
        }
    };
    dummyIndexes.map((val, idx) => {
        newConnectionsWithDummies.splice(val + idx, 0, dummyConnection(dummyDays[idx]));
    });

    setConnections([...oldConnections, ...newConnectionsWithDummies]);
    setLoading(false);
}


// Helperfunction to manage IntermodalRoutingResponse. Appends dummyEntries which will be used during ConnectionList rendering to determine daychanges.
const sendConnectionsToOverlay = (setConnections: React.Dispatch<React.SetStateAction<Connection[]>>, connections: Connection[], setAllConnectionsWithoutDummies: React.Dispatch<React.SetStateAction<Connection[]>>, dateFormat: string, setLoading: React.Dispatch<React.SetStateAction<boolean>>) => {
    let dummyIndexes = [0];
    let connectionsWithDummies = [...connections];
    let previousConnectionDay = moment.unix(connections[0].stops[0].departure.schedule_time);
    let dummyDays = [previousConnectionDay.format(dateFormat)];
    setAllConnectionsWithoutDummies(connections);
    connectionsWithDummies[0].id = 0;
    // Find all pairs of entries that start on different days.
    for (let i = 1; i < connections.length; i++){
        connectionsWithDummies[i].id = i;
        if (moment.unix(connections[i].stops[0].departure.schedule_time).day() != previousConnectionDay.day()){
            dummyIndexes.push(i);
            dummyDays.push(moment.unix(connections[i].stops[0].departure.schedule_time).format(dateFormat));
            previousConnectionDay.add(1, 'day');
        }
    };
    // For every Day change, expand list of connections at the right index
    dummyIndexes.map((val, idx) => {
        connectionsWithDummies.splice(val + idx, 0, dummyConnection(dummyDays[idx]));
    });
    // Reset connections before sending new ones. This fixes a lot of bugs in Connectionrender where deprecated data would be used during rerendering
    setConnections([]);
    // Send new connections to Overlay
    setConnections(connectionsWithDummies);
    setLoading(false);
};


// If IntermodalRouting fails, set Connections to []
const handleErrors = (response, setLoading: React.Dispatch<React.SetStateAction<boolean>>, setConnections: React.Dispatch<React.SetStateAction<Connection[]>>) => {
    if (!response.ok) {
        setLoading(false);
        setConnections([]);
        throw Error(response.statusText);
    }
    return response;
}


export const Search: React.FC<SearchTypes> = (props) => {
    
    // StartModes
    const [startModes, setStartModes] = useState<Mode[]>(getModes('motis.routing.from_modes'));
    
    // Destination_modes tracks the ModePicker for the Destination
    const [destinationModes, setDestinationModes] = useState<Mode[]>(getModes('motis.routing.to_modes'));
    
    // searchTime
    // SearchTime stores the currently displayed Time
    const [searchTime, setSearchTime] = useState<string>(moment().format('HH:mm'));
    
    // searchTimeSelected manipulates the div 'gb-input-group' to highlight it if focused
    const [searchTimeSelected, setSearchTimeSelected] = useState<string>('');

    // Ref tracking if the searchTime Inputfield is focused
    const searchTimeRef = React.useRef(null);
    
    // SearchType
    const [searchType, setSearchType] = useState<string>('Accessibility');
    
    // SearchDirection
    const [searchDirection, setSearchDirection] = useState<string>('Forward');
    
    // Interval used to request earlier Connections
    const [searchBackward, setSearchBackward] = useState<Interval>(null);

    // Interval used to request later Connections
    const [searchForward, setSearchForward] = useState<Interval>(null);

    // Save currently displayed List of Connections. Will be extended with every fetch.
    const [allConnectionsWithoutDummies, setAllConnectionsWithoutDummies] = useState<Connection[]>([]);

    // This fetch is used to get a new set of connection data. Previous Connections will be discarded.
    const fetchNewRoutingData = () => {
        let requestURL = 'https://europe.motis-project.de/?elm=IntermodalConnectionRequest';

        fetch(requestURL, getRoutingOptions(startModes, props.start, searchType, searchDirection, destinationModes, props.destination, {begin: props.searchDate.unix(), end: props.searchDate.unix() + 3600 * 2}, 5, true, true))
                .then(res => handleErrors(res, props.setLoading, props.setConnections))
                .then(res => res.json())
                .then((res: IntermodalRoutingResponse) => {
                    // If IntermodalRoutingResponse is empty, stop Loadinganimation and show no-results
                    if(res.content.connections.length === 0){
                        props.setLoading(false);
                        props.setConnections(null)
                        return;
                    }
                    res.content.connections.map((c: Connection) => c.new = '');
                    sendConnectionsToOverlay(props.setConnections, res.content.connections, setAllConnectionsWithoutDummies, props.translation.dateFormat, props.setLoading);
                    setSearchBackward({begin: res.content.interval_begin - 3600 * 2, end: res.content.interval_begin - 1});
                    setSearchForward({begin: res.content.interval_end + 1, end: res.content.interval_end + 3600 * 2});
                    window.portEvents.pub('mapSetMarkers', {'startPosition': getFromLocalStorage('motis.routing.from_location').pos,
                                                            'startName': getFromLocalStorage('motis.routing.from_location').name,
                                                            'destinationPosition': getFromLocalStorage('motis.routing.to_location').pos,
                                                            'destinationName': getFromLocalStorage('motis.routing.to_location').name});
                    
                    window.portEvents.pub('mapSetConnections', {'mapId': 'map', 'connections': mapConnections(res.content.connections), 'lowestId': 0});
                })
                .catch(_error => {})
    };

    const handleFetch = useFetchPreventionOnDoubleClick(fetchNewRoutingData);


    // This Effect is one of 2 IntermodalConnectionRequest API Calls.
    // If this one is triggered, then we want to discard the currently shown Connections and load a new list
    useEffect(() => {
        if (props.searchDate) {
            let currDate = props.searchDate.unix();
            setSearchForward({begin: currDate, end: currDate + 3600 * 2});
            setSearchBackward({begin: currDate, end: currDate + 3600 * 2});
        }
        if (props.start !== null && props.destination !== null && props.searchDate && startModes && destinationModes) {
            props.setLoading(true);

            handleFetch();
        }
    }, [props.start, startModes, props.destination, destinationModes, searchDirection, props.searchDate]);


    // This Effect is one of 2 IntermodalConnectionRequest API Calls.
    // If this one is triggered, then we want to keep the currently shown Connections and add the newly fetched ones to this list
    useEffect(() => {
        if (props.start !== null && props.destination !== null && searchForward !== null && searchBackward !== null && (props.extendBackwardFlag || props.extendForwardFlag)) {
            let requestURL = 'https://europe.motis-project.de/?elm=IntermodalConnectionRequest';
            let searchIntv = props.extendBackwardFlag ? searchBackward : searchForward;

            fetch(requestURL, getRoutingOptions(startModes, props.start, searchType, searchDirection, destinationModes, props.destination, searchIntv, 3, props.extendBackwardFlag, props.extendForwardFlag))
                .then(res => handleErrors(res, props.setLoading, props.setConnections))
                .then(res => res.json())
                .then((res: IntermodalRoutingResponse) => {
                    // If IntermodalRoutingResponse is empty, stop Loadinganimation and keep oldConnections
                    if(res.content.connections.length === 0){
                        props.setExtendBackwardFlag(false);
                        props.setExtendForwardFlag(false);
                        return;
                    }
                    // Earlier Button was clicked
                    if(props.extendBackwardFlag){
                        // Update the connection list only if new connections were fetched. If the newly fetched data is identical to the currently displayed data, dont expand the displayed list
                        if(!equal(res.content.connections[0].stops, allConnectionsWithoutDummies[0].stops)) {
                            // All newly fetched Connections have a different classname than the rest
                            res.content.connections.map((c: Connection) => c.new = ' new');
                            allConnectionsWithoutDummies.map((c: Connection) => c.new = '');
                            appendConnectionsAtHead(props.setConnections, allConnectionsWithoutDummies, res.content.connections, props.connections, setAllConnectionsWithoutDummies, props.translation.dateFormat, props.setLoading);
                            // New Interval for searching even earlier connections
                            setSearchBackward({begin: res.content.interval_begin - 3600 * 2, end: res.content.interval_begin - 1});
                            window.portEvents.pub('mapSetConnections', {'mapId': 'map', 'connections': mapConnections([...res.content.connections, ...allConnectionsWithoutDummies]), 'lowestId': 0});
                        }
                        props.setExtendBackwardFlag(false);
                    } 
                    // Later Button was clicked
                    else {
                        // Update the connection list only if new connections were fetched. If the newly fetched data is identical to the currently displayed data, dont expand the displayed list
                        if(!equal(res.content.connections.at(-1).stops, allConnectionsWithoutDummies.at(-1).stops)) {
                            // All newly fetched Connections have a different classname than the rest
                            res.content.connections.map((c: Connection) => c.new = ' new');
                            allConnectionsWithoutDummies.map((c: Connection) => c.new = '');
                            appendConnectionsAtTail(props.setConnections, allConnectionsWithoutDummies, res.content.connections, props.connections, setAllConnectionsWithoutDummies, props.translation.dateFormat, props.setLoading);
                            // New Interval for searching even later connections
                            setSearchForward({begin: res.content.interval_end + 1, end: res.content.interval_end + 3600 * 2});
                            window.portEvents.pub('mapSetConnections', {'mapId': 'map', 'connections': mapConnections([...allConnectionsWithoutDummies, ...res.content.connections]), 'lowestId': 0});
                        }
                        props.setExtendForwardFlag(false);
                    }
                    window.portEvents.pub('mapSetMarkers', {'startPosition': getFromLocalStorage('motis.routing.from_location').pos,
                                                            'startName': getFromLocalStorage('motis.routing.from_location').name,
                                                            'destinationPosition': getFromLocalStorage('motis.routing.to_location').pos,
                                                            'destinationName': getFromLocalStorage('motis.routing.to_location').name});
                    
                })
                .catch(_error => {});
        }
    }, [props.extendForwardFlag, props.extendBackwardFlag]);


    useEffect(() => {
        window.portEvents.sub('mapSetMarkers', function(){
            if(markerSearch){
                if(markerSearch[0]){
                    props.setStart(markerSearch[1]);
                }else{
                    props.setDestination(markerSearch[1]);
                }
            }
        });
    });


    useOutsideAlerter(searchTimeRef, setSearchTimeSelected);


    return (
        <div id={`search${props.tripViewHidden ? '' : '-hidden'}`}>
            <div className='pure-g gutters'>
                <div className='pure-u-1 pure-u-sm-12-24 from-location'>
                    <div>
                        <SearchInputField   translation={props.translation}
                                        label={props.translation.search.start}
                                        station={props.start}
                                        setSearchDisplay={props.setStart}
                                        localStorageStation='motis.routing.from_location'/>
                        <Modepicker translation={props.translation} 
                                    title={props.translation.search.startTransports}
                                    setModes={setStartModes}
                                    localStorageModes='motis.routing.from_modes'
                                    modes={startModes}/>
                    </div>
                    <div className='swap-locations-btn'>
                        <label className='gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select'>
                            <input  type='checkbox' 
                                    onClick={() => {
                                        let swapStation = props.destination;
                                        props.setDestination(props.start);
                                        props.setStart(swapStation);
                            }}/>
                            <i className='icon'>swap_vert</i>
                        </label>
                    </div>
                </div>
                <div className='pure-u-1 pure-u-sm-12-24'>
                    <DatePicker translation={props.translation}
                                currentDate={props.searchDate}
                                setCurrentDate={props.setSearchDate}
                                scheduleInfo={props.scheduleInfo}/>
                </div>
            </div>
            <div className='pure-g gutters'>
                <div className='pure-u-1 pure-u-sm-12-24 to-location'>
                    <div>
                        <SearchInputField   translation={props.translation}
                                            label={props.translation.search.destination}
                                            station={props.destination}
                                            setSearchDisplay={props.setDestination}
                                            localStorageStation='motis.routing.to_location'/>
                        <Modepicker translation={props.translation} 
                                    title={props.translation.search.destinationTransports} 
                                    setModes={setDestinationModes}
                                    localStorageModes='motis.routing.to_modes'
                                    modes={destinationModes} />
                    </div>
                </div>
                <div className='pure-u-1 pure-u-sm-9-24'>
                    <div>
                        <div className='label'>{props.translation.search.time}</div>
                        <div className={`gb-input-group ${searchTimeSelected}`}>
                            <div className='gb-input-icon'><i className='icon'>schedule</i></div>
                            <input
                                className='gb-input'
                                ref={searchTimeRef}
                                tabIndex={4} 
                                value={searchTime}
                                onChange={(e) => {
                                    setSearchTime(e.currentTarget.value);
                                    if (e.currentTarget.value.split(':').length == 2) {
                                        let [hour, minute] = e.currentTarget.value.split(':');
                                        if (!isNaN(+hour) && !isNaN(+minute)){
                                            let newSearchTime = moment(props.searchDate);
                                            newSearchTime.hour(hour as unknown as number > 23 ? 23 : hour as unknown as number);
                                            newSearchTime.minute(minute as unknown as number > 59 ? 59 : minute as unknown as number);
                                            props.setSearchDate(newSearchTime);
                                }}}}
                                onKeyDown={(e) => {
                                    if (e.key == 'Enter'){
                                        setSearchTime(props.searchDate.format('HH:mm'));
                                    }
                                }}
                                onFocus={() => setSearchTimeSelected('gb-input-group-selected')}/>
                            <div className='gb-input-widget'>
                                <div className='hour-buttons'>
                                    <div><a
                                            className='gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select' 
                                            onClick={() => {
                                                let newSearchDate = props.searchDate.clone().subtract(1, 'h')
                                                props.setSearchDate(newSearchDate); 
                                                setSearchTime(newSearchDate.format('HH:mm'));}}>
                                            <i className='icon'>chevron_left</i></a></div>
                                    <div><a
                                            className='gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select' 
                                            onClick={() => {
                                                let newSearchDate = props.searchDate.clone().add(1, 'h')
                                                props.setSearchDate(newSearchDate);
                                                setSearchTime(newSearchDate.format('HH:mm'));}}>
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
                                onChange={(e) => setSearchDirection(e.currentTarget.value)}/>
                        <label htmlFor='search-forward'>{props.translation.search.departure}</label>
                    </form>
                    <form>
                        <input  type='radio' 
                                id='search-backward' 
                                name='time-option' 
                                value='Backward' 
                                checked={searchDirection === 'Backward'} 
                                onChange={(e) => setSearchDirection(e.currentTarget.value)}/>
                        <label htmlFor='search-backward'>{props.translation.search.arrival}</label>
                    </form>
                </div>
            </div>
        </div>
    )
};