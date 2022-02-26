import React, { useEffect, useState } from 'react';

import moment from 'moment';

import { DatePicker } from './DatePicker';
import { Mode, IntermodalRoutingResponse } from './IntermodalRoutingTypes';
import { Connection, Position, Station } from './ConnectionTypes';
import { Translations } from './Localization';
import { Address } from './SuggestionTypes';
import { SearchInputField } from './SearchInputField';
import { Modepicker } from './ModePicker';
import { getFromLocalStorage, setLocalStorage } from './LocalStorage';
import { Interval } from './RoutingTypes';
import { markerSearch } from './RailvizContextMenu';


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
            cons.push({'id': i, 'stations': stations, 'trains': [], 'walks': []});
        }
    }
    return cons;
};

export const Search: React.FC<{'setConnections': React.Dispatch<React.SetStateAction<Connection[]>>, 'translation': Translations}> = (props) => {
 
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
    const [searchDate, setSearchDate] = useState<moment.Moment>(moment());
    
    // SearchTime
    const [searchTime, setSearchTime] = useState<string>(moment().format('HH:mm'));
    
    // SearchType
    const [searchType, setSearchType] = useState<string>('Accessibility');
    
    // SearchDirection
    const [searchDirection, setSearchDirection] = useState<string>('Forward');
    
    useEffect(() => {
        if (start !== null && destination !== null) {
            props.setConnections(null);
            let requestURL = 'https://europe.motis-project.de/?elm=IntermodalConnectionRequest';
            //console.log('Fire searchQuery')

            //let interval = {begin: searchDate.unix(), end: searchDate.unix() + 7200};
            //console.log(searchDate.format('LLLL'))

            fetch(requestURL, getRoutingOptions(startType, startModes, start, searchType, searchDirection, destinationType, destinationModes, destination, {begin: searchDate.unix(), end: searchDate.unix() + 7200}))
                .then(res => res.json())
                .then((res: IntermodalRoutingResponse) => {
                    console.log("Response came in");
                    console.log(res);
                    props.setConnections(res.content.connections);
                    window.portEvents.pub('mapSetMarkers', {'startPosition': getFromLocalStorage("motis.routing.from_location").pos,
                                                            'startName': getFromLocalStorage("motis.routing.from_location").name,
                                                            'destinationPosition': getFromLocalStorage("motis.routing.to_location").pos,
                                                            'destinationName': getFromLocalStorage("motis.routing.to_location").name});
                    
                    window.portEvents.pub('mapSetConnections', {'mapId': 'map', 'connections': mapConnections(res.content.connections), 'lowestId': 0});
                });
        }
    }, [start, startModes, destination, destinationModes, searchDirection]);

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

    useEffect(() => {
        console.log("UseEffect Trigger on searchDate. WARUM TRIGGERST DU NICHT??? >:V")
    }, [searchDate]);


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
                                            className='gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select' onClick={() => {setSearchDate(searchDate.subtract(1, 'h')); setSearchTime(searchDate.format('HH:mm'))}}><i
                                                className='icon'>chevron_left</i></a></div>
                                    <div><a
                                            className='gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select' onClick={() => {setSearchDate(searchDate.add(1, 'h')); setSearchTime(searchDate.format('HH:mm'))}}><i
                                                className='icon'>chevron_right</i></a></div>
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