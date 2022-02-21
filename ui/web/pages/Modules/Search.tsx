import React, { useEffect, useState } from 'react';

import moment from 'moment';

import { DatePicker } from './DatePicker';
import { Mode, IntermodalRoutingRequest, IntermodalRoutingResponse, IntermodalPretripStartInfo, PretripStartInfo } from './IntermodalRoutingTypes';
import { Connection, Station } from './ConnectionTypes';
import { Translations } from './Localization';
import { Address } from './SuggestionTypes';
import { SearchInputField } from './SearchInputField';
import { Modepicker } from './ModePicker';
import { getFromLocalStorage } from './LocalStorage';


const getRoutingOptions = (startType: string, startModes: Mode[], start: Station | Address, searchType: string, searchDirection: string, destinationType: string, destinationModes: Mode[], destination: Station | Address ) => {
    return {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({  destination: {type: "Module", target: "/intermodal"}, 
                                content_type: 'IntermodalRoutingRequest',
                                content: {start_type: startType, start_modes: startModes, start: { station: start, min_connection_count: 5, interval: { begin: 1644335520, end: 1644342720 }, extend_interval_later: true, extend_interval_earlier: true }, search_type: searchType, search_dir: searchDirection, destination_type: destinationType, destination_modes: destinationModes, destination: destination } })
    };
};


function addHours(date: Date, hours: number): Date {
    let res = new Date(date);
    res.setHours(res.getHours() + hours);
    return res;
};


export const Search: React.FC<{'setConnections': React.Dispatch<React.SetStateAction<Connection[]>>, 'translation': Translations}> = (props) => {

    // Boolean used as Communication between Inputfield and API Fetch
    const [searchQuery, setSearchQuery] = useState<boolean>(true);
    

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
    const [currentDate, setCurrentDate] = useState(moment());
    
    // SearchTime
    const [searchTime, setSearchTime] = useState<string>(currentDate.format('HH:mm'));
    
    // SearchType
    const [searchType, setSearchType] = useState<string>('Accessibility');
    
    // SearchDirection
    const [searchDirection, setSearchDirection] = useState<string>('Forward');

    
    useEffect(() => {
        if (start !== null && destination !== null) {
            let requestURL = 'https://europe.motis-project.de/?elm=IntermodalConnectionRequest';
            //console.log('Fire searchQuery')

            fetch(requestURL, getRoutingOptions(startType, startModes, start, searchType, searchDirection, destinationType, destinationModes, destination))
                .then(res => res.json())
                .then((res: IntermodalRoutingResponse) => {
                    console.log("Response came in");
                    console.log(res);
                    props.setConnections(res.content.connections);
                });
        }
    }, [searchQuery, startModes, destinationModes]);
    
    return (
        <div id='search'>
            <div className='pure-g gutters'>
                <div className='pure-u-1 pure-u-sm-12-24 from-location'>
                    <div>
                        <SearchInputField   translation={props.translation}
                                        label={props.translation.search.start}
                                        searchQuery={searchQuery}
                                        setSearchQuery={setSearchQuery}
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
                    <DatePicker translation={props.translation}/>
                </div>
            </div>
            <div className='pure-g gutters'>
                <div className='pure-u-1 pure-u-sm-12-24 to-location'>
                    <div>
                        <SearchInputField   translation={props.translation}
                                            label={props.translation.search.destination}
                                            searchQuery={searchQuery}
                                            setSearchQuery={setSearchQuery}
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
                                }}/>
                            <div className='gb-input-widget'>
                                <div className='hour-buttons'>
                                    <div><a
                                            className='gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select' onClick={() => {setCurrentDate(currentDate.subtract(1, 'h')); setSearchTime(currentDate.format('HH:mm'))}}><i
                                                className='icon'>chevron_left</i></a></div>
                                    <div><a
                                            className='gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select' onClick={() => {setCurrentDate(currentDate.add(1, 'h')); setSearchTime(currentDate.format('HH:mm'))}}><i
                                                className='icon'>chevron_right</i></a></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div className='pure-u-1 pure-u-sm-3-24 time-option'>
                    <form>
                        <input type='radio' id='search-forward' name='time-option' defaultChecked={searchDirection === 'Forward'} onClick={() => setSearchDirection('Forward')}/>
                        <label htmlFor='search-forward'>{props.translation.search.departure}</label>
                    </form>
                    <form>
                        <input type='radio' id='search-backward' name='time-option' defaultChecked={searchDirection === 'Backward'} onClick={() => setSearchDirection('Backward')}/>
                        <label htmlFor='search-backward'>{props.translation.search.arrival}</label>
                    </form>
                </div>
            </div>
        </div>
    )
};