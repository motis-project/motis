import React, { useEffect, useState } from 'react';

import { Modepicker } from './ModePicker';
import { DatePicker } from './DatePicker';
import { IntermodalRoutingRequest, IntermodalRoutingResponse, IntermodalPretripStartInfo, PretripStartInfo } from './IntermodalRoutingTypes';
import { Connection, Station } from './ConnectionTypes';
import { Mode } from './ModePicker';
import { Interval } from './RoutingTypes';
import { Translations } from './Localization';
import { AddressSuggestionResponse, Address, StationSuggestionResponse } from './SuggestionTypes';
import { Proposals } from './Proposals';


interface Destination {
    name: string,
    id: string
}


interface Start {
    station: Station,
    min_connection_count: number,
    interval: Interval,
    extend_interval_later: boolean,
    extend_interval_earlier: boolean
}


const getRoutingOptions = (startType: string, startModes: Mode[], start: string, searchType: string, searchDirection: string, destinationType: string, destinationModes: Mode[], destination: string ) => {
    return {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({  destination: {type: "Module", target: "/intermodal"}, 
                                content_type: 'IntermodalRoutingRequest',
                                content: {start_type: startType, start_modes: startModes, start: { station: { name: start, id: 'delfi_de:06411:4734:64:63'}, min_connection_count: 5, interval: { begin: 1640951760, end: 1640958960 }, extend_interval_later: true, extend_interval_earlier: true }, search_type: searchType, search_dir: searchDirection, destination_type: destinationType, destination_modes: destinationModes, destination: {name: destination, id: 'delfi_de:06412:1204:3:3' }} })
    }
}


const getPostRequest = (body: any) => {
    return {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify( body )
    }
}


const getDefaultMode = () => {
    return {mode_type: '', mode: { search_options: { profile: 'default', duration_limit: 30 } } }
}


function addHours(date: Date, hours: number): Date {
    let res = new Date(date);
    res.setHours(res.getHours() + hours);
    return res;
}


const fetchSuggestions = (input: string, setAddresses: React.Dispatch<React.SetStateAction<Address[]>>, setStations: React.Dispatch<React.SetStateAction<Station[]>>) => {
    let requestURL = 'https://europe.motis-project.de/?elm=AddressSuggestions'

    let body = {
        destination: { type: 'Module', target: '/address'},
        content_type: 'AddressRequest',
        content: { input: input }
    }

    fetch(requestURL, getPostRequest(body))
            .then(res => res.json())
            .then((res: AddressSuggestionResponse) => {
                console.log("Response came in");
                console.log(res);
                setAddresses(res.content.guesses)
            })

    requestURL = 'https://europe.motis-project.de/?elm=StationSuggestions'

    let body2 = {
        destination: { type: 'Module', target: '/guesser'},
        content_type: 'StationGuesserRequest',
        content: { guess_count: 6, input: input }
    }

    fetch(requestURL, getPostRequest(body))
            .then(res => res.json())
            .then((res: StationSuggestionResponse) => {
                console.log("Response came in");
                console.log(res);
                setStations(res.content.guesses)
            })
}


export const Search: React.FC<{'setConnections': React.Dispatch<React.SetStateAction<Connection[]>>, 'translation': Translations}> = (props) => {

    const [searchQuery, setSearchQuery] = useState<boolean>(true);
    
    // StartType
    const [startType, setStartType] = useState<string>('PretripStart');
    
    // StartModes
    const [startModes, setStartModes] = useState<Mode[]>([]);
    
    // Start
    const [start, setStart] = useState<string>('Darmstadt Hauptbahnhof')//<Start>({ station: { name: 'Darmstadt Hauptbahnhof', id: 'delfi_de:06411:4734:64:63'}, min_connection_count: 5, interval: { begin: 1640430180, end: 164043738 }, extend_interval_later: true, extend_interval_earlier: true });
    
    // SearchType
    const [searchType, setSearchType] = useState<string>('Accessibility');
    
    // SearchDirection
    const [searchDirection, setSearchDirection] = useState<string>('Forward');
    
    // DestinationType
    const [destinationType, setDestinationType] = useState<string>('InputStation');
    
    // Destination_modes tracks the ModePicker for the Destination
    const [destinationModes, setDestinationModes] = useState<Mode[]>([]);
    
    // Destination holds the Value of 'to location' input field
    const [destination, setDestination] = useState<string>("Frankfurt (Main) Westbahnhof")//<Destination>({name: 'Frankfurt (Main) Westbahnhof', id: 'delfi_D_de:06412:1204' });

    // Current Date
    const [currentDate, setCurrentDate] = useState<Date>(new Date());

    // SearchTime
    const [searchTime, setSearchTime] = useState<Date>(new Date());

    // show Start Suggestions
    const [showStartSuggestions, setShowStartSuggestions] = useState<boolean>(false);

    // fetch Start Address Suggestions
    const [fetchStartSuggestions, setFetchStartSuggestions] = useState<boolean>(false);

    // Start Address Suggestions
    const [startAddressSuggestions, setStartAddressSuggestions] = useState<Address[]>([]);

    // Start Station Suggestions
    const [startStationSuggestions, setStartStationSuggestions] = useState<Station[]>([]);

    // fetch Destination Address Suggestions
    const [fetchDestinationAddressSuggestions, setFetchDestinationAddressSuggestions] = useState<boolean>(false);

    // Destination Address Suggestions
    const [destinationAddressSuggestions, setDestinationAddressSuggestions] = useState<Address[]>([]);

    // Destination Station Suggestions
    const [destinationStationSuggestions, setDestinationStationSuggestions] = useState<Station[]>([]);

    // show Destination Suggestions
    const [showDestinationSuggestions, setShowDestinationSuggestions] = useState<boolean>(false);
    
    useEffect(() => {
        let requestURL = 'https://europe.motis-project.de/?elm=IntermodalConnectionRequest'
        //console.log('Fire searchQuery')

        fetch(requestURL, getRoutingOptions(startType, startModes, start, searchType, searchDirection, destinationType, destinationModes, destination))
            .then(res => res.json())
            .then((res: IntermodalRoutingResponse) => {
                console.log("Response came in");
                console.log(res)
                props.setConnections(res.content.connections)
            })

    }, [searchQuery]);

    //
    useEffect(() => {
        fetchSuggestions(start, setStartAddressSuggestions, setStartStationSuggestions);
    }, [fetchStartSuggestions])
    
    useEffect(() => {
        fetchSuggestions(destination, setDestinationAddressSuggestions, setDestinationStationSuggestions)
    }, [fetchDestinationAddressSuggestions])
    
    return (
        <div id='search'>
            <div className='pure-g gutters'>
                <div className='pure-u-1 pure-u-sm-12-24 from-location'>
                    <div>
                        <form>
                            <div className='label'>
                                {props.translation.search.start}
                            </div>
                            <div className='gb-input-group'>
                                <div className='gb-input-icon'>
                                    <i className='icon'>place</i>
                                    </div>
                            <input  className='gb-input' tabIndex={1} value={start} 
                                    onChange={e => {
                                        //e.preventDefault();
                                        //console.log("Start changed")
                                        setStart(e.currentTarget.value)
                                        if (e.currentTarget.value.length >= 3) {
                                            setFetchStartSuggestions(!fetchStartSuggestions);
                                        }
                                    } }
                                    onKeyPress={e => {
                                        if (e.key == "Enter") {
                                            e.preventDefault();
                                            console.log("Pressed Enter in Start")
                                            setSearchQuery(!searchQuery)
                                        }
                                    } }
                                    onFocus={_ => {
                                        setShowStartSuggestions(true);
                                    } }
                                    onBlur={_ => {
                                        setShowStartSuggestions(false);
                                    } } /></div>
                        </form>
                        <div className='paper' style={showStartSuggestions && startAddressSuggestions.length > 0 ? {} : {display: 'none'}}>
                            <Proposals addresses={startAddressSuggestions} stations={startStationSuggestions}/>
                        </div>
                    </div>
                    <Modepicker translation={props.translation} start={true}/>{/* modes={startModes} setModes={setStartModes}/>*/}
                    <div className='swap-locations-btn'>
                        <label className='gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select'>
                            <input  type='checkbox' 
                                    onClick={() => {
                                        let tmp = destination;
                                        setDestination(start);
                                        setStart(tmp);
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
                        <div>
                            <div className='label'>{props.translation.search.destination}</div>
                            <div className='gb-input-group'>
                                <div className='gb-input-icon'><i className='icon'>place</i></div>
                                <input  className='gb-input' tabIndex={2} value={destination}
                                        onChange={e => {
                                            //e.preventDefault();
                                            //console.log("Start changed")
                                            setDestination(e.currentTarget.value)
                                            if (e.currentTarget.value.length >= 3) {
                                                setFetchDestinationAddressSuggestions(!fetchDestinationAddressSuggestions);
                                            }
                                        } }
                                        onKeyPress={e => {
                                            if (e.key == "Enter") {
                                                e.preventDefault();
                                                console.log("Pressed Enter in Destination")
                                                setSearchQuery(!searchQuery)
                                            }
                                        } }
                                        onFocus={_ => {
                                            setShowDestinationSuggestions(true);
                                        } }
                                        onBlur={_ => {
                                            setShowDestinationSuggestions(false);
                                        } }/>
                            </div>
                        </div>
                        <div className='paper' style={showDestinationSuggestions && destinationAddressSuggestions.length > 0 ? {} : {display: 'none'}}>
                            <Proposals addresses={destinationAddressSuggestions} stations={destinationStationSuggestions}/>
                        </div>
                    </div>
                    <Modepicker translation={props.translation} start={false}/>{/*} modes={destinationModes} setModes={setDestinationModes}/>*/}
                </div> 
                <div className='pure-u-1 pure-u-sm-9-24'>
                    <div>
                        <div className='label'>{props.translation.search.time}</div>
                        <div className='gb-input-group'>
                            <div className='gb-input-icon'><i className='icon'>schedule</i></div>
                            <input
                                className='gb-input' tabIndex={4} value={searchTime.getHours() + ':' + ('0' + searchTime.getMinutes()).slice(-2)} />
                            <div className='gb-input-widget'>
                                <div className='hour-buttons'>
                                    <div><a
                                            className='gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select' onClick={() => setSearchTime(addHours(searchTime, -1))}><i
                                                className='icon'>chevron_left</i></a></div>
                                    <div><a
                                            className='gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select' onClick={() => setSearchTime(addHours(searchTime, 1))}><i
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
}