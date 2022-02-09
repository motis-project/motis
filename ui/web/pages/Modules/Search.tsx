import React, { useEffect, useState } from 'react';

import { Modepicker } from './ModePicker';
import { DatePicker } from './DatePicker';
import { IntermodalRoutingRequest, IntermodalRoutingResponse, IntermodalPretripStartInfo, PretripStartInfo } from './IntermodalRoutingTypes';
import { Connection, Station } from './ConnectionTypes';
import { Mode } from './ModePicker';
import { Interval } from './RoutingTypes';


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
                                content: {start_type: startType, start_modes: startModes, start: { station: { name: start, id: 'delfi_de:06411:4734:64:63'}, min_connection_count: 5, interval: { begin: 1644335520, end: 1644342720 }, extend_interval_later: true, extend_interval_earlier: true }, search_type: searchType, search_dir: searchDirection, destination_type: destinationType, destination_modes: destinationModes, destination: {name: destination, id: 'delfi_de:06412:1204:3:3' }} })
    }
}


const getDefaultMode = () => {
    return {mode_type: '', mode: { search_options: { profile: 'default', duration_limit: 30 } } }
}


export const Search: React.FC<{'setConnections': React.Dispatch<React.SetStateAction<Connection[]>>}> = (props) => {

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
    
    return (
        <div id='search'>
            <div className='pure-g gutters'>
                <div className='pure-u-1 pure-u-sm-12-24 from-location'>
                    <div>
                        <form>
                            <div className='label'>
                                Start
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
                                    } }
                                    onKeyPress={e => {
                                        if (e.key == "Enter") {
                                            e.preventDefault();
                                            console.log("Pressed Enter in Start")
                                            setSearchQuery(!searchQuery)
                                        }
                                    } } /></div>
                        </form>
                        <div className='paper hide'>
                            <ul className='proposals'></ul>
                        </div>
                    </div>
                    <Modepicker start={true}/>{/* modes={startModes} setModes={setStartModes}/>*/}
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
                    <DatePicker />
                </div>
            </div>
            <div className='pure-g gutters'>
                <div className='pure-u-1 pure-u-sm-12-24 to-location'>
                    <div>
                        <div>
                            <div className='label'>Ziel</div>
                            <div className='gb-input-group'>
                                <div className='gb-input-icon'><i className='icon'>place</i></div>
                                <input  className='gb-input' tabIndex={2} value={destination}
                                        onChange={e => {
                                            //e.preventDefault();
                                            //console.log("Start changed")
                                            setDestination(e.currentTarget.value)
                                        } }
                                        onKeyPress={e => {
                                            if (e.key == "Enter") {
                                                e.preventDefault();
                                                console.log("Pressed Enter in Destination")
                                                setSearchQuery(!searchQuery)
                                            }
                                        } }/>
                            </div>
                        </div>
                        <div className='paper hide'>
                            <ul className='proposals'></ul>
                        </div>
                    </div>
                    <Modepicker start={false}/>{/*} modes={destinationModes} setModes={setDestinationModes}/>*/}
                </div> 
                <div className='pure-u-1 pure-u-sm-9-24'>
                    <div>
                        <div className='label'>Uhrzeit</div>
                        <div className='gb-input-group'>
                            <div className='gb-input-icon'><i className='icon'>schedule</i></div><input
                                className='gb-input' tabIndex={4} />
                            <div className='gb-input-widget'>
                                <div className='hour-buttons'>
                                    <div><a
                                            className='gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select'><i
                                                className='icon'>chevron_left</i></a></div>
                                    <div><a
                                            className='gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select'><i
                                                className='icon'>chevron_right</i></a></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div className='pure-u-1 pure-u-sm-3-24 time-option'>
                    <form>
                        <input type='radio' id='search-forward' name='time-option' defaultChecked={searchDirection === 'Forward'} onClick={() => setSearchDirection('Forward')}/>
                        <label htmlFor='search-forward'>Abfahrt</label>
                    </form>
                    <form>
                        <input type='radio' id='search-backward' name='time-option' defaultChecked={searchDirection === 'Backward'} onClick={() => setSearchDirection('Backward')}/>
                        <label htmlFor='search-backward'>Ankunft</label>
                    </form>
                </div>
            </div>
        </div>
    )
}